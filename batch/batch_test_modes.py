from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm

from ..config import AIModelConfig
from ..data_process import DatabaseBuilder
from ..model import ReconstructionTrainer, predict_and_compare
from ..data_process import parse_preprocess_steps

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _to_jsonable_dict(data: dict[str, Any]) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, Path):
            converted[key] = str(value)
        else:
            converted[key] = value
    return converted


def _write_manifest(path: Path, dataset_name: str, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"dataset_name": dataset_name, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _absolutize_record_paths(records: list[dict[str, Any]], database_dir: Path) -> list[dict[str, Any]]:
    """将样本中的相对 npy 路径转换为绝对路径，避免 split manifest 目录变化导致找不到文件。"""
    converted: list[dict[str, Any]] = []
    for record in records:
        item = dict(record)
        item["waveform_path"] = str((database_dir / item["waveform_path"]).resolve())
        item["field_path"] = str((database_dir / item["field_path"]).resolve())
        converted.append(item)
    return converted


def _split_train_test(
    records: list[dict[str, Any]],
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []
    rng = np.random.default_rng(seed)
    indices = np.arange(len(records))
    rng.shuffle(indices)
    test_count = max(1, int(len(records) * test_ratio))
    test_indices = set(indices[:test_count].tolist())
    train_records = [records[i] for i in range(len(records)) if i not in test_indices]
    test_records = [records[i] for i in range(len(records)) if i in test_indices]
    return train_records, test_records


def _build_base_config(run_dir: Path, device: str, epochs: int, seed: int) -> AIModelConfig:
    cfg = AIModelConfig()
    # 将批量测试产物限定在本次 run 目录，避免污染默认 data/result。
    cfg.data_root = run_dir / "database"
    cfg.result_root = run_dir / "result"
    cfg.device = device
    cfg.epochs = epochs
    cfg.random_seed = seed
    cfg.ensure_dirs()
    return cfg


def _resolve_device(device_name: str) -> torch.device:
    normalized = str(device_name).strip().lower()
    if normalized in {"gpu", "cuda", "cuda:0"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def _plot_summary(summary_rows: list[dict[str, Any]], output_path: Path) -> str:
    if plt is None:
        return "matplotlib 不可用，已跳过绘图。"

    mode_names = [str(row["mode_name"]) for row in summary_rows]
    total_loss = [float(row["test_total_loss"]) for row in summary_rows]
    temp_mae = [float(row["test_temperature_mae"]) for row in summary_rows]
    train_seconds = [float(row["train_seconds"]) for row in summary_rows]
    residual_loss = [float(row["test_physics_residual_loss"]) for row in summary_rows]

    plt.figure(figsize=(12, 8))
    axes = [plt.subplot(2, 2, i + 1) for i in range(4)]
    metrics = [
        ("Test Total Loss", total_loss),
        ("Test Temperature MAE", temp_mae),
        ("Train Seconds", train_seconds),
        ("Test Physics Residual", residual_loss),
    ]
    for ax, (title, values) in zip(axes, metrics):
        ax.bar(mode_names, values)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return "ok"


def _load_existing_summary(summary_csv: Path) -> list[dict[str, Any]]:
    if not summary_csv.exists():
        return []
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [dict(row) for row in reader]
    # Convert numeric fields back when possible.
    numeric_keys = {
        "physics_residual_weight",
        "train_seconds",
        "test_total_loss",
        "test_field_loss",
        "test_acoustic_loss",
        "test_temperature_loss",
        "test_smoothness_loss",
        "test_physics_residual_loss",
        "test_temperature_mae",
        "test_temperature_rmse",
        "test_samples",
    }
    for row in rows:
        for key in numeric_keys:
            if key in row and row[key] not in {"", None}:
                try:
                    row[key] = float(row[key])
                except ValueError:
                    pass
        if "learnable_branch_weights" in row:
            row["learnable_branch_weights"] = str(row["learnable_branch_weights"]).lower() == "true"
    return rows


def _collect_mse_records(
    builder: DatabaseBuilder,
    data_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    manifest_10 = builder.import_experimental_csvs(
        source_dir=data_root / "10times",
        material_key="metal_matrix",
        output_name="mse_10times_manifest.json",
    )
    manifest_mm = builder.import_experimental_csvs(
        source_dir=data_root / "data_multimaterial",
        material_key="carbon_silicon",
        output_name="mse_multimaterial_manifest.json",
    )
    records_10 = json.loads(manifest_10.read_text(encoding="utf-8")).get("records", [])
    records_mm = json.loads(manifest_mm.read_text(encoding="utf-8")).get("records", [])
    records = records_10 + records_mm
    stats = {
        "mse_10times_records": len(records_10),
        "mse_multimaterial_records": len(records_mm),
        "mse_total_records": len(records),
    }
    return records, stats


def _evaluate_on_manifest(
    cfg: AIModelConfig,
    checkpoint_path: Path,
    test_manifest_path: Path,
    artifacts_dir: Path | None = None,
    num_field_samples: int = 6,
) -> dict[str, float | str]:
    """对测试 manifest 运行推理，并在 ``artifacts_dir`` 产出样本级预测/真实对比产物。"""

    if artifacts_dir is None:
        artifacts_dir = checkpoint_path.parent / f"{checkpoint_path.stem}_predictions"

    metrics = predict_and_compare(
        cfg=cfg,
        checkpoint_path=checkpoint_path,
        manifest_path=test_manifest_path,
        output_dir=artifacts_dir,
        compute_loss=True,
        enable_plots=num_field_samples > 0,
        num_field_samples=num_field_samples,
    )

    if metrics.get("samples", 0) == 0:
        return {
            "test_total_loss": float("nan"),
            "test_field_loss": float("nan"),
            "test_acoustic_loss": float("nan"),
            "test_temperature_loss": float("nan"),
            "test_smoothness_loss": float("nan"),
            "test_physics_residual_loss": float("nan"),
            "test_temperature_mae": float("nan"),
            "test_temperature_rmse": float("nan"),
            "test_samples": 0.0,
            "test_artifacts_dir": str(artifacts_dir),
        }

    return {
        "test_total_loss": float(metrics.get("total_loss", float("nan"))),
        "test_field_loss": float(metrics.get("field_loss", float("nan"))),
        "test_acoustic_loss": float(metrics.get("acoustic_loss", float("nan"))),
        "test_temperature_loss": float(metrics.get("temperature_loss", float("nan"))),
        "test_smoothness_loss": float(metrics.get("smoothness_loss", float("nan"))),
        "test_physics_residual_loss": float(metrics.get("physics_residual_loss", float("nan"))),
        "test_temperature_mae": float(metrics["temperature_mae"]),
        "test_temperature_rmse": float(metrics["temperature_rmse"]),
        "test_samples": float(metrics["samples"]),
        "test_artifacts_dir": str(metrics["artifacts_dir"]),
    }


def run_batch_test(
    data_root: Path,
    result_root: Path,
    test_ratio: float,
    epochs: int,
    device: str,
    seed: int,
    physics_residual_weight: float,
    preprocess_steps: list[str] | tuple[str, ...] | None = None,
    clip_quantile: float = 1.0,
    smooth_window: int = 11,
    existing_run_dir: Path | None = None,
    resume: bool = False,
) -> Path:
    run_name = datetime.now().strftime("batch_test_%Y%m%d_%H%M%S")
    run_dir = existing_run_dir if existing_run_dir is not None else (result_root / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _build_base_config(run_dir, device=device, epochs=epochs, seed=seed)
    base_cfg.preprocess_steps = ",".join(preprocess_steps or [])
    base_cfg.clip_quantile = float(clip_quantile)
    base_cfg.smooth_window = int(smooth_window)
    builder = DatabaseBuilder(base_cfg)

    all_records, import_stats = _collect_mse_records(
        builder,
        data_root=data_root,
    )
    if len(all_records) < 10:
        raise RuntimeError(
            f"可用样本过少（{len(all_records)}），不足以做稳定训练/测试。"
        )

    train_records, test_records = _split_train_test(all_records, test_ratio=test_ratio, seed=seed)
    train_records = _absolutize_record_paths(train_records, base_cfg.database_dir)
    test_records = _absolutize_record_paths(test_records, base_cfg.database_dir)
    train_manifest = run_dir / "manifests" / "mse_train_manifest.json"
    test_manifest = run_dir / "manifests" / "mse_test_manifest.json"
    _write_manifest(train_manifest, "mse_train_split", train_records)
    _write_manifest(test_manifest, "mse_test_split", test_records)

    modes = [
        {"name": "normal_fixed", "training_mode": "normal", "learnable_branch_weights": False},
        {"name": "normal_learnable", "training_mode": "normal", "learnable_branch_weights": True},
        {"name": "residual_pinn_fixed", "training_mode": "residual_pinn", "learnable_branch_weights": False},
        {"name": "residual_pinn_learnable", "training_mode": "residual_pinn", "learnable_branch_weights": True},
    ]

    summary_csv = run_dir / "summary.csv"
    summary_rows: list[dict[str, Any]] = _load_existing_summary(summary_csv) if resume else []
    completed_mode_names = {str(item.get("mode_name", "")) for item in summary_rows}
    mode_bar = tqdm(modes, desc="batch-test-modes", unit="mode")
    for mode in mode_bar:
        mode_dir = run_dir / mode["name"]
        mode_bar.set_postfix(mode=mode["name"])
        if mode["name"] in completed_mode_names:
            continue
        mode_cfg = _build_base_config(mode_dir, device=device, epochs=epochs, seed=seed)
        mode_cfg.training_mode = mode["training_mode"]
        mode_cfg.learnable_branch_weights = bool(mode["learnable_branch_weights"])
        mode_cfg.physics_residual_weight = physics_residual_weight
        mode_cfg.preprocess_steps = base_cfg.preprocess_steps
        mode_cfg.clip_quantile = base_cfg.clip_quantile
        mode_cfg.smooth_window = base_cfg.smooth_window
        mode_cfg.ensure_dirs()

        trainer = ReconstructionTrainer(mode_cfg)
        t0 = time.perf_counter()
        checkpoint = trainer.train(
            manifest_path=train_manifest,
            checkpoint_name=f"{mode['name']}.pt",
        )
        train_seconds = time.perf_counter() - t0
        predictions_dir = mode_dir / "predictions"
        test_metrics = _evaluate_on_manifest(
            mode_cfg,
            checkpoint_path=checkpoint,
            test_manifest_path=test_manifest,
            artifacts_dir=predictions_dir,
        )

        row = {
            "mode_name": mode["name"],
            "training_mode": mode_cfg.training_mode,
            "learnable_branch_weights": mode_cfg.learnable_branch_weights,
            "physics_residual_weight": mode_cfg.physics_residual_weight,
            "train_seconds": train_seconds,
            "checkpoint": str(checkpoint),
            **test_metrics,
        }
        summary_rows.append(row)
        # 每完成一个 mode 立即落盘，便于断点续跑。
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)

    if not summary_rows:
        raise RuntimeError("没有可用的模式结果可汇总，请检查数据与配置。")

    plot_path = run_dir / "compare_plot.png"
    plot_status = _plot_summary(summary_rows, plot_path)

    report = {
        "run_dir": str(run_dir),
        "data_root": str(data_root),
        "import_stats": import_stats,
        "split": {
            "train_samples": len(train_records),
            "test_samples": len(test_records),
            "test_ratio": test_ratio,
            "seed": seed,
        },
        "base_config": _to_jsonable_dict(asdict(base_cfg)),
        "summary": summary_rows,
        "plot_path": str(plot_path),
        "plot_status": plot_status,
    }
    report_json = run_dir / "report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在 --data-root（默认 database/raw）下的 10times / data_multimaterial 上批量测试各模式"
    )
    parser.add_argument("--data-root", type=str, default="database/raw")
    parser.add_argument("--result-root", type=str, default="result")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics-residual-weight", type=float, default=0.1)
    parser.add_argument(
        "--preprocess",
        type=str,
        default="",
        help="实验波形预处理流水线，逗号分隔: clip,smooth,detrend,robust_norm,zscore",
    )
    parser.add_argument("--clip-quantile", type=float, default=1.0)
    parser.add_argument("--smooth-window", type=int, default=11)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    data_root = Path(args.data_root)
    result_root = Path(args.result_root)
    if not data_root.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_root}")
    if not (0.0 < args.test_ratio < 1.0):
        raise ValueError("--test-ratio 必须在 (0, 1) 范围内")
    if args.epochs <= 0:
        raise ValueError("--epochs 必须大于 0")
    preprocess_steps = parse_preprocess_steps(args.preprocess)
    run_dir = run_batch_test(
        data_root=data_root,
        result_root=result_root,
        test_ratio=args.test_ratio,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        physics_residual_weight=args.physics_residual_weight,
        preprocess_steps=preprocess_steps,
        clip_quantile=args.clip_quantile,
        smooth_window=args.smooth_window,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
