from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from .batch_test_modes import (
    _absolutize_record_paths,
    _build_base_config,
    _collect_mse_records,
    _evaluate_on_manifest,
    _split_train_test,
    _to_jsonable_dict,
    _write_manifest,
)
from ..config import AIModelConfig
from ..data_process import parse_preprocess_steps
from ..data_process import DatabaseBuilder
from ..model import ReconstructionTrainer


def _parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in str(raw).split(",")]
    parsed = [float(item) for item in values if item]
    if not parsed:
        raise ValueError("权重列表不能为空")
    return parsed


def _write_search_outputs(
    run_dir: Path,
    base_cfg: AIModelConfig,
    data_root: Path,
    import_stats: dict[str, int],
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    test_ratio: float,
    seed: int,
    training_mode: str,
    physics_residual_weight: float,
    network_weights: list[float],
    physical_weights: list[float],
    summary_rows: list[dict[str, Any]],
) -> None:
    if not summary_rows:
        return

    ordered_rows = sorted(summary_rows, key=lambda item: (float(item["test_temperature_mae"]), float(item["test_total_loss"])))
    summary_csv = run_dir / "summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ordered_rows[0].keys()))
        writer.writeheader()
        writer.writerows(ordered_rows)

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
        "search_space": {
            "training_mode": training_mode,
            "physics_residual_weight": physics_residual_weight,
            "network_weights": network_weights,
            "physical_weights": physical_weights,
        },
        "completed_cases": len(ordered_rows),
        "summary": ordered_rows,
    }
    report_json = run_dir / "report.json"
    report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_search_config(
    run_dir: Path,
    device: str,
    epochs: int,
    seed: int,
    training_mode: str,
    physics_residual_weight: float,
    network_weight: float,
    physical_weight: float,
) -> AIModelConfig:
    cfg = _build_base_config(run_dir, device=device, epochs=epochs, seed=seed)
    cfg.training_mode = training_mode
    cfg.learnable_branch_weights = False
    cfg.physics_residual_weight = physics_residual_weight
    cfg.fixed_weight_cnn = network_weight
    cfg.fixed_weight_lstm = network_weight
    cfg.fixed_weight_material = physical_weight
    cfg.fixed_weight_dimension = physical_weight
    cfg.fixed_weight_mode = physical_weight
    cfg.ensure_dirs()
    return cfg


def run_fixed_weight_search(
    data_root: Path,
    result_root: Path,
    test_ratio: float,
    epochs: int,
    device: str,
    seed: int,
    training_mode: str,
    physics_residual_weight: float,
    network_weights: list[float],
    physical_weights: list[float],
    preprocess_steps: list[str] | tuple[str, ...] | None = None,
    clip_quantile: float = 1.0,
    smooth_window: int = 11,
) -> Path:
    run_name = datetime.now().strftime("fixed_weight_search_%Y%m%d_%H%M%S")
    run_dir = result_root / run_name
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
        raise RuntimeError(f"可用样本过少（{len(all_records)}），不足以做稳定训练/测试。")

    train_records, test_records = _split_train_test(all_records, test_ratio=test_ratio, seed=seed)
    train_records = _absolutize_record_paths(train_records, base_cfg.database_dir)
    test_records = _absolutize_record_paths(test_records, base_cfg.database_dir)
    train_manifest = run_dir / "manifests" / "mse_train_manifest.json"
    test_manifest = run_dir / "manifests" / "mse_test_manifest.json"
    _write_manifest(train_manifest, "mse_train_split", train_records)
    _write_manifest(test_manifest, "mse_test_split", test_records)

    summary_rows: list[dict[str, Any]] = []
    for network_weight in network_weights:
        for physical_weight in physical_weights:
            case_name = f"nw_{network_weight:g}_pw_{physical_weight:g}"
            case_dir = run_dir / case_name
            cfg = _build_search_config(
                case_dir,
                device=device,
                epochs=epochs,
                seed=seed,
                training_mode=training_mode,
                physics_residual_weight=physics_residual_weight,
                network_weight=network_weight,
                physical_weight=physical_weight,
            )
            cfg.preprocess_steps = base_cfg.preprocess_steps
            cfg.clip_quantile = base_cfg.clip_quantile
            cfg.smooth_window = base_cfg.smooth_window
            trainer = ReconstructionTrainer(cfg)
            t0 = time.perf_counter()
            checkpoint = trainer.train(
                manifest_path=train_manifest,
                checkpoint_name=f"{case_name}.pt",
            )
            train_seconds = time.perf_counter() - t0
            test_metrics = _evaluate_on_manifest(cfg, checkpoint_path=checkpoint, test_manifest_path=test_manifest)
            row = {
                "case_name": case_name,
                "training_mode": cfg.training_mode,
                "learnable_branch_weights": cfg.learnable_branch_weights,
                "physics_residual_weight": cfg.physics_residual_weight,
                "network_weight": network_weight,
                "physical_weight": physical_weight,
                "train_seconds": train_seconds,
                "checkpoint": str(checkpoint),
                **test_metrics,
            }
            summary_rows.append(row)
            _write_search_outputs(
                run_dir=run_dir,
                base_cfg=base_cfg,
                data_root=data_root,
                import_stats=import_stats,
                train_records=train_records,
                test_records=test_records,
                test_ratio=test_ratio,
                seed=seed,
                training_mode=training_mode,
                physics_residual_weight=physics_residual_weight,
                network_weights=network_weights,
                physical_weights=physical_weights,
                summary_rows=summary_rows,
            )

    _write_search_outputs(
        run_dir=run_dir,
        base_cfg=base_cfg,
        data_root=data_root,
        import_stats=import_stats,
        train_records=train_records,
        test_records=test_records,
        test_ratio=test_ratio,
        seed=seed,
        training_mode=training_mode,
        physics_residual_weight=physics_residual_weight,
        network_weights=network_weights,
        physical_weights=physical_weights,
        summary_rows=summary_rows,
    )
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="搜索 ai_model fixed 模式下的网络/物性分支权重")
    parser.add_argument("--data-root", type=str, default="database/raw")
    parser.add_argument("--result-root", type=str, default="result")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--training-mode", type=str, default="residual_pinn", choices=("normal", "residual_pinn"))
    parser.add_argument("--physics-residual-weight", type=float, default=0.1)
    parser.add_argument("--network-weights", type=str, default="0.75,1.0,1.25")
    parser.add_argument("--physical-weights", type=str, default="1.0,2.0,3.0")
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
    if args.epochs < 1000:
        raise ValueError("--epochs 不能低于 1000")

    preprocess_steps = parse_preprocess_steps(args.preprocess)
    run_dir = run_fixed_weight_search(
        data_root=data_root,
        result_root=result_root,
        test_ratio=args.test_ratio,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        training_mode=args.training_mode,
        physics_residual_weight=args.physics_residual_weight,
        network_weights=_parse_float_list(args.network_weights),
        physical_weights=_parse_float_list(args.physical_weights),
        preprocess_steps=preprocess_steps,
        clip_quantile=args.clip_quantile,
        smooth_window=args.smooth_window,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
