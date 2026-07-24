"""在已经训练完的 batch_preprocess_test/batch_test 目录上重跑预测对比。

用法:
    # 对一个 batch_preprocess_test_xxx 目录（包含多个 pipeline 子目录）重跑
    python -m ai_model.batch.rerun_predict --run-dir ai_model/result/batch_preprocess_test_20260408_210753

    # 对单个 batch_test_xxx 目录重跑
    python -m ai_model.batch.rerun_predict --run-dir ai_model/result/batch_preprocess_test_20260408_210753/01_base/batch_test_20260408_210753

该脚本不会重新训练，只重新推理，产物写到每个 mode 目录下的 ``predictions/``，
并在 run-dir 下写 ``all_predictions_summary.csv`` 汇总样本级温度误差指标。
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .batch_test_modes import _resolve_project_path

import torch
from tqdm.auto import tqdm

from ..config import AIModelConfig
from ..model import predict_and_compare, sync_config_for_inference


MODE_NAMES = (
    "normal_fixed",
    "normal_learnable",
    "residual_pinn_fixed",
    "residual_pinn_learnable",
)


def _iter_batch_test_dirs(run_dir: Path) -> list[Path]:
    """支持传入 batch_preprocess_test_xxx 或单个 batch_test_xxx。"""
    if (run_dir / "manifests").is_dir() and any((run_dir / m).is_dir() for m in MODE_NAMES):
        return [run_dir]
    batch_dirs: list[Path] = []
    for pipeline_dir in sorted(run_dir.iterdir()):
        if not pipeline_dir.is_dir():
            continue
        for child in sorted(pipeline_dir.iterdir()):
            if child.is_dir() and child.name.startswith("batch_test_"):
                batch_dirs.append(child)
    return batch_dirs


def _locate_checkpoint(mode_dir: Path, mode_name: str) -> Path | None:
    candidates = [
        mode_dir / "result" / "train" / "checkpoint" / mode_name / f"{mode_name}.pt",
        mode_dir / "outputs" / "checkpoints" / f"{mode_name}.pt",
    ]
    for path in candidates:
        if path.exists():
            return path
    # 兜底：兼容新旧目录，遍历候选 checkpoint 根目录。
    for ckpt_dir in (
        mode_dir / "result" / "train" / "checkpoint",
        mode_dir / "outputs" / "checkpoints",
    ):
        if not ckpt_dir.is_dir():
            continue
        if ckpt_dir.name == "checkpoint":
            nested = sorted(ckpt_dir.glob("*/*.pt"))
            if nested:
                return nested[0]
        matches = sorted(ckpt_dir.glob("*.pt"))
        if matches:
            return matches[0]
    return None


def _locate_test_manifest(batch_test_dir: Path) -> Path | None:
    path = batch_test_dir / "manifests" / "mse_test_manifest.json"
    return path if path.exists() else None


def rerun_predictions(
    run_dir: Path,
    device: str,
    overwrite: bool,
    num_field_samples: int,
) -> Path:
    batch_test_dirs = _iter_batch_test_dirs(run_dir)
    if not batch_test_dirs:
        raise RuntimeError(f"未在 {run_dir} 找到任何 batch_test_* 目录。")

    aggregated_rows: list[dict[str, Any]] = []
    outer_bar = tqdm(batch_test_dirs, desc="batch-test-dirs", unit="pipeline")
    for batch_test_dir in outer_bar:
        outer_bar.set_postfix(pipeline=batch_test_dir.parent.name)
        test_manifest = _locate_test_manifest(batch_test_dir)
        if test_manifest is None:
            tqdm.write(f"[skip] 找不到 test manifest: {batch_test_dir}")
            continue

        pipeline_name = batch_test_dir.parent.name

        for mode_name in MODE_NAMES:
            mode_dir = batch_test_dir / mode_name
            if not mode_dir.is_dir():
                continue
            checkpoint = _locate_checkpoint(mode_dir, mode_name)
            if checkpoint is None:
                tqdm.write(f"[skip] {pipeline_name}/{mode_name}: 没找到 checkpoint")
                continue
            predictions_dir = mode_dir / "predictions"
            if predictions_dir.exists() and not overwrite:
                tqdm.write(f"[keep] {pipeline_name}/{mode_name}: predictions/ 已存在，使用 --overwrite 强制重算")
                metrics_path = predictions_dir / "metrics.json"
                if metrics_path.exists():
                    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                else:
                    continue
            else:
                cfg = AIModelConfig()
                sync_config_for_inference(cfg, checkpoint)
                if device:
                    cfg.device = device
                metrics = predict_and_compare(
                    cfg=cfg,
                    checkpoint_path=checkpoint,
                    manifest_path=test_manifest,
                    output_dir=predictions_dir,
                    sync_config_from_checkpoint=False,
                    enable_plots=True,
                    num_field_samples=num_field_samples,
                )
                tqdm.write(
                    f"[done] {pipeline_name}/{mode_name}: "
                    f"MAE={metrics.get('temperature_mae', float('nan')):.4f}  "
                    f"RMSE={metrics.get('temperature_rmse', float('nan')):.4f}"
                )

            aggregated_rows.append(
                {
                    "pipeline_name": pipeline_name,
                    "mode_name": mode_name,
                    "batch_test_dir": str(batch_test_dir),
                    "checkpoint": str(checkpoint),
                    "predictions_dir": str(predictions_dir),
                    "samples": metrics.get("samples", 0),
                    "temperature_mae": metrics.get("temperature_mae", float("nan")),
                    "temperature_rmse": metrics.get("temperature_rmse", float("nan")),
                    "field_loss": metrics.get("field_loss", float("nan")),
                    "acoustic_loss": metrics.get("acoustic_loss", float("nan")),
                    "total_loss": metrics.get("total_loss", float("nan")),
                }
            )

    if not aggregated_rows:
        raise RuntimeError("没有可用的预测结果，请检查 --run-dir 下是否存在 checkpoint/manifest。")

    summary_csv = run_dir / "all_predictions_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(aggregated_rows[0].keys()))
        writer.writeheader()
        writer.writerows(aggregated_rows)
    summary_json = run_dir / "all_predictions_summary.json"
    summary_json.write_text(
        json.dumps(aggregated_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary_csv


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="对已完成训练的 batch_test 目录重跑预测对比")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="batch_preprocess_test_xxx 或 batch_test_xxx 目录",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="若 predictions/ 已存在也强制重算",
    )
    parser.add_argument(
        "--num-field-samples",
        type=int,
        default=6,
        help="抽样绘制真/预/误差三联图的样本数",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = _resolve_project_path(args.run_dir)
    if not run_dir.exists():
        raise FileNotFoundError(f"--run-dir 不存在: {run_dir}")
    summary_csv = rerun_predictions(
        run_dir=run_dir,
        device=args.device,
        overwrite=args.overwrite,
        num_field_samples=args.num_field_samples,
    )
    print(summary_csv)


if __name__ == "__main__":
    main()
