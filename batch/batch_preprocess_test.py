from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .batch_test_modes import run_batch_test
from ..data_process import parse_preprocess_steps
from tqdm.auto import tqdm


def _parse_pipeline_group(raw: str) -> list[dict[str, Any]]:
    """
    Parse pipeline groups from string.

    Format:
      "base;clip,smooth;clip,smooth,detrend,zscore"
    """
    groups = [item.strip() for item in str(raw).split(";") if item.strip()]
    pipelines: list[dict[str, Any]] = []
    for item in groups:
        lowered = item.lower()
        if lowered in {"base", "none", "no_preprocess"}:
            pipelines.append({"name": "base", "steps": []})
            continue
        steps = parse_preprocess_steps(item)
        name = "_".join(steps)
        pipelines.append({"name": name, "steps": steps})
    # Deduplicate by name while preserving order.
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for pipeline in pipelines:
        name = str(pipeline["name"])
        if name in seen:
            continue
        seen.add(name)
        uniq.append(pipeline)
    return uniq


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_aggregate_outputs(
    run_dir: Path,
    aggregated_rows: list[dict[str, Any]],
    run_index: list[dict[str, Any]],
) -> None:
    if aggregated_rows:
        summary_csv = run_dir / "all_pipelines_summary.csv"
        with summary_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(aggregated_rows[0].keys()))
            writer.writeheader()
            writer.writerows(aggregated_rows)
        summary_json = run_dir / "all_pipelines_summary.json"
        summary_json.write_text(json.dumps(aggregated_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    index_json = run_dir / "run_index.json"
    index_json.write_text(json.dumps(run_index, ensure_ascii=False, indent=2), encoding="utf-8")


def _latest_batch_test_dir(root: Path) -> Path | None:
    candidates = [p for p in root.glob("batch_test_*") if p.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def run_batch_preprocess_test(
    data_root: Path,
    result_root: Path,
    pipelines: list[dict[str, Any]],
    test_ratio: float,
    epochs: int,
    device: str,
    seed: int,
    physics_residual_weight: float,
    clip_quantile: float,
    smooth_window: int,
    resume_run_dir: Path | None = None,
) -> Path:
    run_name = datetime.now().strftime("batch_preprocess_test_%Y%m%d_%H%M%S")
    run_dir = resume_run_dir if resume_run_dir is not None else (result_root / run_name)
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_json = run_dir / "all_pipelines_summary.json"
    index_json = run_dir / "run_index.json"
    aggregated_rows: list[dict[str, Any]] = _read_json(summary_json) if summary_json.exists() else []
    run_index: list[dict[str, Any]] = _read_json(index_json) if index_json.exists() else []
    completed = {str(item.get("pipeline_name", "")) for item in run_index}

    pipeline_bar = tqdm(
        list(enumerate(pipelines, start=1)),
        desc="preprocess-experiments",
        unit="pipeline",
    )
    total = len(pipelines)
    for idx, pipeline in pipeline_bar:
        pipeline_name = str(pipeline["name"])
        steps = list(pipeline["steps"])
        pipeline_bar.set_postfix(current=f"{idx}/{total}", name=pipeline_name)
        if pipeline_name in completed:
            continue
        case_result_root = run_dir / f"{idx:02d}_{pipeline_name}"
        case_result_root.mkdir(parents=True, exist_ok=True)
        existing_case_run_dir = _latest_batch_test_dir(case_result_root)

        case_run_dir = run_batch_test(
            data_root=data_root,
            result_root=case_result_root,
            test_ratio=test_ratio,
            epochs=epochs,
            device=device,
            seed=seed,
            physics_residual_weight=physics_residual_weight,
            preprocess_steps=steps,
            clip_quantile=clip_quantile,
            smooth_window=smooth_window,
            existing_run_dir=existing_case_run_dir,
            resume=True,
        )

        report = _read_json(case_run_dir / "report.json")
        summary_rows = report.get("summary", [])
        aggregated_rows = [row for row in aggregated_rows if row.get("pipeline_name") != pipeline_name]
        for row in summary_rows:
            agg = {
                "pipeline_name": pipeline_name,
                "preprocess_steps": ",".join(steps),
                "clip_quantile": clip_quantile,
                "smooth_window": smooth_window,
                "source_run_dir": str(case_run_dir),
                **row,
            }
            aggregated_rows.append(agg)

        run_index = [item for item in run_index if item.get("pipeline_name") != pipeline_name]
        run_index.append(
            {
                "pipeline_name": pipeline_name,
                "preprocess_steps": steps,
                "run_dir": str(case_run_dir),
            }
        )
        _write_aggregate_outputs(run_dir, aggregated_rows, run_index)
        tqdm.write(f"[done] pipeline {idx}/{total}: {pipeline_name}")

    if not aggregated_rows:
        raise RuntimeError("没有可汇总结果，请检查输入数据或 pipeline 配置。")
    _write_aggregate_outputs(run_dir, aggregated_rows, run_index)
    return run_dir


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="一次性批量跑多种预处理 pipeline，并汇总四模式结果。")
    parser.add_argument("--data-root", type=str, default="database/raw")
    parser.add_argument("--result-root", type=str, default="result")
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--physics-residual-weight", type=float, default=0.1)
    parser.add_argument("--clip-quantile", type=float, default=1.0)
    parser.add_argument("--smooth-window", type=int, default=11)
    parser.add_argument(
        "--pipelines",
        type=str,
        default="base;clip,smooth;clip,smooth,detrend,zscore;smooth,robust_norm",
        help="分号分隔多组 pipeline，例如: base;clip,smooth;clip,smooth,detrend,zscore",
    )
    parser.add_argument(
        "--resume-run-dir",
        type=str,
        default="",
        help="断点续跑目录（batch_preprocess_test_xxx）。不传则新建任务。",
    )
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

    pipelines = _parse_pipeline_group(args.pipelines)
    if not pipelines:
        raise ValueError("至少需要 1 组 pipeline")
    resume_run_dir = Path(args.resume_run_dir) if str(args.resume_run_dir).strip() else None
    if resume_run_dir is not None and not resume_run_dir.exists():
        raise FileNotFoundError(f"--resume-run-dir 不存在: {resume_run_dir}")

    run_dir = run_batch_preprocess_test(
        data_root=data_root,
        result_root=result_root,
        pipelines=pipelines,
        test_ratio=args.test_ratio,
        epochs=args.epochs,
        device=args.device,
        seed=args.seed,
        physics_residual_weight=args.physics_residual_weight,
        clip_quantile=args.clip_quantile,
        smooth_window=args.smooth_window,
        resume_run_dir=resume_run_dir,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
