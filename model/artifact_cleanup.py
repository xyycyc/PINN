from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..config import AIModelConfig
from .rule_registry import rule_csv_path
from .trainer import MATERIAL_ROUTER_KIND


@dataclass
class CleanupPlan:
    target_checkpoint: Path
    target_is_incremental: bool
    target_train_name: str
    target_report_paths: list[Path] = field(default_factory=list)
    dependent_incrementals: list[Path] = field(default_factory=list)
    dependent_report_paths: dict[str, list[Path]] = field(default_factory=dict)
    predict_dirs: dict[str, list[Path]] = field(default_factory=dict)
    csv_rows_for_target: int = 0
    material_router_paths: list[Path] = field(default_factory=list)
    material_group_checkpoints: list[Path] = field(default_factory=list)


@dataclass
class CleanupResult:
    removed_files: list[Path] = field(default_factory=list)
    removed_dirs: list[Path] = field(default_factory=list)
    updated_csv_rows: int = 0
    promoted_incrementals: list[tuple[Path, Path]] = field(default_factory=list)
    removed_csv_rows: int = 0


def _load_rule_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _write_rule_rows(csv_path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "created_at",
        "dimension",
        "mode",
        "material",
        "parameter_name",
        "parameter_path",
        "train_name",
        "training_mode",
        "physics_residual_weight",
        "learnable_branch_weights",
        "fixed_weight_cnn",
        "fixed_weight_lstm",
        "fixed_weight_material",
        "fixed_weight_dimension",
        "fixed_weight_mode",
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def _load_bundle(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_optional(path_text: str) -> Path | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    try:
        return Path(text).resolve()
    except OSError:
        return None


def _require_checkpoint_root_path(
    config: AIModelConfig,
    path: str | Path,
    *,
    label: str,
) -> Path:
    """Reject cleanup targets that escape the configured checkpoint root."""

    root = config.train_checkpoint_root.resolve()
    candidate = Path(path).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError(
            f"{label} 不在当前输出根的 checkpoint 目录内，已拒绝清理: {candidate}"
        )
    return candidate


def is_incremental_checkpoint(path: Path) -> bool:
    bundle = _load_bundle(path)
    return bool(bundle.get("base_checkpoint"))


def _infer_report_paths(config: AIModelConfig, checkpoint_path: Path) -> tuple[bool, list[Path], str]:
    train_name = checkpoint_path.parent.name
    report_root = config.train_report_root / train_name
    bundle = _load_bundle(checkpoint_path)
    stamp = str(bundle.get("incremental_stamp", "")).strip()
    if bundle.get("base_checkpoint"):
        if not stamp:
            stamp = checkpoint_path.stem
        return True, [report_root / "Incremental" / stamp], train_name
    stem = checkpoint_path.stem
    candidates = sorted(report_root.glob(f"{stem}_*"))
    summary = report_root / "training_summary.json"
    if summary.is_file():
        candidates.append(summary)
    return False, candidates, train_name


def _find_material_router_group(
    config: AIModelConfig,
    checkpoint_path: Path,
) -> tuple[list[Path], list[Path]]:
    """Return routers and all sample-material checkpoints in the same group."""

    target = checkpoint_path.resolve()
    root = config.train_checkpoint_root
    if not root.is_dir():
        return [], []
    routers: list[Path] = []
    checkpoints: set[Path] = set()
    for router_path in root.rglob("*__material_router.json"):
        try:
            payload = json.loads(router_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if str(payload.get("router_kind", "")) != MATERIAL_ROUTER_KIND:
            continue
        material_checkpoints = payload.get("checkpoints", [])
        if not isinstance(material_checkpoints, list):
            continue
        resolved_checkpoints: set[Path] = set()
        for item in material_checkpoints:
            if not isinstance(item, dict):
                continue
            raw = str(item.get("checkpoint", "")).strip()
            if not raw:
                continue
            path = Path(raw)
            resolved_checkpoints.add(
                (path if path.is_absolute() else router_path.parent / path).resolve()
            )
        if target in resolved_checkpoints:
            routers.append(router_path.resolve())
            checkpoints.update(resolved_checkpoints)
    return sorted(set(routers)), sorted(checkpoints)


def _collect_predict_dirs(config: AIModelConfig, checkpoints: set[Path]) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    roots = [config.predict_inference_root, config.predict_batch_root]
    for root in roots:
        if not root.exists():
            continue
        for run_dir in root.iterdir():
            if not run_dir.is_dir():
                continue
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            try:
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            checkpoint = _resolve_optional(str(payload.get("checkpoint", "")))
            if checkpoint is not None and checkpoint in checkpoints:
                result.setdefault(str(checkpoint), []).append(run_dir)
    return result


def find_incremental_dependents(config: AIModelConfig, checkpoint_path: Path) -> list[Path]:
    target = checkpoint_path.resolve()
    root = config.train_checkpoint_root
    if not root.exists():
        return []
    found: list[Path] = []
    for pt_path in root.glob("*/*.pt"):
        try:
            resolved = pt_path.resolve()
        except OSError:
            continue
        if resolved == target:
            continue
        bundle = _load_bundle(resolved)
        base = _resolve_optional(str(bundle.get("base_checkpoint", "")))
        if base is not None and base == target:
            found.append(resolved)
    found.sort()
    return found


def build_cleanup_plan(
    *,
    data_root: str | Path,
    result_root: str | Path,
    checkpoint_path: str | Path,
) -> CleanupPlan:
    config = AIModelConfig()
    config.data_root = config.resolve_path(data_root)
    config.result_root = config.resolve_path(result_root)

    target = _require_checkpoint_root_path(
        config,
        checkpoint_path,
        label="目标 checkpoint",
    )
    target_is_incremental, target_reports, target_train = _infer_report_paths(config, target)
    material_routers, material_group = _find_material_router_group(config, target)
    group_checkpoints = set(material_group) or {target}
    group_checkpoints = {
        _require_checkpoint_root_path(config, path, label="分材料 checkpoint")
        for path in group_checkpoints
    }
    material_routers = [
        _require_checkpoint_root_path(config, path, label="材料路由文件")
        for path in material_routers
    ]
    for group_checkpoint in group_checkpoints:
        _is_incremental, reports, _train_name = _infer_report_paths(config, group_checkpoint)
        target_reports.extend(reports)
    target_reports = sorted(set(target_reports))
    dependents = sorted(
        {
            dependent
            for group_checkpoint in group_checkpoints
            for dependent in find_incremental_dependents(config, group_checkpoint)
        }
    )
    all_for_predict = {*group_checkpoints, *material_routers, *dependents}
    predict_dirs = _collect_predict_dirs(config, all_for_predict)
    dependent_reports = {
        str(dep): _infer_report_paths(config, dep)[1] for dep in dependents
    }

    csv_file = rule_csv_path(config.data_root)
    rows = _load_rule_rows(csv_file)
    target_rows = 0
    for row in rows:
        row_path = _resolve_optional(str(row.get("parameter_path", "")))
        if row_path in group_checkpoints:
            target_rows += 1

    return CleanupPlan(
        target_checkpoint=target,
        target_is_incremental=target_is_incremental,
        target_train_name=target_train,
        target_report_paths=target_reports,
        dependent_incrementals=dependents,
        dependent_report_paths=dependent_reports,
        predict_dirs=predict_dirs,
        csv_rows_for_target=target_rows,
        material_router_paths=material_routers,
        material_group_checkpoints=sorted(group_checkpoints) if material_routers else [],
    )


def _safe_unlink(path: Path) -> bool:
    if not path.exists():
        return False
    if path.is_file():
        path.unlink(missing_ok=True)
        return True
    return False


def _safe_rmtree(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    shutil.rmtree(path)
    return True


def _cleanup_empty_dirs(path: Path, stop_at: Path) -> None:
    current = path
    stop = stop_at.resolve()
    while True:
        try:
            resolved = current.resolve()
        except OSError:
            break
        if resolved == stop or resolved == resolved.parent:
            break
        if not current.exists() or not current.is_dir():
            break
        try:
            next(current.iterdir())
            break
        except StopIteration:
            current.rmdir()
            current = current.parent


def _make_promoted_train_name(row: dict[str, str], stamp: str) -> str:
    dim = str(row.get("dimension", "")).strip() or "one"
    mode = str(row.get("mode", "")).strip() or "steady"
    material = str(row.get("material", "")).strip() or "material"
    return f"{dim}_{mode}_{material}_{stamp}_inc"


def _promote_incremental(
    *,
    config: AIModelConfig,
    rows: list[dict[str, str]],
    checkpoint_path: Path,
    csv_path: Path,
) -> tuple[Path, Path]:
    bundle = _load_bundle(checkpoint_path)
    stamp = str(bundle.get("incremental_stamp", "")).strip() or checkpoint_path.stem
    row = next(
        (
            item
            for item in rows
            if _resolve_optional(str(item.get("parameter_path", ""))) == checkpoint_path
        ),
        None,
    )
    if row is None:
        row = {}
    train_name = _make_promoted_train_name(row, stamp)
    checkpoint_dir = config.train_checkpoint_root / train_name
    report_dir = config.train_report_root / train_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    new_checkpoint = checkpoint_dir / checkpoint_path.name
    if new_checkpoint.exists():
        new_checkpoint = checkpoint_dir / f"{stamp}_{checkpoint_path.name}"
    shutil.move(str(checkpoint_path), str(new_checkpoint))

    old_report = config.train_report_root / checkpoint_path.parent.name / "Incremental" / stamp
    if old_report.exists():
        target_report = report_dir / stamp
        if target_report.exists():
            shutil.rmtree(target_report)
        shutil.move(str(old_report), str(target_report))
    else:
        target_report = report_dir

    moved_bundle = _load_bundle(new_checkpoint)
    if "base_checkpoint" in moved_bundle:
        moved_bundle["base_checkpoint"] = ""
        torch.save(moved_bundle, new_checkpoint)

    for item in rows:
        if _resolve_optional(str(item.get("parameter_path", ""))) != checkpoint_path:
            continue
        item["parameter_path"] = str(new_checkpoint)
        item["parameter_name"] = new_checkpoint.name
        item["train_name"] = train_name
    _write_rule_rows(csv_path, rows)
    return new_checkpoint, target_report


def execute_cleanup(
    *,
    data_root: str | Path,
    result_root: str | Path,
    checkpoint_path: str | Path,
    include_dependents: bool,
    promote_dependents: bool,
) -> CleanupResult:
    if include_dependents and promote_dependents:
        raise ValueError("include_dependents 与 promote_dependents 不能同时为真")

    config = AIModelConfig()
    config.data_root = config.resolve_path(data_root)
    config.result_root = config.resolve_path(result_root)
    plan = build_cleanup_plan(
        data_root=config.data_root,
        result_root=config.result_root,
        checkpoint_path=checkpoint_path,
    )

    if plan.dependent_incrementals and not include_dependents and not promote_dependents:
        raise ValueError("存在依赖该权重的增量模型，需先选择保留或一并删除")

    csv_file = rule_csv_path(config.data_root)
    rows = _load_rule_rows(csv_file)
    result = CleanupResult()

    if promote_dependents:
        for dep in plan.dependent_incrementals:
            new_ckpt, _ = _promote_incremental(
                config=config,
                rows=rows,
                checkpoint_path=dep,
                csv_path=csv_file,
            )
            result.promoted_incrementals.append((dep, new_ckpt))
        rows = _load_rule_rows(csv_file)

    checkpoints_to_remove = set(plan.material_group_checkpoints) or {plan.target_checkpoint}
    if include_dependents:
        checkpoints_to_remove.update(plan.dependent_incrementals)

    report_paths: list[Path] = list(plan.target_report_paths)
    if include_dependents:
        for dep in plan.dependent_incrementals:
            report_paths.extend(plan.dependent_report_paths.get(str(dep), []))

    predict_dirs: list[Path] = []
    for model_source in {*checkpoints_to_remove, *plan.material_router_paths}:
        predict_dirs.extend(plan.predict_dirs.get(str(model_source), []))

    for path in sorted(set(predict_dirs), key=lambda p: len(str(p)), reverse=True):
        if _safe_rmtree(path):
            result.removed_dirs.append(path)

    for path in sorted(set(report_paths), key=lambda p: len(str(p)), reverse=True):
        if path.is_dir():
            if _safe_rmtree(path):
                result.removed_dirs.append(path)
        elif _safe_unlink(path):
            result.removed_files.append(path)

    for ckpt in sorted(checkpoints_to_remove, key=lambda p: len(str(p)), reverse=True):
        if _safe_unlink(ckpt):
            result.removed_files.append(ckpt)
        _cleanup_empty_dirs(ckpt.parent, config.train_checkpoint_root)
        _cleanup_empty_dirs(config.train_report_root / ckpt.parent.name / "Incremental", config.train_report_root)
        _cleanup_empty_dirs(config.train_report_root / ckpt.parent.name, config.train_report_root)

    for router_path in plan.material_router_paths:
        if _safe_unlink(router_path):
            result.removed_files.append(router_path)
        _cleanup_empty_dirs(router_path.parent, config.train_checkpoint_root)

    filtered_rows: list[dict[str, str]] = []
    removed_rows = 0
    for row in rows:
        row_path = _resolve_optional(str(row.get("parameter_path", "")))
        if row_path in checkpoints_to_remove:
            removed_rows += 1
            continue
        filtered_rows.append(row)
    if filtered_rows != rows:
        _write_rule_rows(csv_file, filtered_rows)
    result.updated_csv_rows = len(filtered_rows)
    result.removed_csv_rows = removed_rows
    return result
