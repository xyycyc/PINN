from __future__ import annotations

import csv
import json
import os
import tempfile
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from ..artifact_paths import validate_artifact_basename
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
    # Preserve extension columns and replace a complete registry atomically.
    fieldnames.extend(
        sorted({key for row in rows for key in row if key and key not in fieldnames})
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=csv_path.parent,
            prefix=f".{csv_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({name: row.get(name, "") for name in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, csv_path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _load_bundle(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _resolve_optional(
    path_text: str, *, relative_to: Path | None = None
) -> Path | None:
    text = str(path_text or "").strip()
    if not text:
        return None
    try:
        path = Path(text)
        if relative_to is not None and not path.is_absolute():
            path = relative_to / path
        return path.resolve()
    except (OSError, ValueError):
        return None


def _require_checkpoint_root_path(
    config: AIModelConfig,
    path: str | Path,
    *,
    label: str,
) -> Path:
    """Reject cleanup targets that escape the configured checkpoint root."""

    try:
        return _require_root_path(path, config.train_checkpoint_root, label=label)
    except ValueError as exc:
        raise ValueError(
            f"{label} 不在当前输出根的 checkpoint 目录内，已拒绝清理: {path}"
        ) from exc


def _require_root_path(path: str | Path, root: Path, *, label: str) -> Path:
    root = root.resolve()
    candidate = Path(path).resolve()
    if candidate == root or not candidate.is_relative_to(root):
        raise ValueError(f"{label} 不在允许目录 {root} 内，已拒绝清理: {candidate}")
    return candidate


def is_incremental_checkpoint(path: Path) -> bool:
    bundle = _load_bundle(path)
    return bool(bundle.get("base_checkpoint"))


def _infer_report_paths(
    config: AIModelConfig,
    checkpoint_path: Path,
    *,
    owned_checkpoints: set[Path] | None = None,
) -> tuple[bool, list[Path], str]:
    train_name = checkpoint_path.parent.name
    report_root = _require_root_path(
        config.train_report_root / train_name,
        config.train_report_root,
        label="训练报告目录",
    )
    bundle = _load_bundle(checkpoint_path)
    stamp = str(bundle.get("incremental_stamp", "")).strip()
    if bundle.get("base_checkpoint"):
        stamp = validate_artifact_basename(
            stamp or checkpoint_path.stem, label="增量报告时间戳"
        )
        candidate = _require_root_path(
            report_root / "Incremental" / stamp, report_root, label="增量报告"
        )
        return True, [candidate], train_name
    # Match names literally and give a longer sibling model name precedence.
    prefix = f"{checkpoint_path.stem}_".casefold()
    owned = owned_checkpoints if owned_checkpoints is not None else {checkpoint_path}
    others = [
        path
        for path in checkpoint_path.parent.glob("*.pt")
        if path.resolve() not in owned
    ]
    other_prefixes = [
        f"{path.stem}_".casefold() for path in others if len(path.stem) > len(checkpoint_path.stem)
    ]
    candidates = (
        [
            path
            for path in report_root.iterdir()
            if path.name.casefold().startswith(prefix)
            and not any(path.name.casefold().startswith(other) for other in other_prefixes)
        ]
        if report_root.is_dir()
        else []
    )
    if stamp:  # Reports retained when an incremental model became independent.
        stamp = validate_artifact_basename(stamp, label="独立增量报告时间戳")
        candidates.append(report_root / stamp)
    summary = report_root / "training_summary.json"
    if summary.is_file():
        summary_owned = not others
        if others and isinstance(bundle.get("training_summary"), dict):
            try:
                summary_owned = (
                    json.loads(summary.read_text(encoding="utf-8"))
                    == bundle["training_summary"]
                )
            except (OSError, ValueError):
                summary_owned = False
        if summary_owned:
            candidates.append(summary)
    candidates = [
        _require_root_path(path, report_root, label="训练报告") for path in candidates
    ]
    return False, sorted(set(candidates)), train_name


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
        if (
            not isinstance(payload, dict)
            or str(payload.get("router_kind", "")) != MATERIAL_ROUTER_KIND
        ):
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


def _collect_predict_dirs(
    config: AIModelConfig, checkpoints: set[Path]
) -> dict[str, list[Path]]:
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
            if not isinstance(payload, dict):
                continue
            checkpoint = _resolve_optional(str(payload.get("checkpoint", "")))
            if checkpoint is not None and checkpoint in checkpoints:
                result.setdefault(str(checkpoint), []).append(
                    _require_root_path(run_dir, root, label="预测结果目录")
                )
    return result


def _incremental_dependency_graph(config: AIModelConfig) -> dict[Path, set[Path]]:
    """Scan each managed checkpoint once, including nested task directories."""
    root = config.train_checkpoint_root.resolve()
    graph: dict[Path, set[Path]] = {}
    if root.is_dir():
        for path in root.rglob("*.pt"):
            resolved = path.resolve()
            if not resolved.is_relative_to(root) or not resolved.is_file():
                continue
            bundle = _load_bundle(resolved)
            base = _resolve_optional(
                str(bundle.get("base_checkpoint", "")), relative_to=resolved.parent
            )
            if base is not None:
                graph.setdefault(base, set()).add(resolved)
    return graph


def _walk_dependents(graph: dict[Path, set[Path]], targets: set[Path]) -> list[Path]:
    visited = set(targets)
    pending = list(targets)
    while pending:
        for child in graph.get(pending.pop(), ()):
            if child not in visited:
                visited.add(child)
                pending.append(child)
    return sorted(visited - targets)


def find_incremental_dependents(
    config: AIModelConfig, checkpoint_path: Path
) -> list[Path]:
    """Return all descendants, safely handling branching and cyclic metadata."""
    return _walk_dependents(
        _incremental_dependency_graph(config), {checkpoint_path.resolve()}
    )


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
    target_is_incremental, target_reports, target_train = _infer_report_paths(
        config, target
    )
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
        _is_incremental, reports, _train_name = _infer_report_paths(
            config, group_checkpoint, owned_checkpoints=group_checkpoints
        )
        target_reports.extend(reports)
    target_reports = sorted(set(target_reports))
    dependents = _walk_dependents(
        _incremental_dependency_graph(config), group_checkpoints
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
        material_group_checkpoints=sorted(group_checkpoints)
        if material_routers
        else [],
    )


def _safe_unlink(path: Path, root: Path) -> bool:
    path = _require_root_path(path, root, label="待删除文件")
    if not path.exists():
        return False
    if path.is_file():
        path.unlink(missing_ok=True)
        return True
    return False


def _safe_rmtree(path: Path, root: Path) -> bool:
    path = _require_root_path(path, root, label="待删除目录")
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
        if resolved == stop or not resolved.is_relative_to(stop):
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
    return validate_artifact_basename(
        f"{dim}_{mode}_{material}_{stamp}_inc", label="保留增量模型的任务名"
    )


@dataclass
class _Promotion:
    original: Path
    checkpoint: Path
    old_report: Path
    report: Path
    train_name: str
    stamp: str
    inherited_config: dict[str, Any]


def _plan_promotion(
    config: AIModelConfig, rows: list[dict[str, str]], checkpoint_path: Path
) -> _Promotion:
    from .checkpoint_runtime import resolve_checkpoint_config_dict

    bundle = _load_bundle(checkpoint_path)
    if not bundle:
        raise ValueError(f"无法读取待保留的增量模型，已取消清理: {checkpoint_path}")
    stamp = validate_artifact_basename(
        str(bundle.get("incremental_stamp", "")).strip() or checkpoint_path.stem,
        label="增量报告时间戳",
    )
    row = next(
        (
            item
            for item in rows
            if _resolve_optional(item.get("parameter_path", "")) == checkpoint_path
        ),
        {},
    )
    train_name = _make_promoted_train_name(row, stamp)
    checkpoint_dir = _require_root_path(
        config.train_checkpoint_root / train_name,
        config.train_checkpoint_root,
        label="独立模型目录",
    )
    report_dir = _require_root_path(
        config.train_report_root / train_name,
        config.train_report_root,
        label="独立报告目录",
    )
    if checkpoint_dir.exists() or report_dir.exists():
        raise ValueError(
            f"保留增量模型的目标任务已存在，已取消清理以避免覆盖: {train_name}"
        )
    inherited = bundle.get("config")
    if not isinstance(inherited, dict):
        inherited = resolve_checkpoint_config_dict(checkpoint_path)
    old_report = _infer_report_paths(config, checkpoint_path)[1][0]
    return _Promotion(
        checkpoint_path,
        checkpoint_dir / checkpoint_path.name,
        old_report,
        report_dir / stamp,
        train_name,
        stamp,
        inherited,
    )


def _promote_incremental(promotion: _Promotion, rows: list[dict[str, str]]) -> Path:
    # Preflight retains metadata only; materialize one model at a time.
    bundle = _load_bundle(promotion.original)
    if not bundle:
        raise ValueError(f"无法读取待保留的增量模型: {promotion.original}")
    if promotion.inherited_config:
        bundle["config"] = promotion.inherited_config
    bundle["base_checkpoint"] = ""
    bundle["incremental_stamp"] = promotion.stamp
    # The original remains intact until the independent checkpoint is fully saved.
    promotion.checkpoint.parent.mkdir(parents=True, exist_ok=False)
    try:
        with promotion.checkpoint.open("xb") as stream:
            torch.save(bundle, stream)
        if promotion.old_report.exists():
            promotion.report.parent.mkdir(parents=True, exist_ok=False)
            shutil.move(str(promotion.old_report), str(promotion.report))
    except Exception:
        promotion.checkpoint.unlink(missing_ok=True)
        raise
    promotion.original.unlink()
    for item in rows:
        if _resolve_optional(item.get("parameter_path", "")) == promotion.original:
            item.update(
                parameter_path=str(promotion.checkpoint),
                parameter_name=promotion.checkpoint.name,
                train_name=promotion.train_name,
            )
    return promotion.checkpoint


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

    if (
        plan.dependent_incrementals
        and not include_dependents
        and not promote_dependents
    ):
        raise ValueError("存在依赖该权重的增量模型，需先选择保留或一并删除")

    csv_file = rule_csv_path(config.data_root)
    rows = _load_rule_rows(csv_file)
    result = CleanupResult()

    if promote_dependents:
        # Validate every destination and inherited config before the first move.
        promotions = [
            _plan_promotion(config, rows, dep) for dep in plan.dependent_incrementals
        ]
        destinations = [item.checkpoint.parent for item in promotions]
        if len(destinations) != len(set(destinations)):
            raise ValueError("多个增量模型的独立任务名重复，已取消清理以避免覆盖")
        for promotion in promotions:
            new_checkpoint = _promote_incremental(promotion, rows)
            _write_rule_rows(csv_file, rows)
            result.promoted_incrementals.append((promotion.original, new_checkpoint))

    checkpoints_to_remove = set(plan.material_group_checkpoints) or {
        plan.target_checkpoint
    }
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
        if _safe_rmtree(path, config.result_root / "predict"):
            result.removed_dirs.append(path)

    for path in sorted(set(report_paths), key=lambda p: len(str(p)), reverse=True):
        if path.is_dir():
            if _safe_rmtree(path, config.train_report_root):
                result.removed_dirs.append(path)
        elif _safe_unlink(path, config.train_report_root):
            result.removed_files.append(path)

    for ckpt in sorted(checkpoints_to_remove, key=lambda p: len(str(p)), reverse=True):
        if _safe_unlink(ckpt, config.train_checkpoint_root):
            result.removed_files.append(ckpt)
        _cleanup_empty_dirs(ckpt.parent, config.train_checkpoint_root)
        _cleanup_empty_dirs(
            config.train_report_root / ckpt.parent.name / "Incremental",
            config.train_report_root,
        )
        _cleanup_empty_dirs(
            config.train_report_root / ckpt.parent.name, config.train_report_root
        )

    for router_path in plan.material_router_paths:
        if _safe_unlink(router_path, config.train_checkpoint_root):
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
