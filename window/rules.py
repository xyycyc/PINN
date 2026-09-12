from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from ..config import AIModelConfig
from ..paths import resolve_project_path
from ..model import (
    DIMENSION_CHOICES,
    MODE_CHOICES,
    default_training_runtime,
    discover_unregistered_checkpoints,
    ensure_rule_csv,
    resolve_training_runtime,
    rule_csv_path,
    validate_rule_triplet,
)

# window/rules.py -> parents[1] == ai_model 包根目录，与 AIModelConfig.repo_root 一致
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
_RULE_FIELDNAMES = (
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
)

NO_CHECKPOINT_LABEL = "暂无检查点"


@dataclass(frozen=True)
class CheckpointOption:
    label: str
    path: str
    created_at: str


def format_checkpoint_label(row: dict[str, str]) -> str:
    created_at = str(row.get("created_at", "")).strip() or "unknown"
    parameter_name = str(row.get("parameter_name", "")).strip() or Path(
        str(row.get("parameter_path", "")).strip()
    ).name
    training_mode = str(row.get("training_mode", "")).strip() or "-"
    return f"{created_at} | {parameter_name} | {training_mode}"


def resolve_data_root(data_root: str | Path) -> Path:
    return resolve_project_path(str(data_root).strip() or "database", repo_root=_PACKAGE_ROOT)


def rule_csv_for_data_root(data_root: str | Path) -> Path:
    return rule_csv_path(resolve_data_root(data_root))


def _repair_rule_csv_text(text: str) -> str | None:
    """修复表头末列与首条记录粘连的旧格式（fixed_weight_mode2026-...）。"""
    lines = text.splitlines()
    if not lines:
        return None
    first = lines[0]
    marker = "fixed_weight_mode"
    idx = first.find(f"{marker}20")
    if idx < 0:
        return None
    header_line = first[: idx + len(marker)]
    tail = first[idx + len(marker) :].lstrip(",")
    if not tail:
        return None
    repaired = [header_line, tail, *lines[1:]]
    return "\n".join(repaired) + ("\n" if text.endswith("\n") else "")


def sync_checkpoint_registry(
    data_root: str | Path,
    result_root: str | Path | None = None,
) -> int:
    """将磁盘上未写入规则表的检查点补登记（含历史增量权重）。"""

    root = resolve_data_root(data_root)
    cfg = AIModelConfig()
    cfg.data_root = cfg.resolve_path(root)
    if result_root is not None:
        cfg.result_root = cfg.resolve_path(result_root)
    csv_path = ensure_rule_csv(cfg.data_root)
    added = discover_unregistered_checkpoints(csv_path, cfg.train_checkpoint_root)
    return len(added)


def load_rule_rows(
    data_root: str | Path,
    result_root: str | Path | None = None,
) -> list[dict[str, str]]:
    try:
        sync_checkpoint_registry(data_root, result_root)
    except Exception:
        pass

    csv_path = rule_csv_for_data_root(data_root)
    if not csv_path.exists():
        return []

    text = csv_path.read_text(encoding="utf-8")
    repaired = _repair_rule_csv_text(text)
    if repaired is not None and repaired != text:
        csv_path.write_text(repaired, encoding="utf-8", newline="\n")
        text = repaired

    rows: list[dict[str, str]] = []
    with io.StringIO(text, newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and all(name in reader.fieldnames for name in _RULE_FIELDNAMES):
            rows = [dict(row) for row in reader]
        else:
            handle.seek(0)
            for raw in csv.reader(handle):
                if len(raw) < len(_RULE_FIELDNAMES):
                    continue
                if raw[0] == "created_at":
                    continue
                rows.append(dict(zip(_RULE_FIELDNAMES, raw[: len(_RULE_FIELDNAMES)], strict=False)))

    return [row for row in rows if str(row.get("material", "")).strip()]


def unique_materials(rows: list[dict[str, str]]) -> list[str]:
    values = sorted({str(row.get("material", "")).strip() for row in rows if str(row.get("material", "")).strip()})
    return values


def resolve_gui_project_path(path_value: str | Path, *, repo_root: str | Path) -> Path:
    """Compatibility wrapper shared with configuration and CLI paths."""
    return resolve_project_path(path_value, repo_root=repo_root)


def manifest_model_kind(manifest_path: str | Path, *, repo_root: str | Path) -> str:
    path = resolve_gui_project_path(manifest_path, repo_root=repo_root)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError):
        return "legacy_grid"
    if payload.get("collection_kind") == "sample_material_dataset_collection":
        return "material_collection"
    return "direct_point_field" if int(payload.get("schema_version", 0)) >= 1 and payload.get("sampling_index") else "legacy_grid"


@lru_cache(maxsize=256)
def _checkpoint_model_kind_cached(path_value: str, mtime_ns: int, size: int) -> str:
    """Read only checkpoint metadata once per on-disk file revision.

    ``mmap=True`` prevents model tensor storage from being copied merely to decide
    which GUI rule list should display the checkpoint.  The stat fields are part
    of the key so an overwritten checkpoint is inspected again automatically.
    """

    del mtime_ns, size  # Used only as cache-key revision markers.
    try:
        bundle = torch.load(path_value, map_location="cpu", mmap=True, weights_only=False)
    except (TypeError, RuntimeError):
        # PyTorch 2.0 and checkpoints written with the legacy serializer do not
        # necessarily support mmap. Keep those installations/checkpoints visible.
        bundle = torch.load(path_value, map_location="cpu", weights_only=False)
    return str(bundle.get("model_kind", "legacy_grid")) if isinstance(bundle, dict) else "legacy_grid"


def checkpoint_model_kind(checkpoint_path: str | Path) -> str:
    path = Path(checkpoint_path).resolve()
    stat = path.stat()
    return _checkpoint_model_kind_cached(str(path), stat.st_mtime_ns, stat.st_size)


def filter_rule_rows_for_model_kind(rows: list[dict[str, str]], model_kind: str) -> list[dict[str, str]]:
    """Keep only checkpoints compatible with the selected manifest representation."""
    compatible: list[dict[str, str]] = []
    for row in rows:
        path = Path(str(row.get("parameter_path", "")))
        if not path.is_file():
            continue
        try:
            actual = checkpoint_model_kind(path)
        except Exception:
            continue
        if actual == model_kind:
            compatible.append(row)
    return compatible


def available_dimensions(rows: list[dict[str, str]], *, material: str) -> list[str]:
    mat = str(material).strip()
    values = sorted(
        {
            str(row.get("dimension", "")).strip().lower()
            for row in rows
            if str(row.get("material", "")).strip() == mat
        }
    )
    return [item for item in values if item in DIMENSION_CHOICES]


def available_modes(rows: list[dict[str, str]], *, material: str, dimension: str) -> list[str]:
    mat = str(material).strip()
    dim = str(dimension).strip().lower()
    values = sorted(
        {
            str(row.get("mode", "")).strip().lower()
            for row in rows
            if str(row.get("material", "")).strip() == mat
            and str(row.get("dimension", "")).strip().lower() == dim
        }
    )
    return [item for item in values if item in MODE_CHOICES]


def list_checkpoint_options(
    rows: list[dict[str, str]],
    *,
    dimension: str,
    mode: str,
    material: str,
) -> list[CheckpointOption]:
    """列出同一规则三元组下已登记的全部检查点（按登记时间新→旧）。"""
    try:
        dim, md, mat = validate_rule_triplet(dimension, mode, material)
    except ValueError:
        return []

    best_by_path: dict[str, tuple[str, dict[str, str]]] = {}
    for row in rows:
        row_dim = str(row.get("dimension", "")).strip().lower()
        row_mode = str(row.get("mode", "")).strip().lower()
        row_mat = str(row.get("material", "")).strip()
        if row_dim != dim or row_mode != md or row_mat != mat:
            continue
        path = str(row.get("parameter_path", "")).strip()
        if not path:
            continue
        created_at = str(row.get("created_at", "")).strip()
        previous = best_by_path.get(path)
        if previous is None or created_at > previous[0]:
            best_by_path[path] = (created_at, row)

    options = [
        CheckpointOption(
            label=format_checkpoint_label(row),
            path=path,
            created_at=created_at,
        )
        for path, (created_at, row) in best_by_path.items()
    ]
    options.sort(key=lambda item: item.created_at, reverse=True)
    return options


def resolve_checkpoint_path(
    rows: list[dict[str, str]],
    *,
    dimension: str,
    mode: str,
    material: str,
) -> str:
    options = list_checkpoint_options(rows, dimension=dimension, mode=mode, material=material)
    return options[0].path if options else ""


def apply_checkpoint_combobox(
    combo: Any,
    rows: list[dict[str, str]],
    *,
    dimension: str,
    mode: str,
    material: str,
    label_to_path: dict[str, str],
    preferred_path: str = "",
) -> str:
    """填充检查点下拉框；返回当前选中项对应的磁盘路径（可能为空）。"""
    label_to_path.clear()
    options = list_checkpoint_options(
        rows,
        dimension=dimension,
        mode=mode,
        material=material,
    )
    if not options:
        combo.combo.configure(values=[NO_CHECKPOINT_LABEL], state="disabled")
        combo.set(NO_CHECKPOINT_LABEL)
        return ""

    labels = [item.label for item in options]
    for item in options:
        label_to_path[item.label] = item.path
    combo.combo.configure(values=labels, state="readonly")

    preferred = str(preferred_path or "").strip()
    selected_label = labels[0]
    if preferred:
        preferred_resolved = Path(preferred).resolve()
        for label, path in label_to_path.items():
            try:
                if Path(path).resolve() == preferred_resolved:
                    selected_label = label
                    break
            except OSError:
                if path == preferred:
                    selected_label = label
                    break
    combo.set(selected_label)
    return label_to_path.get(selected_label, "")


def default_rule_choices() -> tuple[tuple[str, ...], tuple[str, ...]]:
    return DIMENSION_CHOICES, MODE_CHOICES


def resolve_training_runtime_for_ui(
    data_root: str | Path,
    checkpoint_path: str,
) -> dict[str, Any]:
    """为 GUI 解析与基础 checkpoint 一致的训练运行参数。"""

    path = str(checkpoint_path or "").strip()
    if not path:
        return default_training_runtime()
    resolved = Path(path)
    if not resolved.exists():
        return default_training_runtime()
    try:
        return resolve_training_runtime(
            resolved,
            rule_csv_path=rule_csv_for_data_root(data_root),
        )
    except Exception:
        return default_training_runtime()
