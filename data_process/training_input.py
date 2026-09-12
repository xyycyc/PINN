"""Resolve a training dataset folder into train/validation manifests."""

from __future__ import annotations

import json
from pathlib import Path


def _resolve_declared_path(root: Path, value: object) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def resolve_training_manifest_pair(
    input_path: str | Path,
) -> tuple[Path, Path | None]:
    """Return ``(train, validation)`` from a dataset folder or manifest file.

    A material collection is returned as the first item with no validation
    item; collection-aware trainers resolve each material's pair themselves.
    """
    path = Path(input_path).expanduser().resolve()
    if path.is_dir():
        collection = path / "material_collection.json"
        if collection.is_file():
            return collection.resolve(), None
        root = path
        explicit_train: Path | None = None
    elif path.is_file():
        root = path.parent
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}
        if not isinstance(payload, dict):
            raise ValueError(f"训练清单顶层必须是 JSON 对象: {path}")
        if payload.get("collection_kind") == "sample_material_dataset_collection":
            return path, None
        explicit_train = path
    else:
        raise FileNotFoundError(f"训练数据集路径不存在: {path}")

    split_config = root / "split_config.json"
    declared: dict[str, object] = {}
    if split_config.is_file():
        try:
            split_payload = json.loads(split_config.read_text(encoding="utf-8"))
            raw_manifests = split_payload.get("manifests", {}) if isinstance(split_payload, dict) else {}
            if isinstance(raw_manifests, dict):
                declared = raw_manifests
        except (OSError, json.JSONDecodeError):
            declared = {}

    train_path = explicit_train or _resolve_declared_path(root, declared.get("train"))
    if train_path is None:
        train_path = (root / "train_manifest.json").resolve()
    validation_path = _resolve_declared_path(root, declared.get("validation"))
    if validation_path is None:
        candidate = root / "validation_manifest.json"
        validation_path = candidate.resolve() if candidate.is_file() else None

    if not train_path.is_file():
        raise FileNotFoundError(f"训练清单不存在: {train_path}")
    if validation_path is not None and not validation_path.is_file():
        raise FileNotFoundError(f"验证清单不存在: {validation_path}")
    return train_path, validation_path
