"""Sample-level material dataset collections.

A collection groups sibling material folders such as ``wumu`` and ``steel``.
Each material is built and split independently because its mesh/sampling
contract may differ.  Node-level constituent IDs inside a case are deliberately
unrelated to this routing layer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .case_pipeline import build_case_dataset, discover_cases


MATERIAL_COLLECTION_KIND = "sample_material_dataset_collection"
MATERIAL_COLLECTION_VERSION = 1
MATERIAL_COLLECTION_FILE = "material_collection.json"
DEFAULT_MATERIAL_SPLIT = (0.7, 0.1, 0.2)

_KNOWN_MATERIAL_NAMES = {
    "wumu": "钨钼多层材料",
}


def sample_material_name(material_key: str) -> str:
    key = str(material_key).strip()
    return _KNOWN_MATERIAL_NAMES.get(key.casefold(), key)


def _safe_directory_name(value: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    safe = safe.strip("_")
    if not safe:
        raise ValueError(f"材料文件夹名不能转换为安全目录名: {value!r}")
    return safe


def validate_material_split(
    train_ratio: float,
    validation_ratio: float,
    test_ratio: float,
) -> tuple[float, float, float]:
    values = (float(train_ratio), float(validation_ratio), float(test_ratio))
    if not all(0.0 <= value < 1.0 for value in values):
        raise ValueError(f"材料划分比例必须分别位于 [0, 1): {values}")
    if values[0] <= 0.0 or values[2] <= 0.0:
        raise ValueError(f"材料训练集和测试集比例必须大于 0: {values}")
    if abs(sum(values) - 1.0) > 1e-9:
        raise ValueError(f"材料 train/validation/test 比例之和必须为 1: {values}")
    return values


def discover_material_roots(parent_root: str | Path) -> dict[str, Path]:
    """Return direct child folders that contain a supported case layout."""
    parent = Path(parent_root)
    if not parent.is_dir():
        raise FileNotFoundError(f"多材料上层目录不存在: {parent}")
    result: dict[str, Path] = {}
    safe_names: dict[str, str] = {}
    for child in sorted((item for item in parent.iterdir() if item.is_dir()), key=lambda p: p.name.casefold()):
        if not discover_cases(child):
            continue
        material_key = child.name.strip()
        if not material_key:
            continue
        safe = _safe_directory_name(material_key).casefold()
        if safe in safe_names:
            raise ValueError(
                f"材料文件夹输出名冲突: {safe_names[safe]!r} 与 {material_key!r}"
            )
        safe_names[safe] = material_key
        result[material_key] = child.resolve()
    if not result:
        raise ValueError(
            f"多材料目录没有包含 case 的直接子目录: {parent}；"
            "预期结构为 <上层>/<材料文件夹>/case_* 或 worker_*/case_*"
        )
    return result


def build_material_collection(
    parent_root: str | Path,
    output_dir: str | Path,
    *,
    dataset_label: str,
    material_splits: Mapping[str, tuple[float, float, float]] | None = None,
    target_points: int = 10000,
    seed: int = 42,
    waveform_crop_length: int = 1097,
    limit_per_material: int | None = None,
    train_manifest_name: str = "train_manifest.json",
    validation_manifest_name: str = "validation_manifest.json",
    test_manifest_name: str = "test_manifest.json",
) -> Path:
    roots = discover_material_roots(parent_root)
    supplied = dict(material_splits or {})
    unknown = sorted(set(supplied) - set(roots))
    if unknown:
        raise ValueError(f"划分比例引用了不存在的材料文件夹: {unknown}")

    output = Path(output_dir)
    materials_dir = output / "materials"
    materials_dir.mkdir(parents=True, exist_ok=True)
    entries: list[dict[str, Any]] = []
    for material_key, material_root in roots.items():
        train_ratio, validation_ratio, test_ratio = validate_material_split(
            *supplied.get(material_key, DEFAULT_MATERIAL_SPLIT)
        )
        material_output = materials_dir / _safe_directory_name(material_key)
        combined = build_case_dataset(
            material_root,
            material_output,
            target_points=target_points,
            seed=seed,
            waveform_crop_length=waveform_crop_length,
            limit=limit_per_material,
            test_ratio=test_ratio,
            validation_ratio=validation_ratio,
            dataset_label=dataset_label,
            sample_material_key=material_key,
            sample_material_name=sample_material_name(material_key),
            train_manifest_name=train_manifest_name,
            validation_manifest_name=validation_manifest_name,
            test_manifest_name=test_manifest_name,
        )
        manifest_payload = json.loads(combined.read_text(encoding="utf-8"))
        split_path = material_output / "split_config.json"
        split_payload = json.loads(split_path.read_text(encoding="utf-8"))
        entries.append(
            {
                "material_key": material_key,
                "material_name": sample_material_name(material_key),
                "source_root": str(material_root),
                "case_count": len(manifest_payload.get("records", [])),
                "ratios": {
                    "train": train_ratio,
                    "validation": validation_ratio,
                    "test": test_ratio,
                },
                "manifests": dict(split_payload["manifests"]),
                "split_config": str(split_path.resolve()),
            }
        )

    payload = {
        "collection_kind": MATERIAL_COLLECTION_KIND,
        "collection_version": MATERIAL_COLLECTION_VERSION,
        "dataset_label": str(dataset_label).strip(),
        "source_root": str(Path(parent_root).resolve()),
        "materials": entries,
    }
    collection_path = output / MATERIAL_COLLECTION_FILE
    collection_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return collection_path


def load_material_collection(path: str | Path) -> dict[str, Any]:
    collection_path = Path(path)
    payload = json.loads(collection_path.read_text(encoding="utf-8"))
    if payload.get("collection_kind") != MATERIAL_COLLECTION_KIND:
        raise ValueError(f"不是样本级多材料数据集合: {collection_path}")
    if int(payload.get("collection_version", 0)) != MATERIAL_COLLECTION_VERSION:
        raise ValueError(f"不支持的多材料集合版本: {payload.get('collection_version')}")
    materials = payload.get("materials", [])
    if not isinstance(materials, list) or not materials:
        raise ValueError("多材料数据集合没有 materials")
    keys = [str(item.get("material_key", "")).strip() for item in materials if isinstance(item, dict)]
    if len(keys) != len(materials) or any(not key for key in keys) or len(set(keys)) != len(keys):
        raise ValueError("多材料数据集合的 material_key 为空或重复")
    return payload


def resolve_collection_manifest(
    collection_path: str | Path,
    material_entry: Mapping[str, Any],
    kind: str,
) -> Path:
    if kind not in {"combined", "train", "validation", "test"}:
        raise ValueError(f"不支持的集合 manifest 类型: {kind}")
    raw = str(dict(material_entry.get("manifests", {})).get(kind, "")).strip()
    if not raw:
        raise ValueError(
            f"材料 {material_entry.get('material_key')} 缺少 {kind} manifest"
        )
    path = Path(raw)
    if not path.is_absolute():
        path = Path(collection_path).parent / path
    if not path.is_file():
        raise FileNotFoundError(f"材料 manifest 不存在: {path}")
    return path.resolve()


def build_mixed_collection_manifest(
    collection_path: str | Path,
    *,
    output_name: str = "mixed_train_manifest.json",
    split_kind: str = "train",
    normalization: Mapping[str, Any] | None = None,
) -> Path:
    """Merge one split across materials when output grids are identical.

    A mixed direct-point model has one shared output index space.  Therefore
    every material must use the same sampled nodes and waveform length.  The
    per-material manifests remain untouched; this helper writes one derived
    manifest next to the collection. Training normalization is fitted only for
    ``split_kind='train'`` and must be supplied for validation/test splits.
    """
    if split_kind not in {"train", "validation", "test"}:
        raise ValueError(f"不支持的混合清单 split_kind: {split_kind}")
    if split_kind != "train" and normalization is None:
        raise ValueError(f"混合 {split_kind} 清单必须复用训练集 normalization")
    collection_path = Path(collection_path).resolve()
    collection = load_material_collection(collection_path)
    output_path = collection_path.parent / str(output_name)
    records: list[dict[str, Any]] = []
    sample_catalog: list[dict[str, Any]] = []
    temperatures: list[np.ndarray] = []
    reference_sampling: dict[str, np.ndarray] | None = None
    reference_sampling_path: Path | None = None
    reference_payload: dict[str, Any] | None = None
    waveform_length: int | None = None
    constituent_catalogs: dict[str, list[dict[str, Any]]] = {}

    for raw_entry in collection["materials"]:
        entry = dict(raw_entry)
        material_key = str(entry["material_key"]).strip()
        manifest_path = resolve_collection_manifest(collection_path, entry, split_kind)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        sampling_raw = Path(str(payload.get("sampling_index", "")))
        sampling_path = sampling_raw if sampling_raw.is_absolute() else manifest_path.parent / sampling_raw
        with np.load(sampling_path, allow_pickle=False) as sampling:
            current_sampling = {
                key: np.asarray(sampling[key])
                for key in (
                    "node_ids",
                    "coordinates_m",
                    "material_ids",
                    "interface_side",
                    "sample_weights",
                )
            }
        if reference_sampling is None:
            reference_sampling = current_sampling
            reference_sampling_path = sampling_path.resolve()
            reference_payload = payload
        else:
            mismatched = [
                key
                for key, reference_value in reference_sampling.items()
                if current_sampling[key].shape != reference_value.shape
                or not np.allclose(current_sampling[key], reference_value, rtol=0.0, atol=1e-10)
            ]
            if mismatched:
                raise ValueError(
                    f"材料 {material_key!r} 与其他材料的固定采样网格不一致 ({mismatched})；"
                    "混合训练没有共同输出点语义，请启用分别训练"
                )

        source_records = [dict(item) for item in payload.get("records", [])]
        if not source_records:
            raise ValueError(f"材料 {material_key!r} 的 {split_kind} 清单为空: {manifest_path}")
        actual_materials = {
            str(item.get("material_key", "")).strip() for item in source_records
        }
        if actual_materials != {material_key}:
            raise ValueError(
                f"材料 {material_key!r} 的训练清单含有其他路由字段: {sorted(actual_materials)}"
            )
        for record in source_records:
            waveform_path = Path(str(record["waveform_path"]))
            field_path = Path(str(record["field_path"]))
            waveform_path = waveform_path if waveform_path.is_absolute() else manifest_path.parent / waveform_path
            field_path = field_path if field_path.is_absolute() else manifest_path.parent / field_path
            current_length = int(np.load(waveform_path, mmap_mode="r").shape[-1])
            if waveform_length is None:
                waveform_length = current_length
            elif current_length != waveform_length:
                raise ValueError(
                    f"多材料训练波形长度不一致: 期望 {waveform_length}，"
                    f"材料 {material_key!r} 样本为 {current_length}；请先统一波形处理策略"
                )
            with np.load(field_path, allow_pickle=False) as field:
                temperatures.append(np.asarray(field["temperature_k"], np.float64))
            record["sample_id"] = f"{material_key}/{record.get('sample_id', len(records))}"
            record["waveform_path"] = str(waveform_path.resolve())
            record["field_path"] = str(field_path.resolve())
            records.append(record)
        sample_catalog.append(
            {
                "material_key": material_key,
                "material_name": str(entry.get("material_name", material_key)),
                "case_count": len(source_records),
            }
        )
        constituent_catalogs[material_key] = [
            dict(item)
            for item in payload.get("constituent_material_catalog", [])
            if isinstance(item, dict)
        ]

    if reference_payload is None or reference_sampling_path is None or not temperatures:
        raise ValueError("多材料集合没有可用于混合训练的记录")
    joined = np.concatenate(temperatures)
    if split_kind == "train":
        std_k = float(joined.std())
        if std_k <= 0.0:
            raise ValueError("多材料训练温度标准差必须大于 0")
        resolved_normalization: dict[str, Any] = {
            "version": 1,
            "fit_split": "train",
            "method": "dataset_train_zscore",
            "mean_k": float(joined.mean()),
            "std_k": std_k,
            "min_k": float(joined.min()),
            "max_k": float(joined.max()),
            "source": "mixed_sample_material_train_splits",
        }
    else:
        resolved_normalization = dict(normalization or {})
        if resolved_normalization.get("fit_split") != "train":
            raise ValueError("validation/test 必须复用仅由训练集拟合的 normalization")
    unique_catalogs = {
        json.dumps(value, ensure_ascii=False, sort_keys=True)
        for value in constituent_catalogs.values()
    }
    payload = {
        "schema_version": int(reference_payload.get("schema_version", 0)),
        "dataset_name": f"mixed_sample_material_{split_kind}",
        "dataset_label": str(collection.get("dataset_label", "")),
        "source_collection": str(collection_path),
        "sample_material_catalog": sample_catalog,
        "constituent_material_catalog": (
            next(iter(constituent_catalogs.values())) if len(unique_catalogs) == 1 else []
        ),
        "constituent_material_catalog_by_sample_material": constituent_catalogs,
        "sampling_index": str(reference_sampling_path),
        "normalization": resolved_normalization,
        "split": {"name": split_kind, "version": 1, "source": "material_collection"},
        "temperature_summary": {
            "min_k": float(joined.min()),
            "max_k": float(joined.max()),
            "case_count": len(records),
            "point_count": int(reference_sampling["node_ids"].shape[0]),
        },
        "records": records,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path.resolve()


def latest_material_collection(data_root: str | Path) -> Path | None:
    root = Path(data_root) / "data_process"
    if not root.is_dir():
        return None
    candidates = [path for path in root.rglob(MATERIAL_COLLECTION_FILE) if path.is_file()]
    candidates.sort(key=lambda path: (path.stat().st_mtime_ns, str(path)), reverse=True)
    for path in candidates:
        try:
            load_material_collection(path)
            return path.resolve()
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None
