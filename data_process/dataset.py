"""Dataset loading, leakage-safe splitting, and waveform preprocessing contracts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..artifact_paths import (
    validate_artifact_basename,
    validate_distinct_artifact_basenames,
)
from ..config import AIModelConfig
from .material_registry import ensure_material_csv, load_material_to_idx, register_materials
from .preprocess import parse_preprocess_steps
from .preprocess_cache import load_or_apply_preprocessed_waveform

from .split_policy import SPLIT_EXPERIMENT_POLICIES


def latest_split_manifest(data_root: str | Path, kind: str) -> Path | None:
    """Return the newest existing manifest referenced by a split config."""

    if kind not in {"train", "validation", "test", "combined"}:
        raise ValueError(f"不支持的 manifest 类型: {kind}")
    root = Path(data_root)
    candidates: list[Path] = []
    processed_root = root / "data_process"
    if processed_root.is_dir():
        candidates.extend(
            path for path in processed_root.rglob("split_config.json") if path.is_file()
        )
    direct_case = root / "case_temperature_field" / "split_config.json"
    if direct_case.is_file():
        candidates.append(direct_case)
    dated_candidates: list[tuple[int, Path]] = []
    for path in candidates:
        try:
            dated_candidates.append((path.stat().st_mtime_ns, path))
        except OSError:
            continue
    dated_candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)

    for _mtime_ns, config_path in dated_candidates:
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
            manifests = payload.get("manifests", {})
            raw = str(manifests.get(kind, "") if isinstance(manifests, dict) else "").strip()
            if not raw:
                continue
            manifest = Path(raw)
            if not manifest.is_absolute():
                manifest = config_path.parent / manifest
            if manifest.is_file():
                manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
                records = manifest_payload.get("records", [])
                if not isinstance(records, list) or not records:
                    continue
                return manifest.resolve()
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None


def _split_indices(count: int, test_ratio: float, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    if count <= 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    indices = np.arange(count, dtype=np.int64)
    rng.shuffle(indices)
    test_count = int(round(count * float(test_ratio)))
    if count >= 2:
        test_count = min(max(test_count, 1), count - 1)
    else:
        test_count = 0
    return indices[test_count:], indices[:test_count]


def split_manifest_records(
    records: list[dict[str, Any]],
    test_ratio: float = 0.2,
    seed: int = 42,
    experiment_policy: str = "uniform",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split manifest records into train/test with configurable source policy."""
    if not (0.0 < float(test_ratio) < 1.0):
        raise ValueError("test_ratio 必须在 (0, 1) 区间内")
    if experiment_policy not in SPLIT_EXPERIMENT_POLICIES:
        raise ValueError(f"experiment_policy 必须是 {SPLIT_EXPERIMENT_POLICIES} 之一")

    rng = np.random.default_rng(int(seed))
    sim_like_sources = {"simulation", "external_simulation"}
    sim_records: list[dict[str, Any]] = []
    exp_records: list[dict[str, Any]] = []
    other_records: list[dict[str, Any]] = []
    for item in records:
        source = str(item.get("source", ""))
        if source in sim_like_sources:
            sim_records.append(item)
        elif source == "experiment":
            exp_records.append(item)
        else:
            other_records.append(item)

    sim_train_idx, sim_test_idx = _split_indices(len(sim_records), test_ratio, rng)
    train_records = [sim_records[int(i)] for i in sim_train_idx.tolist()]
    test_records = [sim_records[int(i)] for i in sim_test_idx.tolist()]

    if experiment_policy == "uniform":
        exp_train_idx, exp_test_idx = _split_indices(len(exp_records), test_ratio, rng)
        train_records.extend(exp_records[int(i)] for i in exp_train_idx.tolist())
        test_records.extend(exp_records[int(i)] for i in exp_test_idx.tolist())
    elif experiment_policy == "all_experiment_train":
        train_records.extend(exp_records)
    else:  # all_experiment_test
        test_records.extend(exp_records)

    # Unknown source fallback: keep in train to avoid dropping data.
    train_records.extend(other_records)

    stats = {
        "total_records": len(records),
        "train_records": len(train_records),
        "test_records": len(test_records),
        "test_ratio": float(test_ratio),
        "seed": int(seed),
        "experiment_policy": experiment_policy,
        "simulation_like_records": len(sim_records),
        "experiment_records": len(exp_records),
        "other_source_records": len(other_records),
        "train_source_breakdown": {
            "simulation_like": sum(1 for item in train_records if str(item.get("source", "")) in sim_like_sources),
            "experiment": sum(1 for item in train_records if str(item.get("source", "")) == "experiment"),
            "other": sum(
                1
                for item in train_records
                if str(item.get("source", "")) not in sim_like_sources and str(item.get("source", "")) != "experiment"
            ),
        },
        "test_source_breakdown": {
            "simulation_like": sum(1 for item in test_records if str(item.get("source", "")) in sim_like_sources),
            "experiment": sum(1 for item in test_records if str(item.get("source", "")) == "experiment"),
            "other": sum(
                1
                for item in test_records
                if str(item.get("source", "")) not in sim_like_sources and str(item.get("source", "")) != "experiment"
            ),
        },
    }
    return train_records, test_records, stats


def split_manifest_file(
    manifest_path: str | Path,
    test_ratio: float = 0.2,
    seed: int = 42,
    experiment_policy: str = "uniform",
    train_output_name: str = "train_manifest.json",
    test_output_name: str = "test_manifest.json",
) -> tuple[Path, Path, dict[str, Any]]:
    """Split a legacy manifest and persist reproducible train/test artifacts."""
    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("manifest records 字段必须是列表")

    train_records, test_records, stats = split_manifest_records(
        records=records,
        test_ratio=test_ratio,
        seed=seed,
        experiment_policy=experiment_policy,
    )

    train_name = validate_artifact_basename(
        str(train_output_name).strip() or "train_manifest.json",
        label="训练 manifest 文件名",
        suffix=".json",
    )
    test_name = validate_artifact_basename(
        str(test_output_name).strip() or "test_manifest.json",
        label="测试 manifest 文件名",
        suffix=".json",
    )
    validate_distinct_artifact_basenames(
        (train_name, test_name),
        label="训练/测试 manifest 文件名",
        reserved=(manifest_path.name, "split_config.json"),
    )
    train_path = manifest_path.parent / train_name
    test_path = manifest_path.parent / test_name

    dataset_name = str(payload.get("dataset_name", "ai_model_dataset"))
    train_payload = {"dataset_name": f"{dataset_name}_train_split", "records": train_records}
    test_payload = {"dataset_name": f"{dataset_name}_test_split", "records": test_records}
    train_path.write_text(json.dumps(train_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    test_path.write_text(json.dumps(test_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return train_path, test_path, stats


def split_case_records(records: list[dict[str, Any]], *, seed: int = 42,
                       train_ratio: float = 0.7, validation_ratio: float = 0.15
                       ) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Deterministically split unique physical cases and reject derived-wave leakage."""
    test_ratio = 1.0 - float(train_ratio) - float(validation_ratio)
    if abs(test_ratio) <= 1e-9:
        test_ratio = 0.0
    ratios = {
        "train": float(train_ratio),
        "validation": float(validation_ratio),
        "test": float(test_ratio),
    }
    if (
        not 0.0 < ratios["train"] <= 1.0
        or not 0.0 <= ratios["validation"] <= 1.0
        or not 0.0 <= ratios["test"] <= 1.0
        or abs(sum(ratios.values()) - 1.0) > 1e-9
    ):
        raise ValueError("train_ratio/validation_ratio 无效")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        case_key = str(record.get("case_key", "")).strip()
        if not case_key:
            raise ValueError("case 级划分要求每条记录包含 case_key")
        groups.setdefault(case_key, []).append(record)
    keys = np.array(sorted(groups), dtype=object)
    np.random.default_rng(seed).shuffle(keys)
    split_names = ("train", "validation", "test")
    n = len(keys)
    positive_splits = [name for name in split_names if ratios[name] > 0.0]

    raw_counts = {name: n * ratios[name] for name in split_names}
    counts = {name: int(np.floor(raw_counts[name])) for name in split_names}
    remaining = n - sum(counts.values())
    for name in sorted(
        split_names,
        key=lambda item: (raw_counts[item] - counts[item], ratios[item], item),
        reverse=True,
    )[:remaining]:
        counts[name] += 1

    # Small fixtures or newly collected materials may have fewer cases than
    # requested non-zero splits. Keep those cases in the highest-priority
    # proportional buckets instead of forcing synthetic validation/test data.
    if n >= len(positive_splits):
        for name in positive_splits:
            if counts[name] > 0:
                continue
            donors = sorted(
                (
                    candidate
                    for candidate in split_names
                    if counts[candidate] > (1 if ratios[candidate] > 0.0 else 0)
                ),
                key=lambda candidate: (counts[candidate], ratios[candidate], candidate),
                reverse=True,
            )
            if not donors:
                raise ValueError(f"无法为非零划分 {name} 分配 case")
            counts[donors[0]] -= 1
            counts[name] += 1

    n_train = counts["train"]
    n_val = counts["validation"]
    key_sets = {
        "train": set(keys[:n_train]),
        "validation": set(keys[n_train:n_train + n_val]),
        "test": set(keys[n_train + n_val:]),
    }
    result = {name: [item for key in sorted(values) for item in groups[str(key)]] for name, values in key_sets.items()}
    intersections = {"train_validation": sorted(key_sets["train"] & key_sets["validation"]),
                     "train_test": sorted(key_sets["train"] & key_sets["test"]),
                     "validation_test": sorted(key_sets["validation"] & key_sets["test"])}
    if any(intersections.values()):
        raise RuntimeError(f"case 泄漏检查失败: {intersections}")
    return result, {"split_version": 1, "seed": seed, "ratios": ratios,
            "case_counts": {k: len(v) for k, v in key_sets.items()}, "leakage": intersections}


def write_case_split_files(manifest_path: str | Path, *, seed: int = 42,
                           train_ratio: float = 0.7, validation_ratio: float = 0.15,
                           train_name: str = "train_manifest.json",
                           validation_name: str = "validation_manifest.json",
                           test_name: str = "test_manifest.json") -> dict[str, Any]:
    """Write case-level splits and fit temperature normalization on train only."""
    manifest_path = Path(manifest_path)
    train_name = validate_artifact_basename(
        train_name,
        label="训练 manifest 文件名",
        suffix=".json",
    )
    validation_name = validate_artifact_basename(
        validation_name,
        label="验证 manifest 文件名",
        suffix=".json",
    )
    test_name = validate_artifact_basename(
        test_name,
        label="测试 manifest 文件名",
        suffix=".json",
    )
    validate_distinct_artifact_basenames(
        (train_name, validation_name, test_name),
        label="训练/验证/测试 manifest 文件名",
        reserved=(manifest_path.name, "split_config.json"),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits, report = split_case_records(payload.get("records", []), seed=seed,
                                         train_ratio=train_ratio, validation_ratio=validation_ratio)
    train_values: list[np.ndarray] = []
    for record in splits["train"]:
        field_path = manifest_path.parent / str(record["field_path"])
        with np.load(field_path, allow_pickle=False) as field:
            train_values.append(np.asarray(field["temperature_k"], np.float64))
    if not train_values:
        raise ValueError("训练集为空，无法拟合温度标准化统计量")
    values = np.concatenate(train_values)
    normalization = {"version": 1, "fit_split": "train", "method": "dataset_train_zscore",
                     "mean_k": float(values.mean()), "std_k": float(values.std()),
                     "min_k": float(values.min()), "max_k": float(values.max()), "reversible": True}
    names = {"train": train_name, "validation": validation_name, "test": test_name}
    manifests: dict[str, str] = {"combined": str(manifest_path.resolve())}
    distribution: dict[str, Any] = {}
    combined_payload = dict(payload)
    combined_payload["normalization"] = normalization
    combined_payload["split"] = {"name": "combined", "version": 1, "seed": int(seed)}
    manifest_path.write_text(json.dumps(combined_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for split_name, records in splits.items():
        split_temperatures: list[np.ndarray] = []
        split_constituent_materials: set[int] = set()
        interface_nodes = 0
        for record in records:
            with np.load(manifest_path.parent / str(record["field_path"]), allow_pickle=False) as field:
                split_temperatures.append(np.asarray(field["temperature_k"], np.float64))
                split_constituent_materials.update(
                    int(v) for v in np.unique(field["material_ids"])
                )
                interface_nodes = max(interface_nodes, int(np.sum(np.asarray(field["interface_side"]) > 0)))
        joined = np.concatenate(split_temperatures) if split_temperatures else np.array([], dtype=np.float64)
        distribution[split_name] = {
            "case_count": len(records), "temperature_min_k": float(joined.min()) if joined.size else None,
            "temperature_max_k": float(joined.max()) if joined.size else None,
            "temperature_mean_k": float(joined.mean()) if joined.size else None,
            "constituent_material_ids": sorted(split_constituent_materials),
            "interface_node_count_per_case": interface_nodes,
        }
        split_payload = dict(payload)
        split_payload["dataset_name"] = f"{payload.get('dataset_name', 'case_temperature_field')}_{split_name}"
        split_payload["normalization"] = normalization
        split_payload["split"] = {"name": split_name, "version": 1, "seed": int(seed)}
        split_payload["records"] = records
        target = manifest_path.parent / names[split_name]
        target.write_text(json.dumps(split_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        manifests[split_name] = str(target.resolve())
    report = {**report, "normalization": normalization, "distribution": distribution, "manifests": manifests}
    (manifest_path.parent / "split_config.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


class AITemperatureDataset(Dataset):
    """Load legacy grids or fixed-node fields under one training interface."""

    def __init__(self, manifest_path: str | Path, config: AIModelConfig | None = None):
        manifest_path = Path(manifest_path)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.config = config or AIModelConfig()
        self.root = manifest_path.parent
        self.records = data["records"]
        waveform_lengths = {
            int(np.load(self.root / record["waveform_path"], mmap_mode="r").shape[-1])
            for record in self.records
        }
        if len(waveform_lengths) > 1:
            raise ValueError(f"manifest contains inconsistent waveform lengths: {sorted(waveform_lengths)}")
        self.waveform_length = next(iter(waveform_lengths), 0)
        self.schema_version = int(data.get("schema_version", 0))
        self.is_point_field = self.schema_version >= 1 and bool(data.get("sampling_index"))
        self.normalization = data.get("normalization", {}) if self.is_point_field else {}
        self.temperature_mean_k = float(self.normalization.get("mean_k", 0.0))
        self.temperature_std_k = float(self.normalization.get("std_k", 1.0))
        if self.is_point_field:
            if self.normalization.get("fit_split") != "train":
                raise ValueError("固定节点 manifest 必须携带仅由训练集拟合的 normalization")
            if self.temperature_std_k <= 0:
                raise ValueError("固定节点温度标准差必须大于 0")
            sampling_path = self.root / str(data["sampling_index"])
            with np.load(sampling_path, allow_pickle=False) as sampling:
                self.point_count = int(len(sampling["node_ids"]))
                self.point_coordinates_m = np.asarray(
                    sampling["coordinates_m"], np.float64
                )
                self.point_node_ids = np.asarray(sampling["node_ids"], np.int64)
                self.constituent_material_ids = np.asarray(
                    sampling["material_ids"], np.int64
                )
                self.point_interface_side = np.asarray(
                    sampling["interface_side"], np.int64
                )
                self.point_sample_weights = np.asarray(
                    sampling["sample_weights"], np.float32
                )
                self.sampling_metadata = json.loads(str(sampling["metadata_json"].item()))
            for static_values in (
                self.point_coordinates_m,
                self.point_node_ids,
                self.constituent_material_ids,
                self.point_interface_side,
                self.point_sample_weights,
            ):
                static_values.setflags(write=False)
            self.sampling_fingerprint = str(
                self.sampling_metadata.get("source_mesh_fingerprint", "")
            )
            self.constituent_material_catalog = [
                dict(item)
                for item in data.get(
                    "constituent_material_catalog",
                    self.sampling_metadata.get(
                        "constituent_material_catalog",
                        # Schema-v1 compatibility: this catalog was always
                        # node-level even though its old name was ambiguous.
                        self.sampling_metadata.get("material_catalog", []),
                    ),
                )
                if isinstance(item, dict)
            ]
        else:
            self.point_count = 0
            self.point_coordinates_m = np.empty((0, 2), dtype=np.float64)
            self.point_node_ids = np.array([], dtype=np.int64)
            self.constituent_material_ids = np.array([], dtype=np.int64)
            self.point_interface_side = np.array([], dtype=np.int64)
            self.point_sample_weights = np.array([], dtype=np.float32)
            self.sampling_metadata = {}
            self.sampling_fingerprint = ""
            self.constituent_material_catalog = []
        material_registry = ensure_material_csv(self.config.data_root)
        materials_in_manifest = sorted(
            {
                str(item.get("material_key", "")).strip()
                for item in self.records
                if str(item.get("material_key", "")).strip()
            }
        )
        # Only record-level sample materials participate in training/routing.
        # Internal layer/constituent names must never enter material.csv.
        register_materials(material_registry, materials_in_manifest, source="manifest")
        self.material_to_idx = load_material_to_idx(material_registry)
        if not self.material_to_idx:
            self.material_to_idx = {key: idx for idx, key in enumerate(materials_in_manifest)}
        self.dimension_to_idx = {"1d": 0, "2d": 1}
        self.mode_to_idx = {"steady": 0, "transient": 1}
        self.preprocess_steps = parse_preprocess_steps(self.config.preprocess_steps)
        self.clip_quantile = float(self.config.clip_quantile)
        self.smooth_window = int(self.config.smooth_window)
        self.preprocess_cache_root = self.config.data_root

    def __len__(self) -> int:
        return len(self.records)

    def _resize_field(self, field: np.ndarray) -> np.ndarray:
        # 统一温度场网格尺寸，保证 DataLoader 可以直接 batch 化
        target_h, target_w = self.config.field_grid_2d
        if field.ndim == 1:
            field = np.tile(field[None, :], (target_h, 1))
        elif field.ndim == 2 and field.shape[0] == 1:
            field = np.tile(field, (target_h, 1))

        y_old = np.linspace(0.0, 1.0, field.shape[0], dtype=np.float32)
        y_new = np.linspace(0.0, 1.0, target_h, dtype=np.float32)
        x_old = np.linspace(0.0, 1.0, field.shape[1], dtype=np.float32)
        x_new = np.linspace(0.0, 1.0, target_w, dtype=np.float32)
        y_resampled = np.stack(
            [np.interp(x_new, x_old, row).astype(np.float32) for row in field],
            axis=0,
        )
        resized = np.stack(
            [np.interp(y_new, y_old, y_resampled[:, col]).astype(np.float32) for col in range(y_resampled.shape[1])],
            axis=1,
        )
        return resized.astype(np.float32)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        source = str(record.get("source", ""))
        raw_waveform_path = (self.root / record["waveform_path"]).resolve()
        waveform_raw = np.load(raw_waveform_path).astype(np.float32)
        sample_id = str(record.get("sample_id", f"idx_{index}"))
        waveform = load_or_apply_preprocessed_waveform(
            waveform_raw,
            raw_waveform_path=raw_waveform_path,
            sample_id=sample_id,
            cache_root=self.preprocess_cache_root,
            preprocess_steps=self.preprocess_steps,
            clip_quantile=self.clip_quantile,
            smooth_window=self.smooth_window,
            use_cache=True,
        )
        if self.is_point_field:
            with np.load(self.root / record["field_path"], allow_pickle=False) as payload:
                field_k = np.asarray(payload["temperature_k"], np.float32)
                coordinates_m = np.asarray(payload["coordinates_m"], np.float32)
                node_ids = np.asarray(payload["node_ids"], np.int64)
                mesh_material_ids = np.asarray(payload["material_ids"], np.int64)
                interface_side = np.asarray(payload["interface_side"], np.int64)
                sample_weights = np.asarray(payload["sample_weights"], np.float32)
            if field_k.shape != (self.point_count,):
                raise ValueError(f"{record.get('sample_id')} 温度场点数不一致: {field_k.shape}")
            field = ((field_k - self.temperature_mean_k) / self.temperature_std_k).astype(np.float32)
        else:
            field = np.load(self.root / record["field_path"]).astype(np.float32)
            field = self._resize_field(field)

        acoustic = record.get("acoustic", {})
        if source in {"experiment", "experiment_case"}:
            amplitude = float(np.max(np.abs(waveform)))
        else:
            amplitude = float(acoustic.get("amplitude", 0.0))
        acoustic_vec = np.array(
            [
                float(acoustic.get("tof", 0.0)),
                amplitude,
                float(acoustic.get("center_freq", 0.0)),
            ],
            dtype=np.float32,
        )

        result = {
            "waveform": torch.from_numpy(waveform[None, :]),
            "field": torch.from_numpy(field),
            "temperature": torch.tensor([record["temperature_k"]], dtype=torch.float32),
            "acoustic": torch.from_numpy(acoustic_vec),
            # 实验样本通常没有完整温度场标签，只在仿真样本上监督 field。
            "field_mask": torch.tensor([1.0 if self.is_point_field or source in {"simulation", "external_simulation"} else 0.0], dtype=torch.float32),
            # 实验样本仅保留相对可靠的幅值监督，避免无效标签干扰训练。
            "acoustic_mask": torch.tensor(
                [1.0, 1.0, 1.0] if source in {"simulation", "external_simulation"} else [0.0, 1.0, 0.0],
                dtype=torch.float32,
            ),
            "material_id": torch.tensor(self.material_to_idx[record["material_key"]], dtype=torch.long),
            "dimension_id": torch.tensor(self.dimension_to_idx[record["dimension"]], dtype=torch.long),
            "mode_id": torch.tensor(self.mode_to_idx[record["mode"]], dtype=torch.long),
        }
        if self.is_point_field:
            result.update({
                "field_k": torch.from_numpy(field_k),
                "sample_weights": torch.from_numpy(sample_weights),
                "coordinates_m": torch.from_numpy(coordinates_m),
                "node_ids": torch.from_numpy(node_ids),
                "mesh_material_ids": torch.from_numpy(mesh_material_ids),
                "interface_side": torch.from_numpy(interface_side),
            })
        return result
