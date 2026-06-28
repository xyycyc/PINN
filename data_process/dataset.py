from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import AIModelConfig
from .material_registry import ensure_material_csv, load_material_to_idx, register_materials
from .preprocess import parse_preprocess_steps
from .preprocess_cache import load_or_apply_preprocessed_waveform

SPLIT_EXPERIMENT_POLICIES: tuple[str, ...] = (
    "uniform",
    "all_experiment_train",
    "all_experiment_test",
)


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

    train_name = str(train_output_name).strip() or "train_manifest.json"
    test_name = str(test_output_name).strip() or "test_manifest.json"
    train_path = manifest_path.parent / train_name
    test_path = manifest_path.parent / test_name

    dataset_name = str(payload.get("dataset_name", "ai_model_dataset"))
    train_payload = {"dataset_name": f"{dataset_name}_train_split", "records": train_records}
    test_payload = {"dataset_name": f"{dataset_name}_test_split", "records": test_records}
    train_path.write_text(json.dumps(train_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    test_path.write_text(json.dumps(test_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return train_path, test_path, stats


class AITemperatureDataset(Dataset):
    def __init__(self, manifest_path: str | Path, config: AIModelConfig | None = None):
        manifest_path = Path(manifest_path)
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.config = config or AIModelConfig()
        self.root = manifest_path.parent
        self.records = data["records"]
        material_registry = ensure_material_csv(self.config.data_root)
        materials_in_manifest = sorted(
            {
                str(item.get("material_key", "")).strip()
                for item in self.records
                if str(item.get("material_key", "")).strip()
            }
        )
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
        field = np.load(self.root / record["field_path"]).astype(np.float32)
        field = self._resize_field(field)

        acoustic = record.get("acoustic", {})
        if source == "experiment":
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

        return {
            "waveform": torch.from_numpy(waveform[None, :]),
            "field": torch.from_numpy(field),
            "temperature": torch.tensor([record["temperature_k"]], dtype=torch.float32),
            "acoustic": torch.from_numpy(acoustic_vec),
            # 实验样本通常没有完整温度场标签，只在仿真样本上监督 field。
            "field_mask": torch.tensor([1.0 if source in {"simulation", "external_simulation"} else 0.0], dtype=torch.float32),
            # 实验样本仅保留相对可靠的幅值监督，避免无效标签干扰训练。
            "acoustic_mask": torch.tensor(
                [1.0, 1.0, 1.0] if source in {"simulation", "external_simulation"} else [0.0, 1.0, 0.0],
                dtype=torch.float32,
            ),
            "material_id": torch.tensor(self.material_to_idx[record["material_key"]], dtype=torch.long),
            "dimension_id": torch.tensor(self.dimension_to_idx[record["dimension"]], dtype=torch.long),
            "mode_id": torch.tensor(self.mode_to_idx[record["mode"]], dtype=torch.long),
        }
