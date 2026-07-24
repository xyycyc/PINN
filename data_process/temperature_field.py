"""Versioned contracts for case-level waveform/temperature-field data."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

TEMPERATURE_FIELD_SCHEMA_VERSION = 2
SAMPLING_VERSION = "spatial-stratified-v2"
CASE_RE = re.compile(r"^(case_\d+)_T(\d+p\d+)K$")


@dataclass(frozen=True)
class CaseKey:
    worker: str
    case_id: str
    temperature_tag: str

    @classmethod
    def from_case_dir(cls, case_dir: str | Path) -> "CaseKey":
        path = Path(case_dir)
        match = CASE_RE.match(path.name)
        if not match:
            raise ValueError(f"case 目录名无效: {path.name}")
        worker = path.parent.name
        if not worker.startswith("worker_"):
            configs = sorted((path / "configs").glob("*_ultrasonic_config.json"))
            if len(configs) != 1:
                raise ValueError(
                    "扁平 case 目录需要唯一 ultrasonic config 来恢复 worker，"
                    f"实际找到 {len(configs)} 个: {path}"
                )
            try:
                payload = json.loads(configs[0].read_text(encoding="utf-8"))
                referenced = str(
                    payload.get("config", {}).get("io", {}).get("temperature_csv_path", "")
                ).replace("\\", "/")
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"无法从扁平 case 配置恢复 worker: {configs[0]}") from exc
            reference_match = re.search(
                rf"(?:^|/)(worker_\d+)/({re.escape(path.name)})(?:/|$)",
                referenced,
                flags=re.IGNORECASE,
            )
            if reference_match is None:
                raise ValueError(
                    "扁平 case 的 temperature_csv_path 未包含匹配的 worker/case: "
                    f"{referenced!r}"
                )
            worker = reference_match.group(1)
        return cls(worker, match.group(1), f"T{match.group(2)}K")

    @property
    def value(self) -> str:
        return f"{self.worker}/{self.case_id}_{self.temperature_tag}"


def select_waveform(case_dir: str | Path, prefer_corrected: bool = False) -> tuple[Path, str]:
    """Select one observation; rigid-motion data is never an independent case."""
    ultrasonic = Path(case_dir) / "ultrasonic"
    ordered = (
        [("receiver_signal_rigid_corrected.csv", "rigid_corrected"), ("receiver_signal.csv", "raw")]
        if prefer_corrected else
        [("receiver_signal.csv", "raw"), ("receiver_signal_rigid_corrected.csv", "rigid_corrected")]
    )
    for name, version in ordered:
        candidate = ultrasonic / name
        if candidate.is_file():
            return candidate, version
    raise FileNotFoundError(f"未找到原始或刚体校正接收波形: {ultrasonic}")


def mesh_fingerprint(node_ids: np.ndarray, coordinates_m: np.ndarray,
                     material_ids: np.ndarray, interface_side: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (node_ids, coordinates_m, material_ids, interface_side):
        value = np.ascontiguousarray(array)
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


REQUIRED_FIELDS = {
    "temperature_k", "coordinates_m", "node_ids", "material_ids",
    "interface_side", "sample_weights",
}


def validate_temperature_field(sample: Mapping[str, Any]) -> int:
    missing = sorted(REQUIRED_FIELDS - set(sample))
    if missing:
        raise ValueError(f"温度场 schema 缺少字段: {', '.join(missing)}")
    if sample.get("temperature_unit") != "K" or sample.get("coordinate_unit") != "m":
        raise ValueError("温度场单位必须明确为 temperature_unit='K'、coordinate_unit='m'")
    temperature = np.asarray(sample["temperature_k"])
    coordinates = np.asarray(sample["coordinates_m"])
    if temperature.ndim != 1:
        raise ValueError("temperature_k 必须是一维 [P]")
    count = int(temperature.shape[0])
    if count == 0 or count > 10000:
        raise ValueError("温度场点数 P 必须在 [1, 10000]")
    if coordinates.shape != (count, 2):
        raise ValueError(f"coordinates_m 必须为 [{count}, 2]")
    for name in ("node_ids", "material_ids", "interface_side", "sample_weights"):
        if np.asarray(sample[name]).shape != (count,):
            raise ValueError(f"{name} 点数与 temperature_k 不一致")
    if not np.all(np.isfinite(temperature)) or not np.all(np.isfinite(coordinates)):
        raise ValueError("温度或坐标包含非有限值")
    if np.any(np.asarray(sample["sample_weights"]) <= 0):
        raise ValueError("sample_weights 必须全部大于 0")
    return count


def load_manifest_compatible(path: str | Path) -> dict[str, Any]:
    """Read old manifests unchanged and explicitly label their schema generation."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload.get("records"), list):
        raise ValueError("manifest records 字段必须是列表")
    payload.setdefault("schema_version", 0)
    payload.setdefault("compatibility_mode", "legacy-grid" if payload["schema_version"] == 0 else "native")
    return payload
