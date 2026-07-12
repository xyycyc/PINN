"""Build traceable case-level waveform and sampled temperature-field artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .builder import _resample_signal
from .mesh_sampling import deterministic_sample, read_mesh_csv, save_sampling_index
from .temperature_field import (
    CaseKey, TEMPERATURE_FIELD_SCHEMA_VERSION, mesh_fingerprint,
    select_waveform, validate_temperature_field,
)
from .dataset import write_case_split_files


def discover_cases(root: str | Path) -> list[Path]:
    return sorted(p for p in Path(root).glob("worker_*/case_*_T*K") if p.is_dir())


def thermal_csv(case_dir: Path) -> Path:
    return case_dir / "heat" / "thermomechanical_steady" / "csv" / "thermomechanical_steady_nodes.csv"


def validate_case_pair(case_dir: Path) -> dict[str, Any]:
    """Validate the ultrasonic config reference against the mirrored local case."""
    key = CaseKey.from_case_dir(case_dir)
    configs = sorted((case_dir / "configs").glob("*_ultrasonic_config.json"))
    if len(configs) != 1:
        raise ValueError(f"超声配置数量应为 1，实际为 {len(configs)}")
    config_path = configs[0]
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    referenced = str(payload.get("config", {}).get("io", {}).get("temperature_csv_path", "")).strip()
    if not referenced:
        raise ValueError("超声配置缺少 config.io.temperature_csv_path")
    normalized = referenced.replace("\\", "/")
    if case_dir.name not in normalized:
        raise ValueError(f"temperature_csv_path 的 case ID 不一致: {referenced}")
    local_csv = thermal_csv(case_dir)
    if Path(normalized).name != local_csv.name or not local_csv.is_file():
        raise FileNotFoundError(f"temperature_csv_path 本地镜像不存在: {local_csv}")
    if local_csv.stat().st_size <= 0:
        raise ValueError(f"热力 CSV 为空: {local_csv}")
    return {"case_key": key.value, "ultrasonic_config_path": str(config_path),
            "referenced_temperature_csv_path": referenced, "local_temperature_csv_path": str(local_csv)}


def audit_meshes(root: str | Path, coordinate_tolerance_m: float = 1e-10) -> dict[str, Any]:
    cases, failures, accepted = discover_cases(root), [], []
    reference: dict[str, np.ndarray] | None = None
    reference_fp = ""
    for case in cases:
        try:
            mesh = read_mesh_csv(thermal_csv(case))
            fp = mesh_fingerprint(mesh["node_ids"], mesh["coordinates_m"], mesh["material_ids"], mesh["interface_side"])
            if reference is None:
                reference, reference_fp = mesh, fp
                delta, consistent = 0.0, True
            else:
                same_shape = len(mesh["node_ids"]) == len(reference["node_ids"])
                same_ids = same_shape and np.array_equal(mesh["node_ids"], reference["node_ids"])
                same_material = same_shape and np.array_equal(mesh["material_ids"], reference["material_ids"])
                same_side = same_shape and np.array_equal(mesh["interface_side"], reference["interface_side"])
                delta = float(np.max(np.abs(mesh["coordinates_m"] - reference["coordinates_m"]))) if same_shape else float("inf")
                consistent = same_ids and same_material and same_side and delta <= coordinate_tolerance_m
            accepted.append({"case_key": CaseKey.from_case_dir(case).value, "csv": str(thermal_csv(case)),
                             "node_count": len(mesh["node_ids"]), "mesh_fingerprint": fp,
                             "coordinate_max_delta_m": delta, "consistent_with_reference": consistent})
        except Exception as exc:
            failures.append({"case_dir": str(case), "reason": f"{type(exc).__name__}: {exc}"})
    return {"audit_version": 1, "coordinate_tolerance_m": coordinate_tolerance_m,
            "case_count": len(cases), "accepted_count": len(accepted), "failed_count": len(failures),
            "consistent_count": sum(bool(x["consistent_with_reference"]) for x in accepted),
            "reference_mesh_fingerprint": reference_fp, "cases": accepted, "failures": failures,
            "all_meshes_consistent": bool(accepted) and not failures and all(x["consistent_with_reference"] for x in accepted)}


def _read_receiver(path: Path, waveform_length: int) -> tuple[np.ndarray, int]:
    frame = pd.read_csv(path, comment="#")
    columns = [c for c in frame.columns if c not in {"step", "time_s"}]
    if not columns:
        raise ValueError(f"接收波形没有信号列: {path}")
    raw = frame[columns[0]].to_numpy(np.float32)
    return _resample_signal(raw, waveform_length, normalize=False), len(raw)


def build_case_dataset(raw_root: str | Path, output_dir: str | Path, *, target_points: int = 10000,
                       seed: int = 42, waveform_length: int = 2048,
                       prefer_corrected: bool = False, limit: int | None = None,
                       test_ratio: float = 0.2, validation_ratio: float = 0.1) -> Path:
    raw_root, output = Path(raw_root), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    audit = audit_meshes(raw_root)
    (output / "mesh_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    if not audit["all_meshes_consistent"]:
        raise ValueError("网格审计未通过；详情见 mesh_audit.json，禁止按 CSV 行号对齐")
    cases = discover_cases(raw_root)
    if limit is not None and limit >= 0:
        cases = cases[:limit]
    if not cases:
        raise ValueError("没有可建库的 case")
    reference = read_mesh_csv(thermal_csv(cases[0]))
    sampling = deterministic_sample(reference, target_points, seed)
    save_sampling_index(output / "sampling_index.npz", sampling)
    index_by_id = {int(v): i for i, v in enumerate(reference["node_ids"])}
    records, failures, temperatures = [], [], []
    wave_dir, field_dir = output / "waveforms", output / "temperature_fields"
    wave_dir.mkdir(exist_ok=True); field_dir.mkdir(exist_ok=True)
    for case in cases:
        try:
            pair = validate_case_pair(case)
            key = CaseKey.from_case_dir(case)
            waveform_path, waveform_version = select_waveform(case, prefer_corrected)
            waveform, original_length = _read_receiver(waveform_path, waveform_length)
            mesh = read_mesh_csv(thermal_csv(case))
            case_by_id = {int(v): i for i, v in enumerate(mesh["node_ids"])}
            rows = np.array([case_by_id[int(node)] for node in sampling["node_ids"]], np.int64)
            field = {"temperature_k": mesh["temperature_k"][rows], "coordinates_m": mesh["coordinates_m"][rows].astype(np.float32),
                     "node_ids": mesh["node_ids"][rows], "material_ids": mesh["material_ids"][rows],
                     "interface_side": mesh["interface_side"][rows], "sample_weights": sampling["sample_weights"],
                     "temperature_unit": "K", "coordinate_unit": "m"}
            validate_temperature_field(field)
            sample_name = key.value.replace("/", "_")
            wave_file, field_file = wave_dir / f"{sample_name}.npy", field_dir / f"{sample_name}.npz"
            np.save(wave_file, waveform); np.savez_compressed(field_file, **field)
            temperatures.append(field["temperature_k"])
            records.append({"sample_id": sample_name, "case_key": key.value, "source": "experiment_case",
                            "waveform_path": str(wave_file.relative_to(output)), "field_path": str(field_file.relative_to(output)),
                            "temperature_k": float(np.max(field["temperature_k"])), "material_key": "case_mesh",
                            "dimension": "2d", "mode": "steady",
                            "meta": {"schema_version": TEMPERATURE_FIELD_SCHEMA_VERSION, "temperature_unit": "K", "coordinate_unit": "m",
                                     "sampling_version": sampling["sampling_version"], "mesh_fingerprint": sampling["source_mesh_fingerprint"],
                                     "waveform_version": waveform_version, "original_waveform_path": str(waveform_path),
                                     "original_waveform_length": original_length, "temperature_csv_path": str(thermal_csv(case)),
                                     "ultrasonic_config_path": pair["ultrasonic_config_path"],
                                     "referenced_temperature_csv_path": pair["referenced_temperature_csv_path"]}})
        except Exception as exc:
            failures.append({"case_dir": str(case), "reason": f"{type(exc).__name__}: {exc}"})
    if failures:
        (output / "pairing_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError(f"{len(failures)} 个 case 配对失败；未静默跳过，详见 pairing_failures.json")
    stack = np.concatenate(temperatures).astype(np.float64)
    summary = {"min_k": float(stack.min()), "max_k": float(stack.max()),
               "case_count": len(records), "point_count": int(target_points)}
    payload = {"schema_version": TEMPERATURE_FIELD_SCHEMA_VERSION, "dataset_name": "case_temperature_field_db",
               "sampling_index": "sampling_index.npz", "temperature_summary": summary, "records": records}
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_case_split_files(manifest, seed=seed,
                           train_ratio=1.0-float(test_ratio)-float(validation_ratio),
                           validation_ratio=float(validation_ratio))
    return manifest
