"""Build traceable case-level waveform and sampled temperature-field artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from ..artifact_paths import (
    validate_artifact_basename,
    validate_distinct_artifact_basenames,
)
from .mesh_sampling import deterministic_sample, read_mesh_csv, save_sampling_index
from .temperature_field import (
    CaseKey, TEMPERATURE_FIELD_SCHEMA_VERSION, mesh_fingerprint,
    select_waveform, validate_temperature_field,
)
from .dataset import write_case_split_files


# Fixed native-rate prefix length shared by every case.  The shortest trace in
# the current variable-CFL database has 3133 samples; 1097 samples preserve a
# useful leading window while keeping every batched tensor rectangular.  No
# interpolation, decimation, or window splitting is performed.
DEFAULT_WAVEFORM_CROP_LENGTH = 1097


def discover_cases(root: str | Path) -> list[Path]:
    path = Path(root)
    candidates = [
        *(item for item in path.glob("case_*_T*K") if item.is_dir()),
        *(item for item in path.glob("worker_*/case_*_T*K") if item.is_dir()),
    ]
    return sorted(set(candidates), key=lambda item: str(item).casefold())


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
    reference: dict[str, Any] | None = None
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
                same_material_catalog = (
                    mesh["constituent_material_catalog"]
                    == reference["constituent_material_catalog"]
                )
                same_side = same_shape and np.array_equal(mesh["interface_side"], reference["interface_side"])
                delta = float(np.max(np.abs(mesh["coordinates_m"] - reference["coordinates_m"]))) if same_shape else float("inf")
                consistent = same_ids and same_material and same_material_catalog and same_side and delta <= coordinate_tolerance_m
            accepted.append({"case_key": CaseKey.from_case_dir(case).value, "csv": str(thermal_csv(case)),
                             "node_count": len(mesh["node_ids"]), "mesh_fingerprint": fp,
                             "constituent_material_catalog": mesh["constituent_material_catalog"],
                             "coordinate_max_delta_m": delta, "consistent_with_reference": consistent})
        except Exception as exc:
            failures.append({"case_dir": str(case), "reason": f"{type(exc).__name__}: {exc}"})
    return {"audit_version": 1, "coordinate_tolerance_m": coordinate_tolerance_m,
            "case_count": len(cases), "accepted_count": len(accepted), "failed_count": len(failures),
            "consistent_count": sum(bool(x["consistent_with_reference"]) for x in accepted),
            "reference_mesh_fingerprint": reference_fp, "cases": accepted, "failures": failures,
            "constituent_material_catalog": (
                reference["constituent_material_catalog"] if reference is not None else []
            ),
            "all_meshes_consistent": bool(accepted) and not failures and all(x["consistent_with_reference"] for x in accepted)}


def crop_waveform_tail(
    values: np.ndarray,
    crop_length: int = DEFAULT_WAVEFORM_CROP_LENGTH,
) -> np.ndarray:
    """Keep a contiguous native-rate prefix; never resample or split a trace."""
    length = int(crop_length)
    if length <= 0:
        raise ValueError(f"waveform_crop_length must be positive, got {crop_length}")
    raw = np.asarray(values, dtype=np.float32).reshape(-1)
    if raw.size == 0:
        raise ValueError("receiver waveform is empty")
    if raw.size < length:
        raise ValueError(
            f"receiver waveform has {raw.size} samples, fewer than waveform_crop_length={length}"
        )
    return raw[:length].copy()


def _read_receiver(
    path: Path,
    waveform_crop_length: int,
) -> tuple[np.ndarray, int, float | None]:
    frame = pd.read_csv(path, comment="#")
    columns = [c for c in frame.columns if c not in {"step", "time_s"}]
    if not columns:
        raise ValueError(f"接收波形没有信号列: {path}")
    raw = frame[columns[0]].to_numpy(np.float32)
    waveform = crop_waveform_tail(raw, waveform_crop_length)
    crop_end_time_s: float | None = None
    if "time_s" in frame.columns:
        crop_end_time_s = float(frame["time_s"].iloc[len(waveform) - 1])
    return waveform, len(raw), crop_end_time_s


def build_case_dataset(raw_root: str | Path, output_dir: str | Path, *, target_points: int = 10000,
                       seed: int = 42, waveform_crop_length: int = DEFAULT_WAVEFORM_CROP_LENGTH,
                       waveform_length: int | None = None,
                       prefer_corrected: bool = False, limit: int | None = None,
                       test_ratio: float = 0.2, validation_ratio: float = 0.1,
                       dataset_label: str | None = None,
                       sample_material_key: str | None = None,
                       sample_material_name: str | None = None,
                       train_manifest_name: str = "train_manifest.json",
                       validation_manifest_name: str = "validation_manifest.json",
                       test_manifest_name: str = "test_manifest.json") -> Path:
    if waveform_length is not None:
        raise ValueError(
            "waveform_length-based resampling is disabled for case datasets; "
            "use waveform_crop_length to keep a fixed native-rate prefix"
        )
    if int(waveform_crop_length) <= 0:
        raise ValueError("waveform_crop_length must be positive")
    train_manifest_name = validate_artifact_basename(
        str(train_manifest_name).strip() or "train_manifest.json",
        label="训练 manifest 文件名",
        suffix=".json",
    )
    validation_manifest_name = validate_artifact_basename(
        str(validation_manifest_name).strip() or "validation_manifest.json",
        label="验证 manifest 文件名",
        suffix=".json",
    )
    test_manifest_name = validate_artifact_basename(
        str(test_manifest_name).strip() or "test_manifest.json",
        label="测试 manifest 文件名",
        suffix=".json",
    )
    validate_distinct_artifact_basenames(
        (train_manifest_name, validation_manifest_name, test_manifest_name),
        label="训练/验证/测试 manifest 文件名",
        reserved=("manifest.json", "split_config.json"),
    )
    raw_root, output = Path(raw_root), Path(output_dir)
    route_material_key = str(sample_material_key or raw_root.name).strip()
    if not route_material_key:
        raise ValueError("样本级材料路由字段不能为空")
    route_material_name = str(sample_material_name or route_material_key).strip()
    cases = discover_cases(raw_root)
    if limit is not None and limit >= 0:
        cases = cases[:limit]
    if not cases:
        raise ValueError("没有可建库的 case")
    output.mkdir(parents=True, exist_ok=True)
    reference = read_mesh_csv(thermal_csv(cases[0]))
    reference_fp = mesh_fingerprint(
        reference["node_ids"],
        reference["coordinates_m"],
        reference["material_ids"],
        reference["interface_side"],
    )
    sampling = deterministic_sample(reference, target_points, seed)
    save_sampling_index(output / "sampling_index.npz", sampling)
    sample_rows = np.asarray(sampling["indices"], dtype=np.int64)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    audit_cases: list[dict[str, Any]] = []
    audit_failures: list[dict[str, str]] = []
    temperature_min_k = float("inf")
    temperature_max_k = float("-inf")
    wave_dir, field_dir = output / "waveforms", output / "temperature_fields"
    wave_dir.mkdir(exist_ok=True); field_dir.mkdir(exist_ok=True)
    iterator = tqdm(cases, desc="build-case-database", unit="case", disable=len(cases) < 2)
    for position, case in enumerate(iterator):
        mesh_audited = False
        try:
            key = CaseKey.from_case_dir(case)
            mesh = reference if position == 0 else read_mesh_csv(thermal_csv(case))
            fp = mesh_fingerprint(
                mesh["node_ids"],
                mesh["coordinates_m"],
                mesh["material_ids"],
                mesh["interface_side"],
            )
            same_shape = len(mesh["node_ids"]) == len(reference["node_ids"])
            same_ids = same_shape and np.array_equal(mesh["node_ids"], reference["node_ids"])
            same_material = same_shape and np.array_equal(mesh["material_ids"], reference["material_ids"])
            same_material_catalog = (
                mesh["constituent_material_catalog"]
                == reference["constituent_material_catalog"]
            )
            same_side = same_shape and np.array_equal(mesh["interface_side"], reference["interface_side"])
            coordinate_delta = (
                float(np.max(np.abs(mesh["coordinates_m"] - reference["coordinates_m"])))
                if same_shape
                else float("inf")
            )
            consistent = (
                same_ids
                and same_material
                and same_material_catalog
                and same_side
                and coordinate_delta <= 1e-10
                and fp == reference_fp
            )
            audit_cases.append(
                {
                    "case_key": key.value,
                    "csv": str(thermal_csv(case)),
                    "node_count": len(mesh["node_ids"]),
                    "mesh_fingerprint": fp,
                    "constituent_material_catalog": mesh["constituent_material_catalog"],
                    "coordinate_max_delta_m": coordinate_delta,
                    "consistent_with_reference": consistent,
                }
            )
            mesh_audited = True
            if not consistent:
                raise ValueError("网格与参考 case 不一致，禁止按行号或节点索引对齐")
            pair = validate_case_pair(case)
            waveform_path, waveform_version = select_waveform(case, prefer_corrected)
            waveform, original_length, crop_end_time_s = _read_receiver(
                waveform_path,
                waveform_crop_length,
            )
            field = {"temperature_k": mesh["temperature_k"][sample_rows], "coordinates_m": sampling["coordinates_m"].astype(np.float32),
                     "node_ids": sampling["node_ids"], "material_ids": sampling["material_ids"],
                     "interface_side": sampling["interface_side"], "sample_weights": sampling["sample_weights"],
                     "temperature_unit": "K", "coordinate_unit": "m"}
            validate_temperature_field(field)
            sample_name = key.value.replace("/", "_")
            wave_file, field_file = wave_dir / f"{sample_name}.npy", field_dir / f"{sample_name}.npz"
            np.save(wave_file, waveform); np.savez_compressed(field_file, **field)
            temperature_min_k = min(temperature_min_k, float(np.min(field["temperature_k"])))
            temperature_max_k = max(temperature_max_k, float(np.max(field["temperature_k"])))
            records.append({"sample_id": sample_name, "case_key": key.value, "source": "experiment_case",
                            "waveform_path": str(wave_file.relative_to(output)), "field_path": str(field_file.relative_to(output)),
                            "temperature_k": float(np.max(field["temperature_k"])),
                            "material_key": route_material_key,
                            "dimension": "2d", "mode": "steady",
                            "meta": {"schema_version": TEMPERATURE_FIELD_SCHEMA_VERSION, "temperature_unit": "K", "coordinate_unit": "m",
                                     "sampling_version": sampling["sampling_version"], "mesh_fingerprint": sampling["source_mesh_fingerprint"],
                                     "sample_material_key": route_material_key,
                                     "sample_material_name": route_material_name,
                                     "waveform_version": waveform_version, "original_waveform_path": str(waveform_path),
                                     "waveform_processing": "fixed_prefix_native_rate",
                                     "waveform_crop_length": int(waveform_crop_length),
                                     "original_waveform_length": original_length,
                                     "cropped_waveform_length": int(len(waveform)),
                                     "crop_end_time_s": crop_end_time_s,
                                     "temperature_csv_path": str(thermal_csv(case)),
                                     "ultrasonic_config_path": pair["ultrasonic_config_path"],
                                     "referenced_temperature_csv_path": pair["referenced_temperature_csv_path"]}})
        except Exception as exc:
            failures.append({"case_dir": str(case), "reason": f"{type(exc).__name__}: {exc}"})
            if not mesh_audited:
                audit_failures.append(
                    {"case_dir": str(case), "reason": f"{type(exc).__name__}: {exc}"}
                )
    audit = {
        "audit_version": 1,
        "coordinate_tolerance_m": 1e-10,
        "case_count": len(cases),
        "accepted_count": len(audit_cases),
        "failed_count": len(audit_failures),
        "consistent_count": sum(bool(item["consistent_with_reference"]) for item in audit_cases),
        "reference_mesh_fingerprint": reference_fp,
        "cases": audit_cases,
        "failures": audit_failures,
        "constituent_material_catalog": reference["constituent_material_catalog"],
        "all_meshes_consistent": (
            bool(audit_cases)
            and not audit_failures
            and all(bool(item["consistent_with_reference"]) for item in audit_cases)
        ),
    }
    (output / "mesh_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        (output / "pairing_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
        raise ValueError(f"{len(failures)} 个 case 建库失败；未静默跳过，详见 pairing_failures.json")
    if not records:
        raise ValueError("没有成功生成任何 case 记录")
    summary = {"min_k": temperature_min_k, "max_k": temperature_max_k,
               "case_count": len(records), "point_count": int(target_points)}
    payload = {"schema_version": TEMPERATURE_FIELD_SCHEMA_VERSION, "dataset_name": "case_temperature_field_db",
               "dataset_label": str(dataset_label or "").strip(),
               "waveform_contract": {
                   "processing": "fixed_prefix_native_rate",
                   "crop_length": int(waveform_crop_length),
                   "resampled": False,
               },
               "sample_material_catalog": [{
                   "material_key": route_material_key,
                   "material_name": route_material_name,
                   "case_count": len(records),
               }],
               "constituent_material_catalog": sampling["constituent_material_catalog"],
               "sampling_index": "sampling_index.npz", "temperature_summary": summary, "records": records}
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_case_split_files(
        manifest,
        seed=seed,
        train_ratio=1.0 - float(test_ratio) - float(validation_ratio),
        validation_ratio=float(validation_ratio),
        train_name=str(train_manifest_name).strip() or "train_manifest.json",
        validation_name=str(validation_manifest_name).strip() or "validation_manifest.json",
        test_name=str(test_manifest_name).strip() or "test_manifest.json",
    )
    return manifest
