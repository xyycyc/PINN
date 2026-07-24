"""Build traceable case-level waveform and sampled temperature-field artifacts."""

from __future__ import annotations

import json
from collections import Counter
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
from .case_adapters import (
    AdaptedCase,
    LEGACY_CASE_FORMAT,
    adapt_case_source,
    detect_case_source_format,
    expected_thermal_csv,
)
from .temperature_field import (
    TEMPERATURE_FIELD_SCHEMA_VERSION,
    mesh_fingerprint,
    select_waveform,
    validate_temperature_field,
)
from .dataset import write_case_split_files


# Fixed native-rate prefix length shared by every case.  The shortest trace in
# the current variable-CFL database has 3133 samples; 1097 samples preserve a
# useful leading window while keeping every batched tensor rectangular.  No
# interpolation, decimation, or window splitting is performed.
DEFAULT_WAVEFORM_CROP_LENGTH = 1097


def discover_cases(root: str | Path) -> list[Path]:
    """Discover supported flat and worker-nested physical case directories."""
    path = Path(root)
    candidates = [
        *(item for item in path.glob("case_*_T*K") if item.is_dir()),
        *(item for item in path.glob("worker_*/case_*_T*K") if item.is_dir()),
    ]
    return sorted(set(candidates), key=lambda item: str(item).casefold())


def thermal_csv(case_dir: Path) -> Path:
    """Resolve the temperature CSV referenced by a case configuration."""
    return expected_thermal_csv(case_dir)


def validate_case_pair(case_dir: Path) -> dict[str, Any]:
    """Validate the ultrasonic config reference against the mirrored local case."""
    adapted = adapt_case_source(case_dir)
    return {
        "case_key": adapted.case_key,
        "source_format": adapted.source_format,
        "ultrasonic_config_path": str(adapted.config_path),
        "referenced_temperature_csv_path": adapted.referenced_temperature_csv_path,
        "local_temperature_csv_path": str(adapted.thermal_path),
    }


def audit_meshes(root: str | Path, coordinate_tolerance_m: float = 1e-10) -> dict[str, Any]:
    """Verify that all accepted cases share a compatible fixed-node mesh contract."""
    cases, failures, accepted = discover_cases(root), [], []
    reference: dict[str, Any] | None = None
    reference_fp = ""
    for case in cases:
        try:
            adapted = adapt_case_source(case)
            mesh = read_mesh_csv(
                adapted.thermal_path,
                material_id_policy=adapted.material_id_policy,
            )
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
            accepted.append({"case_key": adapted.case_key, "csv": str(adapted.thermal_path),
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
    raw = normalize_waveform_frame(frame)
    waveform = crop_waveform_tail(raw, waveform_crop_length)
    crop_end_time_s: float | None = None
    if "time_s" in frame.columns and len(frame) >= len(waveform):
        crop_end_time_s = float(frame["time_s"].iloc[len(waveform) - 1])
    return waveform, len(raw), crop_end_time_s


def normalize_waveform_frame(frame: pd.DataFrame) -> np.ndarray:
    """Normalize supported one-column, one-row, and tabular signal layouts."""

    axis_names = {"step", "time", "time_s", "index", "sample", "sample_index"}
    columns = [
        column
        for column in frame.columns
        if str(column).strip().casefold() not in axis_names
        and not str(column).strip().casefold().startswith("unnamed:")
    ]
    if not columns:
        raise ValueError("接收波形没有信号列")
    if len(frame) == 1 and len(columns) > 1:
        raw_values = frame.loc[frame.index[0], columns].to_numpy()
    else:
        # Multi-channel exports keep receiver order from the source config.  The
        # established contract uses the first signal channel.
        raw_values = frame[columns[0]].to_numpy()
    numeric = pd.to_numeric(pd.Series(raw_values), errors="coerce").to_numpy(
        np.float32
    )
    if numeric.size == 0:
        raise ValueError("接收波形为空")
    if not np.isfinite(numeric).all():
        raise ValueError("接收波形包含非数值或非有限值")
    return numeric


def _failure_record(
    case: Path,
    material: str,
    stage: str,
    exc: Exception,
    *,
    adapted: AdaptedCase | None = None,
    waveform_path: Path | None = None,
) -> dict[str, str]:
    error_type = type(exc).__name__
    error_message = str(exc)
    return {
        "case_id": adapted.case_key if adapted is not None else case.name,
        "material": material,
        "source_format": (
            adapted.source_format
            if adapted is not None
            else detect_case_source_format(case)
        ),
        "thermal_path": str(
            adapted.thermal_path if adapted is not None else thermal_csv(case)
        ),
        "waveform_path": str(
            waveform_path
            if waveform_path is not None
            else case / "ultrasonic" / "receiver_signal.csv"
        ),
        "stage": stage,
        "error_type": error_type,
        "error_message": error_message,
        "case_dir": str(case),
        "reason": f"{error_type}: {error_message}",
    }


def _write_and_raise_failures(
    output: Path,
    failures: list[dict[str, str]],
) -> None:
    failure_path = output / "pairing_failures.json"
    failure_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    first = failures[0]
    print(
        "首个失败 case: {case_id}\n"
        "失败阶段: {stage}\n"
        "异常类型: {error_type}\n"
        "异常信息: {error_message}\n"
        "pairing_failures.json: {failure_path}".format(
            failure_path=failure_path,
            **first,
        )
    )
    counts = Counter(item["error_type"] for item in failures)
    print("失败原因汇总:")
    for error_type, count in sorted(counts.items()):
        print(f"{error_type}: {count}")
    raise ValueError(
        f"{len(failures)} 个 case 建库失败；未静默跳过，详见 {failure_path}"
    )


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
    """Build, audit, sample, and split a fixed-node temperature-field dataset."""
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
    failures: list[dict[str, str]] = []
    adapted_cases: list[AdaptedCase] = []
    for case in cases:
        try:
            adapted_cases.append(adapt_case_source(case))
        except Exception as exc:
            failures.append(
                _failure_record(
                    case,
                    route_material_key,
                    "source_adaptation",
                    exc,
                )
            )
    if failures:
        _write_and_raise_failures(output, failures)

    reference_case = adapted_cases[0]
    try:
        reference = read_mesh_csv(
            reference_case.thermal_path,
            material_id_policy=reference_case.material_id_policy,
        )
    except Exception as exc:
        failures.append(
            _failure_record(
                reference_case.case_dir,
                route_material_key,
                "mesh_read",
                exc,
                adapted=reference_case,
            )
        )
        _write_and_raise_failures(output, failures)
    reference_fp = mesh_fingerprint(
        reference["node_ids"],
        reference["coordinates_m"],
        reference["material_ids"],
        reference["interface_side"],
    )
    try:
        sampling = deterministic_sample(
            reference,
            target_points,
            seed,
            interface_overflow_policy=(
                "error"
                if reference_case.source_format == LEGACY_CASE_FORMAT
                else "paired_stratified"
            ),
        )
    except Exception as exc:
        failures.append(
            _failure_record(
                reference_case.case_dir,
                route_material_key,
                "mesh_sampling",
                exc,
                adapted=reference_case,
            )
        )
        _write_and_raise_failures(output, failures)
    save_sampling_index(output / "sampling_index.npz", sampling)
    sample_rows = np.asarray(sampling["indices"], dtype=np.int64)
    records: list[dict[str, Any]] = []
    audit_cases: list[dict[str, Any]] = []
    audit_failures: list[dict[str, str]] = []
    temperature_min_k = float("inf")
    temperature_max_k = float("-inf")
    wave_dir, field_dir = output / "waveforms", output / "temperature_fields"
    wave_dir.mkdir(exist_ok=True); field_dir.mkdir(exist_ok=True)
    iterator = tqdm(cases, desc="build-case-database", unit="case", disable=len(cases) < 2)
    for position, (case, adapted) in enumerate(zip(iterator, adapted_cases)):
        mesh_audited = False
        stage = "mesh_read"
        waveform_path: Path | None = None
        try:
            mesh = (
                reference
                if position == 0
                else read_mesh_csv(
                    adapted.thermal_path,
                    material_id_policy=adapted.material_id_policy,
                )
            )
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
                    "case_key": adapted.case_key,
                    "csv": str(adapted.thermal_path),
                    "node_count": len(mesh["node_ids"]),
                    "mesh_fingerprint": fp,
                    "constituent_material_catalog": mesh["constituent_material_catalog"],
                    "coordinate_max_delta_m": coordinate_delta,
                    "consistent_with_reference": consistent,
                }
            )
            mesh_audited = True
            stage = "mesh_consistency"
            if not consistent:
                raise ValueError("网格与参考 case 不一致，禁止按行号或节点索引对齐")
            pair = {
                "ultrasonic_config_path": str(adapted.config_path),
                "referenced_temperature_csv_path": (
                    adapted.referenced_temperature_csv_path
                ),
            }
            stage = "waveform_discovery"
            waveform_path, waveform_version = select_waveform(case, prefer_corrected)
            stage = "waveform_read"
            waveform, original_length, crop_end_time_s = _read_receiver(
                waveform_path,
                waveform_crop_length,
            )
            stage = "temperature_field_validation"
            field = {"temperature_k": mesh["temperature_k"][sample_rows], "coordinates_m": sampling["coordinates_m"].astype(np.float32),
                     "node_ids": sampling["node_ids"], "material_ids": sampling["material_ids"],
                     "interface_side": sampling["interface_side"], "sample_weights": sampling["sample_weights"],
                     "temperature_unit": "K", "coordinate_unit": "m"}
            validate_temperature_field(field)
            sample_name = adapted.case_key.replace("/", "_")
            wave_file, field_file = wave_dir / f"{sample_name}.npy", field_dir / f"{sample_name}.npz"
            stage = "artifact_write"
            np.save(wave_file, waveform); np.savez_compressed(field_file, **field)
            temperature_min_k = min(temperature_min_k, float(np.min(field["temperature_k"])))
            temperature_max_k = max(temperature_max_k, float(np.max(field["temperature_k"])))
            meta = {"schema_version": TEMPERATURE_FIELD_SCHEMA_VERSION, "temperature_unit": "K", "coordinate_unit": "m",
                    "sampling_version": sampling["sampling_version"], "mesh_fingerprint": sampling["source_mesh_fingerprint"],
                    "sample_material_key": route_material_key,
                    "sample_material_name": route_material_name,
                    "waveform_version": waveform_version, "original_waveform_path": str(waveform_path),
                    "waveform_processing": "fixed_prefix_native_rate",
                    "waveform_crop_length": int(waveform_crop_length),
                    "original_waveform_length": original_length,
                    "cropped_waveform_length": int(len(waveform)),
                    "crop_end_time_s": crop_end_time_s,
                    "temperature_csv_path": str(adapted.thermal_path),
                    "ultrasonic_config_path": pair["ultrasonic_config_path"],
                    "referenced_temperature_csv_path": pair["referenced_temperature_csv_path"]}
            if adapted.source_format != LEGACY_CASE_FORMAT:
                meta["source_format"] = adapted.source_format
                meta["material_id_policy"] = adapted.material_id_policy
            records.append({"sample_id": sample_name, "case_key": adapted.case_key, "source": "experiment_case",
                            "waveform_path": str(wave_file.relative_to(output)), "field_path": str(field_file.relative_to(output)),
                            "temperature_k": float(np.max(field["temperature_k"])),
                            "material_key": route_material_key,
                            "dimension": "2d", "mode": "steady",
                            "meta": meta})
        except Exception as exc:
            failures.append(
                _failure_record(
                    case,
                    route_material_key,
                    stage,
                    exc,
                    adapted=adapted,
                    waveform_path=waveform_path,
                )
            )
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
        _write_and_raise_failures(output, failures)
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
