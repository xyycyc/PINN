"""Build legacy or inference-only datasets from post-zero waveform CSV files."""

from __future__ import annotations


# Build an inference-only wumu_exp database compatible with a mixed checkpoint:
# python .\database\raw\waveforms_by_file\build_post0_database.py `
#   --source-dir .\database\raw\waveforms_by_file `
#   --output-dir E:\pinn_data\data_process\wumu_exp_inference `
#   --dataset-name wumu_exp_inference `
#   --material-key wumu `
#   --signal-column amplitude_filtered_residual `
#   --time-column time_s `
#   --time-min-s 0.0 `
#   --inference-only `
#   --reference-manifest E:\pinn_data\data_process\metal_silicon_wumu_multi_material_temperature_field\mixed_train_manifest.json `
#   --test-manifest-name wumu_exp_test_manifest.json
#
# --reference-manifest must be the training manifest used by the checkpoint:
# use mixed_train_manifest.json for one shared mixed checkpoint, or use the
# wumu train_manifest.json for a separately trained wumu checkpoint.


import argparse
import ast
import csv
import io
import json
import math
import random
import re
import struct
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DATASET_DIR_NAME = "metal_matrix_waveforms_post0"
DEFAULT_MATERIAL_KEY = "metal_matrix"
DEFAULT_SIGNAL_COLUMN = "amplitude_filtered_residual"


@dataclass
class LoadedWaveform:
    """One parsed waveform with its filename-derived temperature and provenance."""

    csv_path: Path
    temperature_c: float
    signal: list[float]
    meta: dict[str, Any]


@dataclass
class PointFieldReference:
    """Compatibility contract copied from the checkpoint's training manifest."""

    manifest_path: Path
    payload: dict[str, Any]
    sampling_path: Path
    sampling_entries: dict[str, bytes]
    point_count: int
    waveform_length: int


def _script_source_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_output_dir() -> Path:
    database_root = _script_source_dir().parents[1]
    return database_root / "data_process" / DEFAULT_DATASET_DIR_NAME


def _extract_temperature_c(file_name: str) -> float:
    match = re.search(r"^T(?P<temp>[-+]?\d+(?:\.\d+)?)C(?:_|$)", file_name, flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Cannot extract temperature from file name: {file_name}")
    return float(match.group("temp"))


def _sanitize_id(value: str) -> str:
    text = re.sub(r"[^\w.\-]+", "_", str(value).strip())
    return text.strip("._") or "sample"


def _resample_signal(values: list[float], length: int) -> list[float]:
    if length <= 0:
        raise ValueError("target length must be positive")
    values = [float(item) for item in values]
    if not values:
        return [0.0] * int(length)
    if len(values) == int(length):
        return list(values)
    if int(length) == 1:
        return [values[0]]
    if len(values) == 1:
        return [values[0]] * int(length)

    scale = (len(values) - 1) / float(int(length) - 1)
    result: list[float] = []
    for idx in range(int(length)):
        position = idx * scale
        left = int(math.floor(position))
        right = min(left + 1, len(values) - 1)
        weight = position - left
        result.append(values[left] * (1.0 - weight) + values[right] * weight)
    return result


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = _mean(values)
    return math.sqrt(sum((item - mean) ** 2 for item in values) / len(values))


def _median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _shape_repr(shape: tuple[int, ...]) -> str:
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(item) for item in shape) + ")"


def _encode_npy_float32(values: list[float], shape: tuple[int, ...]) -> bytes:
    expected = math.prod(shape)
    if len(values) != expected:
        raise ValueError(f"shape {shape} expects {expected} values, got {len(values)}")

    header = "{'descr': '<f4', 'fortran_order': False, 'shape': " + _shape_repr(shape) + ", }"
    preamble_len = 10
    padding = 16 - ((preamble_len + len(header) + 1) % 16)
    if padding == 16:
        padding = 0
    header_bytes = (header + (" " * padding) + "\n").encode("latin1")

    return b"".join(
        (
            b"\x93NUMPY",
            bytes([1, 0]),
            struct.pack("<H", len(header_bytes)),
            header_bytes,
            struct.pack(f"<{len(values)}f", *[float(item) for item in values]),
        )
    )


def _write_npy_float32(path: Path, values: list[float], shape: tuple[int, ...]) -> None:
    path.write_bytes(_encode_npy_float32(values, shape))


def _read_npy_shape_from_stream(handle: Any) -> tuple[int, ...]:
    if handle.read(6) != b"\x93NUMPY":
        raise ValueError("Not a NumPy .npy payload")
    version = handle.read(2)
    if len(version) != 2:
        raise ValueError("Truncated NumPy version header")
    major = int(version[0])
    if major == 1:
        header_length_raw = handle.read(2)
        if len(header_length_raw) != 2:
            raise ValueError("Truncated NumPy v1 header length")
        header_length = struct.unpack("<H", header_length_raw)[0]
        encoding = "latin1"
    elif major in {2, 3}:
        header_length_raw = handle.read(4)
        if len(header_length_raw) != 4:
            raise ValueError("Truncated NumPy v2/v3 header length")
        header_length = struct.unpack("<I", header_length_raw)[0]
        encoding = "utf-8" if major == 3 else "latin1"
    else:
        raise ValueError(f"Unsupported NumPy .npy version: {major}.{int(version[1])}")
    header = ast.literal_eval(handle.read(header_length).decode(encoding).strip())
    shape = header.get("shape")
    if not isinstance(shape, tuple):
        raise ValueError("NumPy header does not contain a tuple shape")
    return tuple(int(item) for item in shape)


def _read_npy_shape(path: Path) -> tuple[int, ...]:
    with path.open("rb") as handle:
        return _read_npy_shape_from_stream(handle)


def _load_one_csv(
    csv_path: Path,
    *,
    signal_column: str,
    time_column: str,
    time_min_s: float,
) -> LoadedWaveform:
    temperature_c = _extract_temperature_c(csv_path.name)
    signal: list[float] = []
    post_time_s: list[float] = []
    time_us: list[float] = []
    original_length = 0

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {time_column, "time_us", signal_column}
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(f"{csv_path.name} is missing required columns: {missing}")
        for row in reader:
            original_length += 1
            current_time_s = float(row[time_column])
            if current_time_s < float(time_min_s):
                continue
            post_time_s.append(current_time_s)
            time_us.append(float(row["time_us"]))
            signal.append(float(row[signal_column]))

    if not signal:
        raise ValueError(f"{csv_path.name} has no samples with {time_column} >= {time_min_s}")

    meta = {
        "original_csv": str(csv_path.resolve()),
        "original_length": int(original_length),
        "post0_length": int(len(signal)),
        "time_column": time_column,
        "time_min_s": float(time_min_s),
        "first_post_time_s": post_time_s[0],
        "last_post_time_s": post_time_s[-1],
        "first_post_time_us": time_us[0],
        "last_post_time_us": time_us[-1],
        "signal_column": signal_column,
        "signal_mean": _mean(signal),
        "signal_std": _std(signal),
        "signal_peak_abs": max(abs(item) for item in signal),
    }
    if len(signal) > 1:
        meta["dt_us_median"] = _median([time_us[idx] - time_us[idx - 1] for idx in range(1, len(time_us))])
    return LoadedWaveform(
        csv_path=csv_path,
        temperature_c=temperature_c,
        signal=signal,
        meta=meta,
    )


def _load_waveforms(
    source_dir: Path,
    *,
    signal_column: str,
    time_column: str,
    time_min_s: float,
    limit: int | None,
) -> list[LoadedWaveform]:
    csv_files = sorted(source_dir.glob("*.csv"), key=lambda item: (_extract_temperature_c(item.name), item.name))
    if limit is not None:
        csv_files = csv_files[: max(0, int(limit))]
    if not csv_files:
        raise ValueError(f"No CSV files found in {source_dir}")
    return [
        _load_one_csv(
            csv_path,
            signal_column=signal_column,
            time_column=time_column,
            time_min_s=time_min_s,
        )
        for csv_path in csv_files
    ]


def _split_records_by_temperature(
    records: list[dict[str, Any]],
    *,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not (0.0 < float(test_ratio) < 1.0):
        raise ValueError("test_ratio must be in (0, 1)")
    groups: dict[float, list[dict[str, Any]]] = {}
    for record in records:
        temp_c = float(record["meta"]["temperature_c"])
        groups.setdefault(temp_c, []).append(record)

    shuffled = sorted(groups)
    random.Random(int(seed)).shuffle(shuffled)

    test_group_count = int(round(len(shuffled) * float(test_ratio)))
    if len(shuffled) >= 2:
        test_group_count = min(max(test_group_count, 1), len(shuffled) - 1)
    else:
        test_group_count = 0
    test_temperatures = set(float(item) for item in shuffled[:test_group_count])

    train_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []
    for temperature in sorted(groups):
        target = test_records if float(temperature) in test_temperatures else train_records
        target.extend(groups[temperature])

    stats = {
        "total_records": len(records),
        "train_records": len(train_records),
        "test_records": len(test_records),
        "test_ratio": float(test_ratio),
        "seed": int(seed),
        "split_strategy": "temperature_group",
        "temperature_group_count": len(groups),
        "test_temperature_c": sorted(test_temperatures),
        "duplicate_temperature_c": {
            str(temp): count
            for temp, count in sorted(Counter(float(r["meta"]["temperature_c"]) for r in records).items())
            if count > 1
        },
        "train_source_breakdown": {"experiment": len(train_records)},
        "test_source_breakdown": {"experiment": len(test_records)},
    }
    return train_records, test_records, stats


def _write_manifest(path: Path, dataset_name: str, records: list[dict[str, Any]]) -> None:
    payload = {
        "dataset_name": dataset_name,
        "records": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _resolve_manifest_path(raw_path: str | Path, manifest_path: Path) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_path.parent / path
    return path.resolve()


def _load_point_field_reference(manifest_path: str | Path) -> PointFieldReference:
    resolved_manifest = Path(manifest_path).resolve()
    if not resolved_manifest.is_file():
        raise FileNotFoundError(f"Reference manifest does not exist: {resolved_manifest}")

    payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", 0)) < 1 or not payload.get("sampling_index"):
        raise ValueError(
            "Reference manifest must be a fixed-node point-field manifest with "
            "schema_version >= 1 and sampling_index."
        )

    normalization = dict(payload.get("normalization", {}))
    if normalization.get("fit_split") != "train":
        raise ValueError("Reference manifest normalization must be fitted from the training split.")
    if float(normalization.get("std_k", 0.0)) <= 0.0:
        raise ValueError("Reference manifest normalization std_k must be positive.")

    sampling_path = _resolve_manifest_path(str(payload["sampling_index"]), resolved_manifest)
    required_sampling_keys = (
        "node_ids",
        "coordinates_m",
        "material_ids",
        "interface_side",
        "sample_weights",
    )
    if not sampling_path.is_file():
        raise FileNotFoundError(f"Reference sampling index does not exist: {sampling_path}")
    with zipfile.ZipFile(sampling_path, "r") as sampling_file:
        archive_names = set(sampling_file.namelist())
        missing = [key for key in required_sampling_keys if f"{key}.npy" not in archive_names]
        if missing:
            raise ValueError(f"Reference sampling index is missing keys: {missing}")
        sampling_entries = {
            key: sampling_file.read(f"{key}.npy")
            for key in required_sampling_keys
        }
    node_shape = _read_npy_shape_from_stream(io.BytesIO(sampling_entries["node_ids"]))
    if len(node_shape) != 1 or node_shape[0] <= 0:
        raise ValueError(f"Reference node_ids must be a non-empty 1D array, got {node_shape}")
    point_count = int(node_shape[0])

    records = [dict(item) for item in payload.get("records", [])]
    if not records:
        raise ValueError("Reference manifest has no records, so waveform length cannot be determined.")
    waveform_lengths: set[int] = set()
    for record in records:
        waveform_path = _resolve_manifest_path(str(record["waveform_path"]), resolved_manifest)
        waveform_shape = _read_npy_shape(waveform_path)
        if not waveform_shape:
            raise ValueError(f"Reference waveform has scalar shape: {waveform_path}")
        waveform_lengths.add(int(waveform_shape[-1]))
    if len(waveform_lengths) != 1:
        raise ValueError(
            f"Reference manifest contains inconsistent waveform lengths: {sorted(waveform_lengths)}"
        )

    return PointFieldReference(
        manifest_path=resolved_manifest,
        payload=payload,
        sampling_path=sampling_path,
        sampling_entries=sampling_entries,
        point_count=point_count,
        waveform_length=next(iter(waveform_lengths)),
    )


def _write_point_field(
    path: Path,
    *,
    temperature_k: float,
    reference: PointFieldReference,
) -> None:
    temperature_values = [float(temperature_k)] * int(reference.point_count)
    with zipfile.ZipFile(
        path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as field_file:
        field_file.writestr(
            "temperature_k.npy",
            _encode_npy_float32(temperature_values, (int(reference.point_count),)),
        )
        for key, payload in reference.sampling_entries.items():
            field_file.writestr(f"{key}.npy", payload)


def _write_inference_manifest(
    path: Path,
    *,
    dataset_name: str,
    records: list[dict[str, Any]],
    reference: PointFieldReference,
    material_key: str,
) -> None:
    reference_payload = reference.payload
    temperatures = [float(record["temperature_k"]) for record in records]
    payload = {
        "schema_version": int(reference_payload["schema_version"]),
        "dataset_name": dataset_name,
        "dataset_label": str(reference_payload.get("dataset_label", dataset_name)),
        "inference_only": True,
        "has_spatial_ground_truth": False,
        "reference_manifest": str(reference.manifest_path),
        "sample_material_catalog": [
            {
                "material_key": material_key,
                "material_name": material_key,
                "case_count": len(records),
            }
        ],
        "constituent_material_catalog": list(
            reference_payload.get("constituent_material_catalog", [])
        ),
        "constituent_material_catalog_by_sample_material": dict(
            reference_payload.get("constituent_material_catalog_by_sample_material", {})
        ),
        "sampling_index": str(reference.sampling_path),
        "normalization": dict(reference_payload["normalization"]),
        "split": {
            "name": "test",
            "version": 1,
            "source": "wumu_exp_inference_only",
        },
        "temperature_summary": {
            "min_k": min(temperatures),
            "max_k": max(temperatures),
            "case_count": len(records),
            "point_count": int(reference.point_count),
            "spatial_label": "uniform_temperature_proxy_from_file_name",
        },
        "records": records,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_database(args: argparse.Namespace) -> dict[str, Any]:
    """Build the command-line selected legacy or inference-only database variant."""
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    material_key = str(args.material_key).strip() or DEFAULT_MATERIAL_KEY
    dataset_name = str(args.dataset_name).strip() or f"ai_model_{_sanitize_id(output_dir.name)}"
    inference_only = bool(args.inference_only)
    reference: PointFieldReference | None = None
    if inference_only:
        if args.reference_manifest is None:
            raise ValueError("--inference-only requires --reference-manifest")
        reference = _load_point_field_reference(args.reference_manifest)
        target_length = int(reference.waveform_length)
    else:
        if args.reference_manifest is not None:
            raise ValueError("--reference-manifest is only valid together with --inference-only")
        target_length = int(args.target_length)
    if target_length <= 0:
        raise ValueError("target waveform length must be positive")

    loaded = _load_waveforms(
        source_dir,
        signal_column=str(args.signal_column),
        time_column=str(args.time_column),
        time_min_s=float(args.time_min_s),
        limit=args.limit,
    )
    post0_lengths = [len(item.signal) for item in loaded]
    if args.post0_length is None:
        cut_length = min(post0_lengths)
    else:
        cut_length = int(args.post0_length)
        too_short = [item.csv_path.name for item in loaded if len(item.signal) < cut_length]
        if too_short:
            raise ValueError(
                f"post0_length={cut_length} is longer than {len(too_short)} file(s), "
                f"first too-short file: {too_short[0]}"
            )

    wave_dir = output_dir / "waveforms"
    field_dir = output_dir / "fields"
    wave_dir.mkdir(parents=True, exist_ok=True)
    field_dir.mkdir(parents=True, exist_ok=True)

    safe_material = _sanitize_id(material_key)
    records: list[dict[str, Any]] = []
    for idx, item in enumerate(loaded):
        truncated = item.signal[:cut_length]
        waveform = _resample_signal(truncated, target_length)
        temperature_k = float(item.temperature_c) + 273.15

        sample_id = f"exp_{safe_material}_post0_{idx:05d}"
        wave_path = wave_dir / f"{sample_id}.npy"
        _write_npy_float32(wave_path, waveform, (target_length,))
        if reference is None:
            field_path = field_dir / f"{sample_id}.npy"
            field = [0.0] * int(args.field_grid_1d)
            _write_npy_float32(field_path, field, (1, int(args.field_grid_1d)))
            source = "experiment"
            dimension = "1d"
            mode = "transient"
        else:
            field_path = field_dir / f"{sample_id}.npz"
            _write_point_field(
                field_path,
                temperature_k=temperature_k,
                reference=reference,
            )
            source = "experiment_case"
            dimension = "2d"
            mode = "steady"

        meta = dict(item.meta)
        meta.update(
            {
                "temperature_c": float(item.temperature_c),
                "temperature_k": temperature_k,
                "truncated_post0_length": int(cut_length),
                "target_waveform_length": target_length,
                "resample_method": "linear_interp_normalized_axis",
                "inference_only": inference_only,
                "has_spatial_ground_truth": False,
                "field_label_kind": (
                    "uniform_temperature_proxy_from_file_name"
                    if inference_only
                    else "legacy_zero_placeholder"
                ),
                "reference_manifest": (
                    str(reference.manifest_path) if reference is not None else None
                ),
            }
        )
        records.append(
            {
                "sample_id": sample_id,
                "source": source,
                "material_key": material_key,
                "dimension": dimension,
                "mode": mode,
                "temperature_k": temperature_k,
                "waveform_path": f"waveforms/{sample_id}.npy",
                "field_path": f"fields/{field_path.name}",
                "acoustic": {
                    "tof": 0.0,
                    "amplitude": max(abs(item) for item in waveform),
                    "center_freq": 0.0,
                },
                "meta": meta,
            }
        )

    split_result: dict[str, Any] | None = None
    experiment_manifest: Path | None = None
    combined_manifest: Path | None = None
    if inference_only:
        assert reference is not None
        test_manifest = output_dir / str(args.test_manifest_name)
        _write_inference_manifest(
            test_manifest,
            dataset_name=dataset_name,
            records=records,
            reference=reference,
            material_key=material_key,
        )
        split_result = {
            "train_manifest": None,
            "validation_manifest": None,
            "test_manifest": str(test_manifest),
            "split_stats": {
                "total_records": len(records),
                "train_records": 0,
                "validation_records": 0,
                "test_records": len(records),
                "split_strategy": "all_records_inference_only",
            },
        }
    else:
        experiment_manifest = output_dir / "experiment_manifest.json"
        combined_manifest = output_dir / "combined_manifest.json"
        _write_manifest(experiment_manifest, dataset_name, records)
        _write_manifest(combined_manifest, dataset_name, records)

    if not inference_only and bool(args.split_dataset):
        train_records, test_records, split_stats = _split_records_by_temperature(
            records,
            test_ratio=float(args.test_ratio),
            seed=int(args.seed),
        )
        train_manifest = output_dir / str(args.train_manifest_name)
        test_manifest = output_dir / str(args.test_manifest_name)
        _write_manifest(train_manifest, f"{dataset_name}_train_split", train_records)
        _write_manifest(test_manifest, f"{dataset_name}_test_split", test_records)
        split_result = {
            "train_manifest": str(train_manifest),
            "test_manifest": str(test_manifest),
            "split_stats": split_stats,
        }

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "dataset_name": dataset_name,
        "material_key": material_key,
        "records": len(records),
        "temperature_c_range": [
            float(min(item.temperature_c for item in loaded)),
            float(max(item.temperature_c for item in loaded)),
        ],
        "post0_length_unique": sorted(set(int(length) for length in post0_lengths)),
        "truncated_post0_length": int(cut_length),
        "target_waveform_length": target_length,
        "signal_column": str(args.signal_column),
        "time_filter": f"{args.time_column} >= {float(args.time_min_s):g}",
        "inference_only": inference_only,
        "has_spatial_ground_truth": False if inference_only else None,
        "reference_manifest": (
            str(reference.manifest_path) if reference is not None else None
        ),
        "sampling_index": (
            str(reference.sampling_path) if reference is not None else None
        ),
        "manifests": {
            "experiment": str(experiment_manifest) if experiment_manifest else None,
            "combined": str(combined_manifest) if combined_manifest else None,
            "train": split_result["train_manifest"] if split_result else None,
            "validation": split_result.get("validation_manifest") if split_result else None,
            "test": split_result["test_manifest"] if split_result else None,
        },
        "split": split_result,
    }
    (output_dir / "split_config.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def build_inference_database(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    reference_manifest: str | Path,
    dataset_name: str = "wumu_exp_inference",
    material_key: str = "wumu",
    signal_column: str = DEFAULT_SIGNAL_COLUMN,
    time_column: str = "time_s",
    time_min_s: float = 0.0,
    post0_length: int | None = None,
    limit: int | None = None,
    test_manifest_name: str = "test_manifest.json",
) -> dict[str, Any]:
    """Build an external test-only point-field manifest for GUI/CLI integration."""
    return build_database(
        argparse.Namespace(
            source_dir=Path(source_dir),
            output_dir=Path(output_dir),
            dataset_name=str(dataset_name),
            material_key=str(material_key),
            signal_column=str(signal_column),
            time_column=str(time_column),
            time_min_s=float(time_min_s),
            post0_length=post0_length,
            target_length=512,
            field_grid_1d=64,
            limit=limit,
            inference_only=True,
            reference_manifest=Path(reference_manifest),
            split_dataset=False,
            test_ratio=0.2,
            seed=42,
            train_manifest_name="train_manifest.json",
            test_manifest_name=str(test_manifest_name),
        )
    )


def parse_args() -> argparse.Namespace:
    """Parse the standalone post-zero database builder options."""
    parser = argparse.ArgumentParser(
        description="Build a standard ai_model database from post-zero waveform CSV files.",
    )
    parser.add_argument("--source-dir", type=Path, default=_script_source_dir())
    parser.add_argument("--output-dir", type=Path, default=_default_output_dir())
    parser.add_argument("--dataset-name", type=str, default="")
    parser.add_argument("--material-key", type=str, default=DEFAULT_MATERIAL_KEY)
    parser.add_argument("--signal-column", type=str, default=DEFAULT_SIGNAL_COLUMN)
    parser.add_argument("--time-column", type=str, default="time_s")
    parser.add_argument("--time-min-s", type=float, default=0.0)
    parser.add_argument("--post0-length", type=int, default=None, help="Default: use the minimum post-zero length.")
    parser.add_argument("--target-length", type=int, default=512)
    parser.add_argument("--field-grid-1d", type=int, default=64)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--inference-only",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Build a point-field-compatible external test manifest only. "
            "No train or combined manifest is written."
        ),
    )
    parser.add_argument(
        "--reference-manifest",
        type=Path,
        default=None,
        help=(
            "Training manifest used by the prediction checkpoint. Its sampling index, "
            "normalization, schema, and waveform length are reused in inference-only mode."
        ),
    )
    parser.add_argument("--split-dataset", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-manifest-name", type=str, default="train_manifest.json")
    parser.add_argument("--test-manifest-name", type=str, default="test_manifest.json")
    return parser.parse_args()


def main() -> None:
    """Run the standalone builder and print its machine-readable summary."""
    summary = build_database(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
