"""Inventory raw simulation and local experimental assets without modifying them."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from audit_utils import (
    DEFAULT_RAW_ROOT,
    REPOSITORY_RAW_ROOT,
    STATS_ROOT,
    case_temperature_k,
    ensure_output_dirs,
    logical_path,
    write_csv,
    write_json,
)


def summarize_tree(path: Path) -> tuple[int, int, Counter[str]]:
    count = 0
    size = 0
    extensions: Counter[str] = Counter()
    for item in path.rglob("*"):
        if not item.is_file():
            continue
        count += 1
        size += item.stat().st_size
        extensions[item.suffix.lower() or "[none]"] += 1
    return count, size, extensions


def inspect_sim_dataset(root: Path, dataset_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cases = sorted(item for item in dataset_dir.glob("case_*_T*K") if item.is_dir())
    file_count, byte_count, extensions = summarize_tree(dataset_dir)
    temperatures = [case_temperature_k(item.name) for item in cases]
    temperatures = [value for value in temperatures if value is not None]
    paired = 0
    failures: list[dict[str, str]] = []
    waveform_count = field_count = config_count = 0
    for case in cases:
        wave = case / "ultrasonic" / "receiver_signal.csv"
        field = case / "heat" / "thermomechanical_steady" / "csv" / "thermomechanical_steady_nodes.csv"
        configs = sorted((case / "configs").glob("*_ultrasonic_config.json"))
        waveform_count += int(wave.is_file())
        field_count += int(field.is_file())
        config_count += len(configs)
        ok = wave.is_file() and field.is_file() and len(configs) == 1
        reference_ok = False
        if ok:
            try:
                payload = json.loads(configs[0].read_text(encoding="utf-8"))
                io = payload.get("config", payload).get("io", {})
                referenced = str(io.get("temperature_csv_path", "")).replace("\\", "/").casefold()
                reference_ok = case.name.casefold() in referenced and field.name.casefold() in referenced
            except (OSError, ValueError, json.JSONDecodeError):
                reference_ok = False
        if ok and reference_ok:
            paired += 1
        else:
            failures.append({"case": case.name, "reason": "missing artifact, ambiguous config, or reference mismatch"})
    row = {
        "dataset": dataset_dir.name,
        "logical_path": logical_path(dataset_dir, root),
        "source_candidate": "simulation",
        "actual_origin": "simulation",
        "material": dataset_dir.name,
        "physical_cases": len(cases),
        "file_count": file_count,
        "waveform_count": waveform_count,
        "field_count": field_count,
        "config_count": config_count,
        "approximate_bytes": byte_count,
        "file_formats": ";".join(f"{key}:{value}" for key, value in sorted(extensions.items())),
        "temperature_min_k": min(temperatures) if temperatures else None,
        "temperature_max_k": max(temperatures) if temperatures else None,
        "waveform_availability": waveform_count == len(cases) and bool(cases),
        "full_field_availability": field_count == len(cases) and bool(cases),
        "weak_label_availability": dataset_dir.name == "wumu",
        "metadata_availability": config_count >= len(cases) and bool(cases),
        "processing_status": "raw solver exports with postprocess products",
    }
    pairing = {
        "dataset": dataset_dir.name,
        "case_count": len(cases),
        "strict_waveform_field_pairs": paired,
        "failed_pairs": len(failures),
        "failure_examples": failures[:20],
    }
    return row, pairing


def inspect_experiment_dirs() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ten = REPOSITORY_RAW_ROOT / "10times"
    post = REPOSITORY_RAW_ROOT / "waveforms_by_file"
    for name, path, material, status in (
        ("metal_10times_dlm", ten, "metal_matrix", "raw oscilloscope exports"),
        ("wumu_post0_waveforms", post, "wumu", "derived filtered/residual waveform CSVs"),
    ):
        file_count, byte_count, extensions = summarize_tree(path)
        csvs = list(path.rglob("*.csv"))
        if name.startswith("metal"):
            measurements = [p for p in csvs if p.stem.replace(".", "", 1).isdigit() and "_test_out" not in p.parts]
            temperatures = [float(p.stem) for p in measurements]
        else:
            import re
            pattern = re.compile(r"^T([-+]?\d+(?:\.\d+)?)C(?:_|$)", re.I)
            measurements = [p for p in csvs if pattern.match(p.name)]
            temperatures = [float(pattern.match(p.name).group(1)) for p in measurements]
        rows.append({
            "dataset": name,
            "logical_path": logical_path(path),
            "source_candidate": "physical_experiment" if name.startswith("metal") else "processed_experiment",
            "actual_origin": "physical_experiment" if name.startswith("metal") else "processed_experiment",
            "material": material,
            "physical_cases": len(measurements),
            "file_count": file_count,
            "waveform_count": len(measurements),
            "field_count": 0,
            "config_count": 0,
            "approximate_bytes": byte_count,
            "file_formats": ";".join(f"{key}:{value}" for key, value in sorted(extensions.items())),
            "temperature_min_k": None,
            "temperature_max_k": None,
            "filename_temperature_min": min(temperatures) if temperatures else None,
            "filename_temperature_max": max(temperatures) if temperatures else None,
            "filename_temperature_unit": "unknown" if name.startswith("metal") else "degC",
            "waveform_availability": bool(measurements),
            "full_field_availability": False,
            "weak_label_availability": bool(measurements),
            "metadata_availability": name.startswith("metal"),
            "processing_status": status,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-root", type=Path, default=DEFAULT_RAW_ROOT)
    args = parser.parse_args()
    ensure_output_dirs()
    root = args.raw_data_root.resolve()
    sim_rows: list[dict[str, Any]] = []
    pairing_rows: list[dict[str, Any]] = []
    for dataset_dir in sorted(item for item in root.iterdir() if item.is_dir()):
        row, pairing = inspect_sim_dataset(root, dataset_dir)
        sim_rows.append(row)
        pairing_rows.append(pairing)
    inventory = sim_rows + inspect_experiment_dirs()
    write_csv(STATS_ROOT / "data_inventory.csv", inventory)
    total_sim_pairs = sum(item["strict_waveform_field_pairs"] for item in pairing_rows)
    total_sim_cases = sum(item["case_count"] for item in pairing_rows)
    write_json(STATS_ROOT / "pairing_statistics.json", {
        "simulation": pairing_rows,
        "simulation_case_count": total_sim_cases,
        "simulation_exact_waveform_field_pairs": total_sim_pairs,
        "simulation_pair_failures": total_sim_cases - total_sim_pairs,
        "sim_real_exact_pair_count": 0,
        "sim_real_approximate_pair_count": 0,
        "simulation_unpaired_count": total_sim_cases,
        "experiment_unpaired_count": sum(int(row["waveform_count"]) for row in inventory if row["actual_origin"] != "simulation"),
        "is_fundamentally_unpaired": True,
        "rationale": "No experiment record contains a simulation case/configuration identifier; shared temperatures alone were not treated as pairing.",
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
