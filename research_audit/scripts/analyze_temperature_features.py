"""Quantify temperature-feature trends and condition confounding."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from audit_utils import (
    STATS_ROOT,
    ensure_output_dirs,
    json_number,
    linear_r2,
    mutual_information_binned,
    pearson,
    spearman,
    write_csv,
    write_json,
)


FEATURES = [
    "peak_time_s", "peak_time_fraction", "peak_abs", "rms", "energy_mean_square",
    "dominant_frequency_hz", "dominant_frequency_fraction", "spectral_centroid_hz",
    "spectral_centroid_fraction", "spectral_bandwidth_hz", "spectral_bandwidth_fraction",
]


def load_rows() -> list[dict[str, str]]:
    with (STATS_ROOT / "waveform_statistics.csv").open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    ensure_output_dirs()
    rows = load_rows()
    by_dataset: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    correlations: list[dict[str, Any]] = []
    for dataset, records in sorted(by_dataset.items()):
        x = np.asarray([float(row["temperature_value"]) for row in records], dtype=float)
        for feature in FEATURES:
            y = np.asarray([float(row[feature]) for row in records], dtype=float)
            p = pearson(x, y)
            s = spearman(x, y)
            correlations.append({
                "dataset": dataset,
                "domain": records[0]["domain"],
                "temperature_unit": records[0]["temperature_unit"],
                "temperature_semantics": records[0]["temperature_semantics"],
                "feature": feature,
                "sample_count": len(records),
                "pearson_r": json_number(p),
                "spearman_rho": json_number(s),
                "mutual_information_binned": json_number(mutual_information_binned(x, y)),
                "linear_r2": json_number(linear_r2(x, y)),
                "trend_direction": "increasing" if s is not None and s > 0.1 else "decreasing" if s is not None and s < -0.1 else "weak_or_flat",
            })
    write_csv(STATS_ROOT / "temperature_feature_correlations.csv", correlations)

    support_rows = [
        {"dataset": "metal", "domain": "simulation", "unit": "K", "min": 300.0, "max": 1500.0, "semantics": "solver temperature field/case tag", "comparable_support_k_min": 300.0, "comparable_support_k_max": 1500.0},
        {"dataset": "silicon", "domain": "simulation", "unit": "K", "min": 300.0, "max": 1500.0, "semantics": "solver temperature field/case tag", "comparable_support_k_min": 300.0, "comparable_support_k_max": 1500.0},
        {"dataset": "wumu", "domain": "simulation", "unit": "K", "min": 300.0, "max": 1500.0, "semantics": "solver temperature field/case tag", "comparable_support_k_min": 300.0, "comparable_support_k_max": 1500.0},
        {"dataset": "wumu_post0_waveforms", "domain": "experiment", "unit": "degC", "min": min(float(r["temperature_value"]) for r in by_dataset["wumu_post0_waveforms"]), "max": max(float(r["temperature_value"]) for r in by_dataset["wumu_post0_waveforms"]), "semantics": "filename-encoded nominal value; physical measurement semantics unknown", "comparable_support_k_min": min(float(r["temperature_value"]) for r in by_dataset["wumu_post0_waveforms"]) + 273.15, "comparable_support_k_max": max(float(r["temperature_value"]) for r in by_dataset["wumu_post0_waveforms"]) + 273.15},
        {"dataset": "metal_10times_dlm", "domain": "experiment", "unit": "unknown", "min": min(float(r["temperature_value"]) for r in by_dataset["metal_10times_dlm"]), "max": max(float(r["temperature_value"]) for r in by_dataset["metal_10times_dlm"]), "semantics": "numeric filename; physical meaning and unit unknown", "comparable_support_k_min": None, "comparable_support_k_max": None},
    ]
    write_csv(STATS_ROOT / "temperature_support.csv", support_rows)

    confounding: dict[str, Any] = {}
    for dataset in ("wumu_post0_waveforms", "metal_10times_dlm"):
        records = by_dataset[dataset]
        batch_to_temps: dict[str, list[float]] = defaultdict(list)
        temp_to_batches: dict[str, set[str]] = defaultdict(set)
        for row in records:
            batch = row.get("acquisition_batch") or row.get("acquisition_date") or "unknown"
            temp = float(row["temperature_value"])
            batch_to_temps[batch].append(temp)
            temp_to_batches[f"{temp:g}"].add(batch)
        confounding[dataset] = {
            "sample_count": len(records),
            "batch_count": len(batch_to_temps),
            "batch_temperature_ranges": {key: [min(vals), max(vals), len(vals)] for key, vals in sorted(batch_to_temps.items())},
            "temperature_group_count": len(temp_to_batches),
            "temperature_groups_seen_in_one_batch_only": sum(len(value) == 1 for value in temp_to_batches.values()),
            "duplicate_samples_by_temperature": dict(sorted(Counter(float(r["temperature_value"]) for r in records).items())),
            "risk": "HIGH" if any(len(value) == 1 for value in temp_to_batches.values()) else "MEDIUM",
        }
    write_json(STATS_ROOT / "condition_confounding.json", confounding)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
