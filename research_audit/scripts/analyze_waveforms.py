"""Stream waveform files into compact, non-sensitive diagnostic features."""

from __future__ import annotations

import argparse
import csv
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from audit_utils import (
    DEFAULT_RAW_ROOT,
    REPOSITORY_RAW_ROOT,
    STATS_ROOT,
    case_temperature_k,
    ensure_output_dirs,
    json_number,
    write_csv,
)


FEATURE_NAMES = [
    "mean", "std", "rms", "peak_abs", "peak_to_peak", "energy_mean_square",
    "crest_factor", "skewness", "kurtosis", "zero_crossing_rate",
    "peak_time_s", "peak_time_fraction", "dominant_frequency_hz",
    "dominant_frequency_fraction", "spectral_centroid_hz",
    "spectral_centroid_fraction", "spectral_bandwidth_hz",
    "spectral_bandwidth_fraction",
]


def load_named_csv(path: Path, signal_preference: list[str]) -> tuple[np.ndarray, np.ndarray, str]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        header: list[str] | None = None
        while True:
            line = handle.readline()
            if not line:
                raise ValueError(f"No CSV header in {path}")
            if line.startswith("#") or not line.strip():
                continue
            header = next(csv.reader([line]))
            break
        normalized = [item.strip() for item in header]
        time_index = normalized.index("time_s") if "time_s" in normalized else normalized.index("time")
        signal_name = next((name for name in signal_preference if name in normalized), "")
        if not signal_name:
            axes = {"step", "time", "time_s", "index", "sample", "sample_index"}
            signal_name = next(name for name in normalized if name.casefold() not in axes)
        signal_index = normalized.index(signal_name)
        data = np.loadtxt(handle, delimiter=",", usecols=(time_index, signal_index), dtype=np.float64)
    data = np.atleast_2d(data)
    return data[:, 0], data[:, 1], signal_name


def load_dlm_csv(path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        for _ in range(15):
            row = next(csv.reader([handle.readline()]))
            if len(row) >= 2:
                metadata[row[0].strip().strip('"')] = row[1].strip().strip('" ')
        data = np.loadtxt(handle, delimiter=",", usecols=(0, 1), dtype=np.float64)
    data = np.atleast_2d(data)
    return data[:, 0], data[:, 1], metadata


def features(time_s: np.ndarray, signal: np.ndarray) -> dict[str, float | int | None]:
    time_s = np.asarray(time_s, dtype=float)
    signal = np.asarray(signal, dtype=float)
    finite = np.isfinite(time_s) & np.isfinite(signal)
    time_s, signal = time_s[finite], signal[finite]
    if len(signal) < 4:
        raise ValueError("Waveform has fewer than four finite samples")
    dt = float(np.median(np.diff(time_s)))
    duration = float(time_s[-1] - time_s[0])
    mean = float(np.mean(signal))
    centered = signal - mean
    std = float(np.std(signal))
    rms = float(np.sqrt(np.mean(signal * signal)))
    peak_abs = float(np.max(np.abs(signal)))
    peak_index = int(np.argmax(np.abs(signal)))
    scale = std if std > 1e-30 else 1.0
    normalized = centered / scale
    stride = max(1, int(math.ceil(len(signal) / 16384)))
    fft_signal = normalized[::stride]
    fft_dt = dt * stride
    spectrum = np.abs(np.fft.rfft(fft_signal)) ** 2
    frequencies = np.fft.rfftfreq(len(fft_signal), d=fft_dt)
    if len(spectrum):
        spectrum[0] = 0.0
    spectral_sum = float(np.sum(spectrum))
    dominant = float(frequencies[int(np.argmax(spectrum))]) if spectral_sum > 0 else 0.0
    centroid = float(np.sum(frequencies * spectrum) / spectral_sum) if spectral_sum > 0 else 0.0
    bandwidth = float(np.sqrt(np.sum(((frequencies - centroid) ** 2) * spectrum) / spectral_sum)) if spectral_sum > 0 else 0.0
    nyquist = 0.5 / dt if dt > 0 else 0.0
    return {
        "sample_count": int(len(signal)),
        "dt_s": json_number(dt),
        "sampling_rate_hz": json_number(1.0 / dt if dt > 0 else None),
        "time_start_s": json_number(float(time_s[0])),
        "time_end_s": json_number(float(time_s[-1])),
        "duration_s": json_number(duration),
        "mean": mean,
        "std": std,
        "rms": rms,
        "peak_abs": peak_abs,
        "peak_to_peak": float(np.ptp(signal)),
        "energy_mean_square": float(np.mean(signal * signal)),
        "crest_factor": peak_abs / max(rms, 1e-30),
        "skewness": float(np.mean(normalized ** 3)),
        "kurtosis": float(np.mean(normalized ** 4)),
        "zero_crossing_rate": float(np.mean(centered[:-1] * centered[1:] < 0)),
        "peak_time_s": float(time_s[peak_index]),
        "peak_time_fraction": float((time_s[peak_index] - time_s[0]) / max(duration, 1e-30)),
        "dominant_frequency_hz": dominant,
        "dominant_frequency_fraction": dominant / max(nyquist, 1e-30),
        "spectral_centroid_hz": centroid,
        "spectral_centroid_fraction": centroid / max(nyquist, 1e-30),
        "spectral_bandwidth_hz": bandwidth,
        "spectral_bandwidth_fraction": bandwidth / max(nyquist, 1e-30),
    }


def simulation_records(raw_root: Path) -> Iterator[dict[str, Any]]:
    preferences = {
        "metal": ["receiver_velocity_normal", "receiver_displacement_normal"],
        "silicon": ["top_pulse_echo_velocity", "bottom_transmission_velocity"],
        "wumu": ["receiver_velocity_normal", "receiver_displacement_normal"],
    }
    for dataset in ("metal", "silicon", "wumu"):
        for case in sorted((raw_root / dataset).glob("case_*_T*K")):
            path = case / "ultrasonic" / "receiver_signal.csv"
            if not path.is_file():
                continue
            time_s, signal, channel = load_named_csv(path, preferences[dataset])
            yield {
                "sample_id": f"{dataset}/{case.name}",
                "domain": "simulation",
                "dataset": dataset,
                "material": dataset,
                "physical_group": f"{dataset}/{case.name}",
                "temperature_value": case_temperature_k(case.name),
                "temperature_unit": "K",
                "temperature_semantics": "solver case/field temperature tag",
                "channel": channel,
                "amplitude_unit": "solver_velocity_or_displacement",
                **features(time_s, signal),
            }


def post0_records() -> Iterator[dict[str, Any]]:
    root = REPOSITORY_RAW_ROOT / "waveforms_by_file"
    pattern = re.compile(r"^T([-+]?\d+(?:\.\d+)?)C_waveform_(\d{8})_(\d{6})", re.I)
    for path in sorted(root.glob("*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        time_s, signal, channel = load_named_csv(path, ["amplitude_filtered_residual", "amplitude_raw_residual"])
        temperature_c = float(match.group(1))
        yield {
            "sample_id": f"wumu_post0/{path.stem}",
            "domain": "experiment",
            "dataset": "wumu_post0_waveforms",
            "material": "wumu",
            "physical_group": f"wumu_post0/temperature_{temperature_c:g}C",
            "acquisition_batch": match.group(2),
            "temperature_value": temperature_c,
            "temperature_unit": "degC",
            "temperature_semantics": "filename-encoded nominal value; physical meaning unconfirmed",
            "channel": channel,
            "amplitude_unit": "processed_residual_scale",
            **features(time_s, signal),
        }


def dlm_records() -> Iterator[dict[str, Any]]:
    root = REPOSITORY_RAW_ROOT / "10times"
    pattern = re.compile(r"^([-+]?\d+(?:\.\d+)?)\.csv$", re.I)
    for repetition in sorted(item for item in root.iterdir() if item.is_dir() and item.name.isdigit()):
        for path in sorted(repetition.glob("*.csv")):
            match = pattern.match(path.name)
            if not match:
                continue
            time_s, signal, meta = load_dlm_csv(path)
            temperature = float(match.group(1))
            yield {
                "sample_id": f"metal_10times/{repetition.name}/{path.stem}",
                "domain": "experiment",
                "dataset": "metal_10times_dlm",
                "material": "metal_matrix",
                "physical_group": f"metal_10times/temperature_{temperature:g}",
                "acquisition_batch": repetition.name,
                "acquisition_date": meta.get("Date", ""),
                "temperature_value": temperature,
                "temperature_unit": "unknown",
                "temperature_semantics": "numeric filename; physical meaning and unit unconfirmed",
                "channel": meta.get("TraceName", "CH1"),
                "amplitude_unit": meta.get("VUnit", "V"),
                **features(time_s, signal),
            }


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[str(row["dataset"])].append(row)
    for dataset, records in sorted(by_dataset.items()):
        for feature in FEATURE_NAMES:
            values = np.asarray([float(row[feature]) for row in records if row.get(feature) is not None], dtype=float)
            result.append({
                "dataset": dataset,
                "domain": records[0]["domain"],
                "feature": feature,
                "sample_count": len(values),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "q10": float(np.quantile(values, 0.10)),
                "median": float(np.median(values)),
                "q90": float(np.quantile(values, 0.90)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-data-root", type=Path, default=DEFAULT_RAW_ROOT)
    parser.add_argument("--skip-dlm", action="store_true")
    args = parser.parse_args()
    ensure_output_dirs()
    rows = list(simulation_records(args.raw_data_root.resolve()))
    rows.extend(post0_records())
    if not args.skip_dlm:
        rows.extend(dlm_records())
    write_csv(STATS_ROOT / "waveform_statistics.csv", rows)
    write_csv(STATS_ROOT / "waveform_feature_summary.csv", summarize(rows))
    print(f"waveform records: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
