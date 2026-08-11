"""Write evidence-based provenance classifications for all principal datasets."""

from __future__ import annotations

from audit_utils import STATS_ROOT, ensure_output_dirs, write_csv


def main() -> int:
    ensure_output_dirs()
    rows = [
        {"dataset": "metal", "code_label": "experiment_case", "actual_origin": "simulation", "confidence": "HIGH", "evidence": "Per-case heat/ultrasonic solver configs, FEM mesh reference, generated receiver_signal.csv, and solver logs under RAW_DATA_ROOT/metal."},
        {"dataset": "silicon", "code_label": "experiment_case", "actual_origin": "simulation", "confidence": "HIGH", "evidence": "Per-case heat/ultrasonic solver configs, generated thermal field CSV, receiver output, and solver logs under RAW_DATA_ROOT/silicon."},
        {"dataset": "wumu", "code_label": "experiment_case", "actual_origin": "simulation", "confidence": "HIGH", "evidence": "CalibrationBatch solver workspaces, mesh/thermal references, generated FEM fields and FDTD/FEM-style ultrasonic logs under RAW_DATA_ROOT/wumu."},
        {"dataset": "metal_10times_dlm", "code_label": "experiment", "actual_origin": "physical_experiment", "confidence": "HIGH", "evidence": "DLM3000 oscilloscope headers contain CH1, volts, 2.5 GHz sample rate, acquisition date/time, and ten repetition folders."},
        {"dataset": "wumu_post0_waveforms", "code_label": "experiment", "actual_origin": "processed_experiment", "confidence": "HIGH", "evidence": "Timestamped TxxC filenames plus residual/filtered columns; builder identifies source CSV and derives post-zero/resampled waveforms."},
        {"dataset": "wumu_exp_manifest_fields", "code_label": "experiment", "actual_origin": "processed_experiment", "confidence": "HIGH", "evidence": "Fields are code-generated uniform scalar proxies/zeros; no measured spatial coordinates or thermal-imaging source exists."},
        {"dataset": "legacy_metal_matrix_fields", "code_label": "experiment", "actual_origin": "processed_experiment", "confidence": "HIGH", "evidence": "DatabaseBuilder creates zero-valued field arrays for imported oscilloscope CSVs; they are not measured temperature fields."},
    ]
    write_csv(STATS_ROOT / "source_provenance_map.csv", rows, ["dataset", "code_label", "actual_origin", "confidence", "evidence"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
