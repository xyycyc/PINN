from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import AIModelConfig, MATERIAL_LIBRARY, SimulationCase, get_material, simulate_case
from .material_registry import ensure_material_csv, register_material


def _extract_temperature_from_name(file_name: str) -> float:
    match = re.search(r"[-+]?\d*\.?\d+", file_name)
    if not match:
        raise ValueError(f"无法从文件名中提取温度: {file_name}")
    return float(match.group())


def _dlm_csv_skiprows(csv_path: Path) -> int:
    """DLM 示波器导出 CSV 首行为 Header Size，数据从第 17 行起（skiprows=15）。"""
    with csv_path.open("r", encoding="utf-8", errors="replace") as handle:
        first_line = handle.readline()
    if re.search(r"header\s*size", first_line, flags=re.IGNORECASE):
        return 15
    return 0


def _resample_signal(values: np.ndarray, length: int, normalize: bool = True) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return np.zeros(length, dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, values.size, dtype=np.float32)
    x_new = np.linspace(0.0, 1.0, length, dtype=np.float32)
    signal = np.interp(x_new, x_old, values).astype(np.float32)
    if normalize:
        std = float(signal.std())
        if std > 1e-6:
            signal = (signal - signal.mean()) / std
    return signal.astype(np.float32)


@dataclass
class DataRecord:
    sample_id: str
    source: str
    material_key: str
    dimension: str
    mode: str
    temperature_k: float
    waveform_path: str
    field_path: str
    acoustic: dict[str, float]
    meta: dict[str, Any]


class DatabaseBuilder:
    def __init__(self, config: AIModelConfig | None = None):
        self.config = config or AIModelConfig()
        self.config.ensure_dirs()
        self.database_dir = self.config.database_dir
        self.material_registry_path = ensure_material_csv(self.config.data_root)

    def _register_material(self, material_key: str, source: str) -> None:
        register_material(
            self.material_registry_path,
            material_key,
            source=source,
        )

    def _write_record_arrays(
        self,
        sample_id: str,
        waveform: np.ndarray,
        field: np.ndarray,
    ) -> tuple[str, str]:
        wave_dir = self.database_dir / "waveforms"
        field_dir = self.database_dir / "fields"
        wave_dir.mkdir(parents=True, exist_ok=True)
        field_dir.mkdir(parents=True, exist_ok=True)
        wave_path = wave_dir / f"{sample_id}.npy"
        field_path = field_dir / f"{sample_id}.npy"
        np.save(wave_path, waveform.astype(np.float32))
        np.save(field_path, field.astype(np.float32))
        return str(wave_path.relative_to(self.database_dir)), str(field_path.relative_to(self.database_dir))

    def build_simulation_database(
        self,
        samples_per_material: int = 400,
        output_name: str = "simulation_manifest.json",
    ) -> Path:
        rng = np.random.default_rng(self.config.random_seed)
        records: list[dict[str, Any]] = []
        modes = ("steady", "transient")
        dimensions = ("1d", "2d")

        for material_key in MATERIAL_LIBRARY:
            self._register_material(material_key, source="builtin_simulation")
            material = get_material(material_key)
            for idx in range(samples_per_material):
                dimension = dimensions[idx % len(dimensions)]
                mode = modes[(idx // len(dimensions)) % len(modes)]
                if idx == 0:
                    init_temp = float(self.config.min_temperature_k)
                    target_temp = 900.0
                elif idx == 1:
                    init_temp = 520.0
                    target_temp = float(self.config.max_temperature_k)
                else:
                    init_temp = float(rng.uniform(self.config.min_temperature_k, 600.0))
                    target_temp = float(rng.uniform(max(init_temp + 80.0, 380.0), self.config.max_temperature_k))
                case = SimulationCase(
                    material_key=material_key,
                    dimension=dimension,
                    mode=mode,
                    thickness_m=float(rng.uniform(*material.thickness_range_m)),
                    n_time=48,
                    init_temp_k=init_temp,
                    target_temp_k=target_temp,
                    heat_flux_scale=float(rng.uniform(1.5e5, 7.0e5)),
                )
                result = simulate_case(
                    material=material,
                    case=case,
                    waveform_length=self.config.waveform_length,
                    field_grid_1d=self.config.field_grid_1d,
                    field_grid_2d=self.config.field_grid_2d,
                    rng=rng,
                )
                sample_id = f"sim_{material_key}_{idx:05d}"
                wave_rel, field_rel = self._write_record_arrays(
                    sample_id=sample_id,
                    waveform=result["waveform"],
                    field=result["field"],
                )
                records.append(
                    asdict(
                        DataRecord(
                            sample_id=sample_id,
                            source="simulation",
                            material_key=material_key,
                            dimension=dimension,
                            mode=mode,
                            temperature_k=float(result["max_temperature_k"]),
                            waveform_path=wave_rel,
                            field_path=field_rel,
                            acoustic={
                                "tof": float(result["tof"]),
                                "amplitude": float(result["amplitude"]),
                                "center_freq": float(result["center_freq"]),
                            },
                            meta={
                                "mean_temperature_k": float(result["mean_temperature_k"]),
                                "thickness_m": float(case.thickness_m),
                                "init_temperature_k": float(case.init_temp_k),
                                "target_temperature_k": float(case.target_temp_k),
                            },
                        )
                    )
                )

        manifest_path = self.database_dir / output_name
        manifest_path.write_text(
            json.dumps(
                {
                    "dataset_name": "ai_model_simulation_db",
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return manifest_path

    def _import_csv_dataset(
        self,
        source_dir: str | Path,
        material_key: str,
        *,
        output_name: str,
        dataset_name: str,
        source_label: str,
        sample_prefix: str,
        limit: int | None = None,
    ) -> Path:
        source_dir = Path(source_dir)
        self._register_material(material_key, source=source_label)
        records: list[dict[str, Any]] = []
        csv_files = sorted(source_dir.rglob("*.csv"))

        if limit is not None and limit <= 0:
            manifest_path = self.database_dir / output_name
            manifest_path.write_text(
                json.dumps({"dataset_name": dataset_name, "records": []}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            return manifest_path

        for idx, csv_path in enumerate(csv_files):
            if limit is not None and len(records) >= limit:
                break
            if csv_path.name.upper().startswith("BASIC") or csv_path.name.upper().startswith("DOWN"):
                continue
            try:
                temperature_k = _extract_temperature_from_name(csv_path.name)
            except ValueError:
                continue

            try:
                frame = pd.read_csv(csv_path, header=None, skiprows=_dlm_csv_skiprows(csv_path))
                signal = frame.iloc[:, 1].to_numpy(dtype=np.float32)
            except Exception:
                continue

            # 建库仅保存重采样后的原始实验波形；clip/smooth 等在训练/推理阶段由 Dataset + 模型入口处理。
            waveform = _resample_signal(signal, self.config.waveform_length, normalize=False)
            field = np.zeros((1, self.config.field_grid_1d), dtype=np.float32)
            sample_id = f"{sample_prefix}_{material_key}_{idx:05d}"
            wave_rel, field_rel = self._write_record_arrays(sample_id, waveform, field)
            records.append(
                asdict(
                    DataRecord(
                        sample_id=sample_id,
                        source=source_label,
                        material_key=material_key,
                        dimension="1d",
                        mode="transient",
                        temperature_k=float(temperature_k),
                        waveform_path=wave_rel,
                        field_path=field_rel,
                        acoustic={"tof": 0.0, "amplitude": float(np.max(np.abs(waveform))), "center_freq": 0.0},
                        meta={"original_csv": str(csv_path)},
                    )
                )
            )

        manifest_path = self.database_dir / output_name
        manifest_path.write_text(
            json.dumps(
                {
                    "dataset_name": dataset_name,
                    "records": records,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return manifest_path

    def import_experimental_csvs(
        self,
        source_dir: str | Path,
        material_key: str,
        output_name: str = "experiment_manifest.json",
        limit: int | None = None,
    ) -> Path:
        return self._import_csv_dataset(
            source_dir=source_dir,
            material_key=material_key,
            output_name=output_name,
            dataset_name="ai_model_experiment_db",
            source_label="experiment",
            sample_prefix="exp",
            limit=limit,
        )

    def import_external_simulation_csvs(
        self,
        source_dir: str | Path,
        material_key: str,
        output_name: str = "external_simulation_manifest.json",
        limit: int | None = None,
    ) -> Path:
        return self._import_csv_dataset(
            source_dir=source_dir,
            material_key=material_key,
            output_name=output_name,
            dataset_name="ai_model_external_simulation_csv_db",
            source_label="external_simulation",
            sample_prefix="extsim",
            limit=limit,
        )

    def merge_manifests(
        self,
        manifest_paths: list[str | Path],
        output_name: str = "combined_manifest.json",
    ) -> Path:
        records: list[dict[str, Any]] = []
        for manifest_path in manifest_paths:
            manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
            records.extend(manifest.get("records", []))

        output_path = self.database_dir / output_name
        output_path.write_text(
            json.dumps({"dataset_name": "ai_model_combined_db", "records": records}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return output_path

    def validate_requirement_33(self, manifest_path: str | Path) -> dict[str, Any]:
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        records = manifest.get("records", [])
        sim_records = [item for item in records if item.get("source") in {"simulation", "external_simulation"}]
        exp_records = [item for item in records if item.get("source") in {"experiment", "experiment_case"}]
        materials = sorted({item["material_key"] for item in records})
        temperatures: list[float] = []
        for item in records:
            temperatures.append(float(item["temperature_k"]))
            meta = item.get("meta", {})
            for key in ("init_temperature_k", "target_temperature_k"):
                if key in meta:
                    temperatures.append(float(meta[key]))

        report = {
            "total_records": len(records),
            "simulation_records": len(sim_records),
            "experiment_records": len(exp_records),
            "materials": materials,
            "temperature_range_k": [min(temperatures) if temperatures else None, max(temperatures) if temperatures else None],
            "meets_3_3_min_simulation": len(sim_records) >= self.config.min_simulation_samples,
            "meets_3_3_min_experiment": len(exp_records) >= self.config.min_experiment_samples,
            "meets_3_3_material_count": len(materials) >= 3,
            "meets_3_3_temperature_span": bool(
                temperatures
                and min(temperatures) <= self.config.min_temperature_k
                and max(temperatures) >= self.config.max_temperature_k
            ),
        }
        report["meets_3_3"] = all(
            [
                report["meets_3_3_min_simulation"],
                report["meets_3_3_min_experiment"],
                report["meets_3_3_material_count"],
                report["meets_3_3_temperature_span"],
            ]
        )

        report_dir = self.config.train_report_root / "validation"
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "requirement_3_3_report.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report
