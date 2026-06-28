from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .materials import MaterialSpec


def _load_inversion_module():
    # `model/optimize/inversion.py`；本文件在 `ai_model/config/`
    ai_model_root = Path(__file__).resolve().parents[1]
    inversion_path = ai_model_root / "model" / "optimize" / "inversion.py"
    spec = importlib.util.spec_from_file_location("ai_model_inversion", inversion_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载现有前向求解器: {inversion_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_INVERSION = _load_inversion_module()


@dataclass
class SimulationCase:
    material_key: str
    dimension: str
    mode: str
    thickness_m: float
    n_time: int
    init_temp_k: float
    target_temp_k: float
    heat_flux_scale: float


def _constant_heat_flux(case: SimulationCase, rng: np.random.Generator) -> np.ndarray:
    base = np.linspace(0.7, 1.0, case.n_time, dtype=np.float32)
    noise = rng.normal(0.0, 0.03, size=case.n_time).astype(np.float32)
    return case.heat_flux_scale * np.clip(base + noise, 0.6, 1.15)


def _pulse_heat_flux(case: SimulationCase, rng: np.random.Generator) -> np.ndarray:
    pulse_center = rng.uniform(0.25, 0.75)
    pulse_width = rng.uniform(0.08, 0.2)
    time_axis = np.linspace(0.0, 1.0, case.n_time, dtype=np.float32)
    pulse = np.exp(-((time_axis - pulse_center) ** 2) / max(pulse_width ** 2, 1e-6))
    return case.heat_flux_scale * (0.3 + pulse).astype(np.float32)


def build_heat_flux(case: SimulationCase, rng: np.random.Generator) -> np.ndarray:
    if case.mode == "steady":
        return np.full(case.n_time, case.heat_flux_scale, dtype=np.float32)
    if case.mode == "transient":
        return _pulse_heat_flux(case, rng)
    return _constant_heat_flux(case, rng)


def _normalize_field(field: np.ndarray, min_temp: float, max_temp: float) -> np.ndarray:
    denom = max(max_temp - min_temp, 1e-6)
    return ((field - min_temp) / denom).astype(np.float32)


def _make_waveform_from_tof(tof: np.ndarray, amplitude: np.ndarray, length: int) -> np.ndarray:
    t_axis = np.linspace(0.0, 1.0, length, dtype=np.float32)
    waveform = np.zeros(length, dtype=np.float32)
    centers = np.clip((tof - tof.min()) / max(tof.ptp(), 1e-6), 0.05, 0.95)
    width = 0.02
    carrier_freq = 14.0
    for center, amp in zip(centers, amplitude):
        envelope = np.exp(-((t_axis - center) ** 2) / width)
        carrier = np.sin(2.0 * np.pi * carrier_freq * (t_axis - center))
        waveform += amp * envelope * carrier
    max_abs = np.max(np.abs(waveform))
    if max_abs > 0:
        waveform = waveform / max_abs
    return waveform.astype(np.float32)


def simulate_case(
    material: MaterialSpec,
    case: SimulationCase,
    waveform_length: int,
    field_grid_1d: int,
    field_grid_2d: tuple[int, int],
    rng: np.random.Generator,
) -> dict[str, np.ndarray | float | str]:
    nx = field_grid_1d
    nt = max(case.n_time, 8)
    dt = 0.25 if case.mode == "steady" else 0.1
    q_flux = build_heat_flux(case, rng)

    temperature_history = _INVERSION.solve_heat_1d_core(
        q_flux=q_flux,
        L=case.thickness_m,
        Nx=nx,
        Nt=nt,
        dt=dt,
        rho=material.density,
        cp=material.heat_capacity,
        k=material.thermal_conductivity,
        T_init=case.init_temp_k,
    )
    dx = case.thickness_m / (nx - 1)
    tof = np.zeros(nt, dtype=np.float32)
    mean_temperature = temperature_history.mean(axis=1).astype(np.float32)
    amplitude = np.clip(
        1.2 - 0.0005 * (mean_temperature - case.init_temp_k),
        0.15,
        1.0,
    ).astype(np.float32)
    for i in range(nt):
        velocity = material.sound_speed_slope * temperature_history[i, :] + material.sound_speed_bias
        tof[i] = float(2.0 * np.sum(dx / np.clip(velocity, 1.0, None)))

    waveform = _make_waveform_from_tof(tof, amplitude, waveform_length)
    final_field_1d = temperature_history[-1]

    if case.dimension == "1d":
        field = _normalize_field(final_field_1d, case.init_temp_k, case.target_temp_k)
    else:
        ny, nx2 = field_grid_2d
        y_axis = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
        spread = 0.55 if case.mode == "steady" else 0.35
        lateral = np.exp(-(y_axis[:, None] ** 2) / spread)
        field_2d = lateral * final_field_1d[None, :]
        field = _normalize_field(field_2d, case.init_temp_k, case.target_temp_k)
        nx = nx2
        if field.shape[1] != nx2:
            x_old = np.linspace(0.0, 1.0, field.shape[1], dtype=np.float32)
            x_new = np.linspace(0.0, 1.0, nx2, dtype=np.float32)
            field = np.stack(
                [np.interp(x_new, x_old, row).astype(np.float32) for row in field],
                axis=0,
            )

    return {
        "waveform": waveform,
        "field": field.astype(np.float32),
        "tof": float(np.mean(tof)),
        "amplitude": float(np.mean(amplitude)),
        "center_freq": 14.0,
        "max_temperature_k": float(np.max(temperature_history)),
        "mean_temperature_k": float(np.mean(final_field_1d)),
        "dimension": case.dimension,
        "mode": case.mode,
        "material_key": case.material_key,
    }
