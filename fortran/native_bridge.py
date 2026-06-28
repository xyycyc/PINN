from __future__ import annotations

import ctypes
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


def _library_name() -> str:
    if sys.platform == "win32":
        return "ai_numeric.dll"
    if sys.platform == "darwin":
        return "libai_numeric.dylib"
    return "libai_numeric.so"


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    library_path = Path(__file__).resolve().parent / _library_name()
    if not library_path.is_file():
        raise RuntimeError(
            f"Fortran library not found: {library_path}. "
            "On Windows, run: powershell -ExecutionPolicy Bypass "
            "-File fortran/build_fortran.ps1"
        )

    library = ctypes.CDLL(str(library_path))
    double_pointer = ctypes.POINTER(ctypes.c_double)

    library.compute_prediction_metrics.argtypes = [
        double_pointer,
        double_pointer,
        ctypes.c_int64,
        double_pointer,
        double_pointer,
        double_pointer,
    ]
    library.compute_prediction_metrics.restype = None

    library.average_waveform_pairs.argtypes = [
        double_pointer,
        double_pointer,
        ctypes.c_int64,
        ctypes.c_int64,
        double_pointer,
        double_pointer,
    ]
    library.average_waveform_pairs.restype = None
    return library


def _double_pointer(values: np.ndarray) -> ctypes.POINTER(ctypes.c_double):
    return values.ctypes.data_as(ctypes.POINTER(ctypes.c_double))


def compute_prediction_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float, float]:
    true_values = np.ascontiguousarray(y_true, dtype=np.float64).reshape(-1)
    pred_values = np.ascontiguousarray(y_pred, dtype=np.float64).reshape(-1)
    if true_values.shape != pred_values.shape:
        raise ValueError(
            f"Metric inputs must have the same shape: {true_values.shape} != {pred_values.shape}"
        )
    if true_values.size == 0:
        return float("nan"), float("nan"), float("nan")

    mae = ctypes.c_double()
    rmse = ctypes.c_double()
    max_error = ctypes.c_double()
    _load_library().compute_prediction_metrics(
        _double_pointer(true_values),
        _double_pointer(pred_values),
        ctypes.c_int64(true_values.size),
        ctypes.byref(mae),
        ctypes.byref(rmse),
        ctypes.byref(max_error),
    )
    return float(mae.value), float(rmse.value), float(max_error.value)


def average_waveform_pairs(
    time_values: np.ndarray,
    voltage_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    times = np.ascontiguousarray(time_values, dtype=np.float64)
    voltages = np.ascontiguousarray(voltage_values, dtype=np.float64)
    if times.ndim != 2 or voltages.ndim != 2:
        raise ValueError("Waveform inputs must be two-dimensional [run, point] arrays.")
    if times.shape != voltages.shape:
        raise ValueError(f"Waveform input shapes differ: {times.shape} != {voltages.shape}")
    n_runs, n_points = times.shape
    if n_runs == 0 or n_points == 0:
        raise ValueError("Waveform inputs must not be empty.")

    time_average = np.empty(n_points, dtype=np.float64)
    voltage_average = np.empty(n_points, dtype=np.float64)
    _load_library().average_waveform_pairs(
        _double_pointer(times),
        _double_pointer(voltages),
        ctypes.c_int64(n_runs),
        ctypes.c_int64(n_points),
        _double_pointer(time_average),
        _double_pointer(voltage_average),
    )
    return time_average, voltage_average


def backend_name() -> str:
    _load_library()
    return "fortran"
