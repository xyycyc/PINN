# Fortran numerical library

This directory contains all Fortran source code and its Python binding.

## Files

- `ai_numeric.f90`: Fortran implementation.
- `ai_numeric.dll`: Compiled Windows library used at runtime.
- `native_bridge.py`: `ctypes` wrapper used by Python modules.
- `build_fortran.ps1`: Rebuild script for GNU Fortran.

## Exported routines

`compute_prediction_metrics` calculates temperature MAE, RMSE, and maximum
absolute error. It is called by `model/predict.py` when prediction metrics are
created.

`average_waveform_pairs` averages the time and voltage arrays from repeated
experiments. It is called by `database/raw/compute_metal_avg.py`.

Both routines use `real(c_double)` arrays and `integer(c_int64_t)` sizes through
Fortran `ISO_C_BINDING` interfaces.

## Rebuild on Windows

Run from the project root:

```powershell
powershell -ExecutionPolicy Bypass -File fortran/build_fortran.ps1
```

The current build expects `gfortran` to be available in `PATH`. Rebuild the DLL
after changing `ai_numeric.f90`.

The Python wrapper intentionally raises an error when the library is missing or
cannot be loaded. It does not silently replace Fortran calculations with NumPy.
