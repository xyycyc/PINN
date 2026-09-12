"""Check the selected Python and native runtime without reading production data.

Run from the project root: python tools/check_environment.py
"""

from __future__ import annotations

import importlib
import math
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT.parent))


def main() -> int:
    from ai_model.window.runner import format_command

    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version.split()[0]}")
    errors = []
    if sys.version_info < (3, 10):
        errors.append("Python 3.10 or newer is required.")
    for name in (
        "numpy",
        "torch",
        "pandas",
        "scipy",
        "matplotlib",
        "tqdm",
        "PIL",
        "tkinter",
    ):
        try:
            module = importlib.import_module(name)
            version = getattr(
                module, "__version__", getattr(module, "TkVersion", "available")
            )
            print(f"OK {name}: {version}")
        except (ImportError, OSError, RuntimeError) as exc:
            errors.append(f"{name}: {exc}")
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        print("Install project dependencies with this same interpreter:")
        print(
            format_command(
                [
                    sys.executable,
                    "-m",
                    "pip",
                    "install",
                    "-r",
                    str(PACKAGE_ROOT / "requirements.txt"),
                ]
            )
        )
        print(
            "Tkinter must be included in the Python installation; it is not a pip package."
        )
        return 1

    import numpy as np
    from ai_model.fortran.native_bridge import (
        average_waveform_pairs,
        backend_name,
        compute_prediction_metrics,
    )

    try:
        metrics = compute_prediction_metrics(
            np.array([0.0, 2.0, 4.0]), np.array([1.0, 2.0, 6.0])
        )
        expected = (1.0, math.sqrt(5 / 3), 2.0)
        if not all(
            math.isclose(actual, value, rel_tol=1e-12)
            for actual, value in zip(metrics, expected)
        ):
            raise RuntimeError(
                f"Unexpected native metrics: {metrics}; expected {expected}"
            )
        times, voltages = average_waveform_pairs(
            np.array([[0.0, 1.0], [2.0, 3.0]]), np.array([[2.0, 4.0], [4.0, 8.0]])
        )
        np.testing.assert_allclose(times, [1.0, 2.0])
        np.testing.assert_allclose(voltages, [3.0, 6.0])
        print(
            f"OK native backend: {backend_name()} (metrics and waveform averages verified)"
        )
    except (OSError, RuntimeError, AssertionError) as exc:
        print(f"ERROR native backend: {exc}")
        print("Build instructions: fortran/README.md")
        return 1
    print(
        "Environment check passed. GUI interaction can be checked with tools/gui_smoke.py."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
