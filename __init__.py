"""AI-driven temperature field reconstruction with lazy public imports."""

from __future__ import annotations

from importlib import import_module

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_EXPORTS = {
    "AIModelConfig": (".config", "AIModelConfig"),
    "AITemperatureDataset": (".data_process", "AITemperatureDataset"),
    "AIReconstructionModel": (".model", "AIReconstructionModel"),
    "MATERIAL_LIBRARY": (".config", "MATERIAL_LIBRARY"),
    "MaterialSpec": (".config", "MaterialSpec"),
    "DatabaseBuilder": (".data_process", "DatabaseBuilder"),
    "OnlineUpdater": (".model", "OnlineUpdater"),
    "ReconstructionTrainer": (".model", "ReconstructionTrainer"),
    "get_material": (".config", "get_material"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
