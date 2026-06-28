"""AI-driven temperature field reconstruction package."""

import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from .config import AIModelConfig, MATERIAL_LIBRARY, MaterialSpec, get_material
from .data_process import AITemperatureDataset, DatabaseBuilder
from .model import AIReconstructionModel, OnlineUpdater, ReconstructionTrainer

__all__ = [
    "AIModelConfig",
    "AITemperatureDataset",
    "AIReconstructionModel",
    "MATERIAL_LIBRARY",
    "MaterialSpec",
    "DatabaseBuilder",
    "OnlineUpdater",
    "ReconstructionTrainer",
    "get_material",
]
