"""全局配置、材料库与物理仿真（前向）。"""

from .config import AIModelConfig
from .materials import MATERIAL_LIBRARY, MaterialSpec, get_material
from .physics import SimulationCase, build_heat_flux, simulate_case

__all__ = [
    "AIModelConfig",
    "MATERIAL_LIBRARY",
    "MaterialSpec",
    "SimulationCase",
    "build_heat_flux",
    "get_material",
    "simulate_case",
]
