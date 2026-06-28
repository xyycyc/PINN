from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialSpec:
    key: str
    display_name: str
    density: float
    heat_capacity: float
    thermal_conductivity: float
    sound_speed_slope: float
    sound_speed_bias: float
    thickness_range_m: tuple[float, float]
    valid_dimensions: tuple[str, ...]


MATERIAL_LIBRARY: dict[str, MaterialSpec] = {
    "multilayer": MaterialSpec(
        key="multilayer",
        display_name="多层材料",
        density=2300.0,
        heat_capacity=840.0,
        thermal_conductivity=1.8,
        sound_speed_slope=-0.42,
        sound_speed_bias=3300.0,
        thickness_range_m=(0.005, 0.02),
        valid_dimensions=("1d", "2d"),
    ),
    "metal_matrix": MaterialSpec(
        key="metal_matrix",
        display_name="金属基复合材料",
        density=7800.0,
        heat_capacity=520.0,
        thermal_conductivity=18.0,
        sound_speed_slope=-0.65,
        sound_speed_bias=5900.0,
        thickness_range_m=(0.01, 0.03),
        valid_dimensions=("1d", "2d"),
    ),
    "carbon_silicon": MaterialSpec(
        key="carbon_silicon",
        display_name="碳基/硅基复合材料",
        density=1850.0,
        heat_capacity=720.0,
        thermal_conductivity=7.5,
        sound_speed_slope=-0.35,
        sound_speed_bias=4100.0,
        thickness_range_m=(0.005, 0.015),
        valid_dimensions=("1d", "2d"),
    ),
}


def get_material(material_key: str) -> MaterialSpec:
    try:
        return MATERIAL_LIBRARY[material_key]
    except KeyError as exc:
        choices = ", ".join(sorted(MATERIAL_LIBRARY))
        raise ValueError(f"未知材料 `{material_key}`，可选: {choices}") from exc
