"""Deterministic mesh audit and spatially stratified node sampling."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .temperature_field import SAMPLING_VERSION, mesh_fingerprint

CSV_COLUMNS = [
    "node_id",
    "x",
    "y",
    "T",
    "thermal_material_id",
    "thermal_material_name",
    "dup_target_surface",
]


def _fallback_material_name(material_id: int) -> str:
    return "unknown_material" if int(material_id) < 0 else f"material_{int(material_id)}"


def _normalized_material_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    return re.sub(r"\s+", " ", text).strip().casefold()


def _stable_material_id(material_name: str) -> int:
    """Map a normalized name into a reserved positive int32 range."""

    normalized = _normalized_material_name(material_name)
    if not normalized:
        raise ValueError("无效材料 ID 必须提供非空 thermal_material_name")
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    return 1_000_000_000 + int.from_bytes(digest[:8], "big") % 1_000_000_000


def _material_ids(
    frame: pd.DataFrame,
    *,
    material_id_policy: str,
) -> tuple[np.ndarray, bool]:
    if "thermal_material_id" in frame:
        numeric = pd.to_numeric(frame["thermal_material_id"], errors="coerce")
    else:
        numeric = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    invalid = numeric.isna() | (numeric < 0)
    if material_id_policy == "strict":
        return numeric.fillna(-1).to_numpy(np.int32), False
    if material_id_policy != "stable_name_for_invalid":
        raise ValueError(f"未知 material_id_policy: {material_id_policy}")
    if "thermal_material_name" not in frame and bool(invalid.any()):
        raise ValueError("无效材料 ID 无法规范化：缺少 thermal_material_name")

    effective = numeric.fillna(-1).to_numpy(np.int64)
    generated_names: dict[int, str] = {}
    valid_ids = {int(value) for value in effective[~invalid.to_numpy()]}
    names = frame["thermal_material_name"].fillna("").astype(str)
    for row in np.flatnonzero(invalid.to_numpy()):
        normalized_name = _normalized_material_name(names.iloc[row])
        generated_id = _stable_material_id(normalized_name)
        previous_name = generated_names.get(generated_id)
        if previous_name is not None and previous_name != normalized_name:
            raise ValueError(
                "规范化材料名称发生确定性 ID 冲突: "
                f"{previous_name!r}, {normalized_name!r}"
            )
        if generated_id in valid_ids:
            raise ValueError(
                f"规范化材料 ID={generated_id} 与有效原始材料 ID 冲突"
            )
        generated_names[generated_id] = normalized_name
        effective[row] = generated_id
    return effective.astype(np.int32), bool(invalid.any())


def _constituent_material_catalog(
    frame: pd.DataFrame,
    material_ids: np.ndarray,
    *,
    normalize_names: bool = False,
) -> list[dict[str, Any]]:
    """Build the node-level constituent catalog from the thermal CSV.

    These IDs describe layers/constituents inside one sample material.  They are
    never sample-level material classes and must not be used for checkpoint
    routing.
    """
    material_names = (
        frame["thermal_material_name"].fillna("").astype(str).str.strip()
        if "thermal_material_name" in frame
        else pd.Series([""] * len(frame), index=frame.index, dtype=str)
    )
    catalog: list[dict[str, Any]] = []
    for material_id in sorted(int(value) for value in np.unique(material_ids)):
        mask = material_ids == material_id
        names = sorted({value for value in material_names.loc[mask].tolist() if value})
        comparison_names = (
            {_normalized_material_name(value) for value in names}
            if normalize_names
            else set(names)
        )
        if len(comparison_names) > 1:
            raise ValueError(
                f"thermal_material_id={material_id} 对应多个材料名称: {names}"
            )
        catalog.append(
            {
                "material_id": material_id,
                "material_name": names[0] if names else _fallback_material_name(material_id),
                "source_node_count": int(np.sum(mask)),
            }
        )
    return catalog


def read_mesh_csv(
    path: str | Path,
    *,
    material_id_policy: str = "strict",
) -> dict[str, Any]:
    """Read one thermal mesh CSV and preserve node, material, and interface identity."""
    frame = pd.read_csv(path, usecols=lambda c: c in CSV_COLUMNS, low_memory=False)
    required = {"node_id", "x", "y", "T"}
    if not required.issubset(frame.columns):
        raise ValueError(f"热力 CSV 缺少列: {sorted(required - set(frame.columns))}")
    coordinates = frame[["x", "y"]].to_numpy(np.float64)
    material, normalized_invalid_ids = _material_ids(
        frame,
        material_id_policy=material_id_policy,
    )
    duplicated = frame.duplicated(["x", "y"], keep=False).to_numpy()
    # Stable side identity: material first, then node id. Non-interface nodes remain zero.
    side = np.zeros(len(frame), dtype=np.int32)
    if duplicated.any():
        dup_frame = frame.loc[duplicated, ["x", "y", "node_id"]].copy()
        ranks = dup_frame.groupby(["x", "y"], sort=False)["node_id"].rank(method="dense").astype(np.int32)
        side[np.flatnonzero(duplicated)] = ranks.to_numpy()
    return {
        "node_ids": frame["node_id"].to_numpy(np.int64),
        "coordinates_m": coordinates,
        "temperature_k": frame["T"].to_numpy(np.float32),
        "material_ids": material,
        "constituent_material_catalog": _constituent_material_catalog(
            frame,
            material,
            normalize_names=normalized_invalid_ids,
        ),
        "interface_side": side,
    }


def _boundary_mask(xy: np.ndarray) -> np.ndarray:
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    tol = np.maximum((hi - lo) * 1e-9, 1e-12)
    return np.any((np.abs(xy - lo) <= tol) | (np.abs(xy - hi) <= tol), axis=1)


def _paired_interface_selection(
    mesh: dict[str, Any],
    boundary: np.ndarray,
    target_points: int,
    seed: int,
) -> tuple[list[int], np.ndarray]:
    """Keep complete duplicate-coordinate interface groups within a fixed budget."""

    xy = mesh["coordinates_m"]
    interface = mesh["interface_side"] > 0
    selected = set(np.flatnonzero(boundary).tolist())
    groups: dict[tuple[float, float], list[int]] = {}
    for index in np.flatnonzero(interface).tolist():
        groups.setdefault(tuple(float(value) for value in xy[index]), []).append(index)
    for values in groups.values():
        values.sort(key=lambda index: int(mesh["node_ids"][index]))

    budget = target_points - len(selected)
    if budget < 0:
        raise ValueError(
            f"边界节点 {len(selected)} 超过目标点数 {target_points}"
        )
    # If a coordinate group touches the exterior boundary, complete that group
    # first so an interface is never represented by only one side.
    remaining_groups: list[tuple[tuple[float, float], list[int]]] = []
    for coordinate, values in sorted(groups.items()):
        missing = [index for index in values if index not in selected]
        if len(missing) != len(values):
            if len(missing) > budget:
                raise ValueError(
                    "边界上的完整界面坐标组超过剩余采样预算"
                )
            selected.update(missing)
            budget -= len(missing)
        else:
            remaining_groups.append((coordinate, values))

    if remaining_groups and budget:
        coordinates = np.asarray(
            [coordinate for coordinate, _values in remaining_groups],
            dtype=np.float64,
        )
        lo = coordinates.min(axis=0)
        span = np.maximum(np.ptp(coordinates, axis=0), 1e-12)
        approximate_groups = max(1, budget // 2)
        cells_per_axis = max(
            1,
            int(np.ceil(np.sqrt(approximate_groups * 1.5))),
        )
        cells = np.minimum(
            ((coordinates - lo) / span * cells_per_axis).astype(int),
            cells_per_axis - 1,
        )
        buckets: dict[tuple[int, int], list[list[int]]] = {}
        for (_coordinate, values), cell in zip(remaining_groups, cells):
            buckets.setdefault(tuple(cell.tolist()), []).append(values)
        rng = np.random.default_rng(seed)
        for values in buckets.values():
            rng.shuffle(values)
        bucket_keys = sorted(buckets)
        rng.shuffle(bucket_keys)
        cursor = 0
        stalled = 0
        while bucket_keys and budget > 0:
            key = bucket_keys[cursor % len(bucket_keys)]
            group = buckets[key][-1]
            if len(group) <= budget:
                selected.update(group)
                budget -= len(group)
                buckets[key].pop()
                stalled = 0
            else:
                stalled += 1
            if not buckets[key]:
                bucket_keys.remove(key)
                if bucket_keys:
                    cursor %= len(bucket_keys)
            else:
                cursor += 1
            if bucket_keys and stalled >= len(bucket_keys):
                break

    eligible = ~(boundary | interface)
    return sorted(selected), eligible


def deterministic_sample(mesh: dict[str, Any], target_points: int = 10000,
                         seed: int = 42,
                         interface_overflow_policy: str = "error") -> dict[str, Any]:
    """Select a reproducible spatially stratified node set under a fixed budget."""
    count = len(mesh["node_ids"])
    if not 1 <= target_points <= min(10000, count):
        raise ValueError(f"target_points 必须在 [1, {min(10000, count)}]")
    xy = mesh["coordinates_m"]
    boundary = _boundary_mask(xy)
    interface = mesh["interface_side"] > 0
    mandatory = boundary | interface
    selected = list(np.flatnonzero(mandatory))
    candidate_mask = ~mandatory
    if len(selected) > target_points:
        if interface_overflow_policy == "paired_stratified":
            selected, candidate_mask = _paired_interface_selection(
                mesh,
                boundary,
                target_points,
                seed,
            )
        elif interface_overflow_policy == "error":
            raise ValueError(f"关键边界/界面节点 {len(selected)} 超过目标点数 {target_points}")
        else:
            raise ValueError(
                f"未知 interface_overflow_policy: {interface_overflow_policy}"
            )
    remaining_budget = target_points - len(selected)
    candidates = np.flatnonzero(candidate_mask)
    selected_mask = np.zeros(count, dtype=bool)
    selected_mask[np.asarray(selected, dtype=np.int64)] = True
    rng = np.random.default_rng(seed)
    if remaining_budget:
        # Allocate the total target per material in proportion to source-node count,
        # while never dropping mandatory boundary/interface nodes.
        materials, source_counts = np.unique(mesh["material_ids"], return_counts=True)
        mandatory_counts = {
            int(material): int(
                np.sum(selected_mask & (mesh["material_ids"] == material))
            )
            for material in materials
        }
        desired = target_points * source_counts.astype(np.float64) / float(source_counts.sum())
        additional_raw = np.maximum(desired - np.array([mandatory_counts[int(m)] for m in materials]), 0.0)
        if additional_raw.sum() <= 0:
            additional_raw = source_counts.astype(np.float64)
        shares = remaining_budget * additional_raw / additional_raw.sum()
        additions = np.floor(shares).astype(int)
        for pos in np.argsort(-(shares - additions), kind="stable")[: remaining_budget - int(additions.sum())]:
            additions[pos] += 1

        lo, span = xy.min(axis=0), np.maximum(np.ptp(xy, axis=0), 1e-12)
        cells_per_axis = max(1, int(np.ceil(np.sqrt(target_points * 1.5))))
        all_cells = np.minimum(((xy - lo) / span * cells_per_axis).astype(int), cells_per_axis - 1)
        for material, budget in zip(materials.tolist(), additions.tolist()):
            material_candidates = candidates[mesh["material_ids"][candidates] == material]
            if budget > len(material_candidates):
                raise ValueError(f"材料 {material} 的采样预算 {budget} 超过可用内部节点 {len(material_candidates)}")
            groups: dict[tuple[int, int], list[int]] = {}
            for index in material_candidates.tolist():
                key = tuple(all_cells[index].tolist())
                groups.setdefault(key, []).append(index)
            # Seeded shuffle removes the old sorted-material/cell tail bias.
            for values in groups.values():
                values.sort(key=lambda i: int(mesh["node_ids"][i]))
                rng.shuffle(values)
            group_keys = sorted(groups)
            rng.shuffle(group_keys)
            chosen = 0
            cursor = 0
            while chosen < budget and group_keys:
                key = group_keys[cursor % len(group_keys)]
                selected.append(groups[key].pop())
                chosen += 1
                if not groups[key]:
                    group_keys.remove(key)
                    if group_keys:
                        cursor %= len(group_keys)
                else:
                    cursor += 1
    indices = np.array(sorted(selected, key=lambda i: int(mesh["node_ids"][i])), dtype=np.int64)
    # Each selected point represents its source cell population divided by the
    # number of selected points in that same material/cell stratum.
    sample_xy = xy[indices]
    bins = max(1, int(np.sqrt(len(indices))))
    lo, span = sample_xy.min(axis=0), np.maximum(np.ptp(sample_xy, axis=0), 1e-12)
    full_cell = np.minimum(((xy - lo) / span * bins).astype(int), bins - 1)
    sample_cell = full_cell[indices]
    full_keys, full_counts = np.unique(np.column_stack((mesh["material_ids"], full_cell)), axis=0, return_counts=True)
    sample_keys, inverse, sample_counts = np.unique(np.column_stack((mesh["material_ids"][indices], sample_cell)), axis=0, return_inverse=True, return_counts=True)
    population = {tuple(key.tolist()): int(value) for key, value in zip(full_keys, full_counts)}
    represented = np.array([population[tuple(key.tolist())] for key in sample_keys], dtype=np.float64)
    weights = (represented[inverse] / sample_counts[inverse]).astype(np.float32)
    weights /= weights.mean()
    fingerprint = mesh_fingerprint(mesh["node_ids"], xy, mesh["material_ids"], mesh["interface_side"])
    constituent_material_catalog = []
    for item in mesh.get("constituent_material_catalog", []):
        entry = dict(item)
        entry["sampled_point_count"] = int(
            np.sum(mesh["material_ids"][indices] == int(entry["material_id"]))
        )
        constituent_material_catalog.append(entry)
    if not constituent_material_catalog:
        for material_id in sorted(int(value) for value in np.unique(mesh["material_ids"])):
            constituent_material_catalog.append(
                {
                    "material_id": material_id,
                    "material_name": _fallback_material_name(material_id),
                    "source_node_count": int(np.sum(mesh["material_ids"] == material_id)),
                    "sampled_point_count": int(np.sum(mesh["material_ids"][indices] == material_id)),
                }
            )
    return {
        "sampling_version": SAMPLING_VERSION, "source_mesh_fingerprint": fingerprint,
        "indices": indices, "node_ids": mesh["node_ids"][indices],
        "coordinates_m": xy[indices].astype(np.float32), "material_ids": mesh["material_ids"][indices],
        "interface_side": mesh["interface_side"][indices], "sample_weights": weights,
        "constituent_material_catalog": constituent_material_catalog,
        "sampling_method": (
            "mandatory-boundary+paired-interface-stratified+material-grid-round-robin"
            if interface_overflow_policy == "paired_stratified"
            and int(np.sum(mandatory)) > target_points
            else "mandatory-boundary-interface+material-grid-round-robin"
        ),
        "sampling_seed": int(seed),
    }


def save_sampling_index(path: str | Path, sampling: dict[str, Any]) -> Path:
    """Persist sampling arrays and their versioned metadata as a compressed NPZ."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in sampling.items() if isinstance(v, np.ndarray)}
    metadata = {k: v for k, v in sampling.items() if k not in arrays}
    np.savez_compressed(path, **arrays, metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))
    return path
