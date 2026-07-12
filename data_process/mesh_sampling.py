"""Deterministic mesh audit and spatially stratified node sampling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .temperature_field import SAMPLING_VERSION, mesh_fingerprint

CSV_COLUMNS = ["node_id", "x", "y", "T", "thermal_material_id", "dup_target_surface"]


def read_mesh_csv(path: str | Path) -> dict[str, np.ndarray]:
    frame = pd.read_csv(path, usecols=lambda c: c in CSV_COLUMNS, low_memory=False)
    required = {"node_id", "x", "y", "T"}
    if not required.issubset(frame.columns):
        raise ValueError(f"热力 CSV 缺少列: {sorted(required - set(frame.columns))}")
    coordinates = frame[["x", "y"]].to_numpy(np.float64)
    material = (frame["thermal_material_id"].fillna(-1).to_numpy(np.int32)
                if "thermal_material_id" in frame else np.full(len(frame), -1, np.int32))
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
        "interface_side": side,
    }


def _boundary_mask(xy: np.ndarray) -> np.ndarray:
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    tol = np.maximum((hi - lo) * 1e-9, 1e-12)
    return np.any((np.abs(xy - lo) <= tol) | (np.abs(xy - hi) <= tol), axis=1)


def deterministic_sample(mesh: dict[str, np.ndarray], target_points: int = 10000,
                         seed: int = 42) -> dict[str, Any]:
    count = len(mesh["node_ids"])
    if not 1 <= target_points <= min(10000, count):
        raise ValueError(f"target_points 必须在 [1, {min(10000, count)}]")
    xy = mesh["coordinates_m"]
    mandatory = _boundary_mask(xy) | (mesh["interface_side"] > 0)
    selected = list(np.flatnonzero(mandatory))
    if len(selected) > target_points:
        raise ValueError(f"关键边界/界面节点 {len(selected)} 超过目标点数 {target_points}")
    remaining_budget = target_points - len(selected)
    candidates = np.flatnonzero(~mandatory)
    rng = np.random.default_rng(seed)
    if remaining_budget:
        # Allocate the total target per material in proportion to source-node count,
        # while never dropping mandatory boundary/interface nodes.
        materials, source_counts = np.unique(mesh["material_ids"], return_counts=True)
        mandatory_counts = {int(m): int(np.sum(mandatory & (mesh["material_ids"] == m))) for m in materials}
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
    return {
        "sampling_version": SAMPLING_VERSION, "source_mesh_fingerprint": fingerprint,
        "indices": indices, "node_ids": mesh["node_ids"][indices],
        "coordinates_m": xy[indices].astype(np.float32), "material_ids": mesh["material_ids"][indices],
        "interface_side": mesh["interface_side"][indices], "sample_weights": weights,
        "sampling_method": "mandatory-boundary-interface+material-grid-round-robin",
        "sampling_seed": int(seed),
    }


def save_sampling_index(path: str | Path, sampling: dict[str, Any]) -> Path:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {k: v for k, v in sampling.items() if isinstance(v, np.ndarray)}
    metadata = {k: v for k, v in sampling.items() if k not in arrays}
    np.savez_compressed(path, **arrays, metadata_json=np.array(json.dumps(metadata, ensure_ascii=False)))
    return path
