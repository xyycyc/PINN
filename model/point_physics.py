"""Sparse spatial physics for fixed, irregular temperature-field nodes."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import cKDTree

from ..data_process.mesh_sampling import boundary_mask

POINT_PHYSICS_OPERATOR_VERSION = 1
DEFAULT_NEIGHBOR_COUNT = 4
DEFAULT_DISTANCE_EPSILON = 1e-12
POINT_SMOOTHNESS_COEFFICIENT = 0.05


class PointPhysicsOperator(nn.Module):
    """Build and cache a deterministic sparse graph for fixed physical nodes."""

    def __init__(
        self,
        coordinates_m: np.ndarray,
        node_ids: np.ndarray,
        material_ids: np.ndarray,
        interface_side: np.ndarray,
        *,
        neighbor_count: int = DEFAULT_NEIGHBOR_COUNT,
        distance_epsilon: float = DEFAULT_DISTANCE_EPSILON,
    ) -> None:
        super().__init__()
        coordinates = np.asarray(coordinates_m, dtype=np.float64)
        nodes = np.asarray(node_ids, dtype=np.int64)
        materials = np.asarray(material_ids, dtype=np.int64)
        interfaces = np.asarray(interface_side, dtype=np.int64)
        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError("coordinates_m must have shape [P, 2]")
        point_count = int(coordinates.shape[0])
        if not 1 <= point_count <= 10000:
            raise ValueError("point count must be in [1, 10000]")
        for name, values in (
            ("node_ids", nodes),
            ("material_ids", materials),
            ("interface_side", interfaces),
        ):
            if values.shape != (point_count,):
                raise ValueError(f"{name} must have shape [{point_count}]")
        if not np.all(np.isfinite(coordinates)):
            raise ValueError("coordinates_m contains non-finite values")
        if int(neighbor_count) <= 0:
            raise ValueError("neighbor_count must be > 0")
        if not np.isfinite(distance_epsilon) or float(distance_epsilon) <= 0:
            raise ValueError("distance_epsilon must be finite and > 0")

        self.point_count = point_count
        self.neighbor_count = int(neighbor_count)
        self.distance_epsilon = float(distance_epsilon)

        edge_pairs = self._build_undirected_edges(
            coordinates,
            nodes,
            materials,
            interfaces,
        )
        if edge_pairs:
            edge_array = np.asarray(edge_pairs, dtype=np.int64)
            edge_source = edge_array[:, 0]
            edge_target = edge_array[:, 1]
            undirected_deltas = coordinates[edge_target] - coordinates[edge_source]
            undirected_distance_squared = np.einsum(
                "ij,ij->i",
                undirected_deltas,
                undirected_deltas,
            )
            edge_weight = 1.0 / (
                undirected_distance_squared + self.distance_epsilon
            )
            directed_source = np.concatenate((edge_source, edge_target))
            directed_target = np.concatenate((edge_target, edge_source))
            deltas = coordinates[directed_target] - coordinates[directed_source]
            distance_squared = np.einsum("ij,ij->i", deltas, deltas)
            raw_weight = 1.0 / (distance_squared + self.distance_epsilon)
            source_weight_sum = np.bincount(
                directed_source,
                weights=raw_weight,
                minlength=point_count,
            )
            directed_weight = raw_weight / source_weight_sum[directed_source]
        else:
            edge_source = np.empty(0, dtype=np.int64)
            edge_target = np.empty(0, dtype=np.int64)
            edge_weight = np.empty(0, dtype=np.float64)
            directed_source = np.empty(0, dtype=np.int64)
            directed_target = np.empty(0, dtype=np.int64)
            directed_weight = np.empty(0, dtype=np.float64)

        degree = np.bincount(directed_source, minlength=point_count)
        exterior_boundary = boundary_mask(coordinates)
        interior_mask = (interfaces == 0) & ~exterior_boundary & (degree > 0)
        self.interior_count = int(np.sum(interior_mask))

        self.register_buffer(
            "edge_source",
            torch.from_numpy(np.ascontiguousarray(edge_source)),
        )
        self.register_buffer(
            "edge_target",
            torch.from_numpy(np.ascontiguousarray(edge_target)),
        )
        self.register_buffer(
            "edge_weight",
            torch.from_numpy(np.ascontiguousarray(edge_weight.astype(np.float32))),
        )
        self.register_buffer(
            "laplacian_source",
            torch.from_numpy(np.ascontiguousarray(directed_source)),
        )
        self.register_buffer(
            "laplacian_target",
            torch.from_numpy(np.ascontiguousarray(directed_target)),
        )
        self.register_buffer(
            "laplacian_weight",
            torch.from_numpy(np.ascontiguousarray(directed_weight.astype(np.float32))),
        )
        self.register_buffer(
            "interior_node_mask",
            torch.from_numpy(np.ascontiguousarray(interior_mask)),
        )
        self.register_buffer(
            "boundary_node_mask",
            torch.from_numpy(np.ascontiguousarray(exterior_boundary)),
        )

    def _build_undirected_edges(
        self,
        coordinates: np.ndarray,
        node_ids: np.ndarray,
        material_ids: np.ndarray,
        interface_side: np.ndarray,
    ) -> list[tuple[int, int]]:
        active = interface_side == 0
        edges: set[tuple[int, int]] = set()
        for material_id in np.unique(material_ids[active]):
            indices = np.flatnonzero(active & (material_ids == material_id))
            if len(indices) < 2:
                continue
            tree = cKDTree(coordinates[indices])
            available_neighbors = min(self.neighbor_count, len(indices) - 1)
            for local_source, source in enumerate(indices):
                candidates = self._nearest_valid_neighbors(
                    tree,
                    coordinates[indices],
                    indices,
                    node_ids,
                    local_source,
                    available_neighbors,
                )
                for target in candidates:
                    first, second = sorted((int(source), int(target)))
                    edges.add((first, second))
        return sorted(edges)

    @staticmethod
    def _nearest_valid_neighbors(
        tree: cKDTree,
        local_coordinates: np.ndarray,
        global_indices: np.ndarray,
        node_ids: np.ndarray,
        local_source: int,
        neighbor_count: int,
    ) -> list[int]:
        """Resolve distance ties by node id and array index without a dense matrix."""

        local_count = len(global_indices)
        query_count = min(local_count, max(neighbor_count + 1, 2))
        valid: list[tuple[float, int, int, int]] = []
        while True:
            distances, neighbors = tree.query(
                local_coordinates[local_source],
                k=query_count,
            )
            distances = np.atleast_1d(distances)
            neighbors = np.atleast_1d(neighbors)
            valid = []
            for distance, local_target in zip(distances, neighbors):
                local_target = int(local_target)
                if (
                    local_target == local_source
                    or local_target >= local_count
                    or not np.isfinite(distance)
                    or float(distance) <= 0.0
                ):
                    continue
                target = int(global_indices[local_target])
                valid.append(
                    (float(distance), int(node_ids[target]), target, local_target)
                )
            if len(valid) >= neighbor_count or query_count == local_count:
                break
            query_count = min(local_count, query_count * 2)

        if not valid:
            return []
        valid.sort()
        cutoff_index = min(neighbor_count, len(valid)) - 1
        cutoff_distance = valid[cutoff_index][0]
        tied_neighbors = tree.query_ball_point(
            local_coordinates[local_source],
            r=np.nextafter(cutoff_distance, np.inf),
        )
        tied_valid: list[tuple[float, int, int]] = []
        for local_target in tied_neighbors:
            local_target = int(local_target)
            if local_target == local_source:
                continue
            delta = local_coordinates[local_target] - local_coordinates[local_source]
            distance = float(np.sqrt(np.dot(delta, delta)))
            if distance <= 0.0:
                continue
            target = int(global_indices[local_target])
            tied_valid.append((distance, int(node_ids[target]), target))
        tied_valid.sort()
        return [target for _, _, target in tied_valid[:neighbor_count]]

    def _validate_prediction(self, prediction: torch.Tensor) -> None:
        if prediction.ndim != 2 or prediction.shape[1] != self.point_count:
            raise ValueError(
                f"prediction must have shape [B, {self.point_count}], "
                f"got {tuple(prediction.shape)}"
            )

    def smoothness_loss(self, prediction: torch.Tensor) -> torch.Tensor:
        """Mean absolute temperature difference over unique undirected edges."""

        self._validate_prediction(prediction)
        if self.edge_source.numel() == 0:
            return prediction.sum() * 0.0
        difference = prediction.index_select(1, self.edge_source) - prediction.index_select(
            1,
            self.edge_target,
        )
        return difference.abs().mean()

    def laplacian_loss(self, prediction: torch.Tensor) -> torch.Tensor:
        """Mean squared normalized graph Laplacian over valid interior centers."""

        self._validate_prediction(prediction)
        if (
            self.laplacian_source.numel() == 0
            or self.interior_count == 0
        ):
            return prediction.sum() * 0.0
        differences = prediction.index_select(
            1,
            self.laplacian_target,
        ) - prediction.index_select(1, self.laplacian_source)
        weighted = differences * self.laplacian_weight.to(dtype=prediction.dtype)
        laplacian = prediction.new_zeros((prediction.shape[0], self.point_count))
        laplacian.scatter_add_(
            1,
            self.laplacian_source.unsqueeze(0).expand(prediction.shape[0], -1),
            weighted,
        )
        return laplacian[:, self.interior_node_mask].square().mean()

    def metadata(self) -> dict[str, Any]:
        """Return portable configuration; graph edges are rebuilt from sampling data."""

        return {
            "operator_version": POINT_PHYSICS_OPERATOR_VERSION,
            "neighbor_count": self.neighbor_count,
            "distance_weight": "inverse_squared",
            "distance_epsilon": self.distance_epsilon,
            "same_material_only": True,
            "exclude_interface_nodes": True,
            "laplacian_excludes_boundary": True,
            "smoothness_coefficient": POINT_SMOOTHNESS_COEFFICIENT,
        }
