from __future__ import annotations

import unittest

import numpy as np
import torch

from ai_model.model.point_physics import PointPhysicsOperator


class PointPhysicsOperatorTests(unittest.TestCase):
    @staticmethod
    def _grid() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        coordinates = np.asarray(
            [(x, y) for y in range(4) for x in range(4)],
            dtype=np.float64,
        )
        node_ids = np.asarray(
            [100 + (index * 7) % 16 for index in range(16)],
            dtype=np.int64,
        )
        material_ids = np.asarray([1] * 8 + [2] * 8, dtype=np.int64)
        interface_side = np.zeros(16, dtype=np.int64)
        interface_side[[5, 10]] = (1, 2)
        return coordinates, node_ids, material_ids, interface_side

    def test_graph_is_deterministic_and_respects_material_and_interface(self):
        values = self._grid()
        first = PointPhysicsOperator(*values)
        second = PointPhysicsOperator(*values)

        self.assertTrue(torch.equal(first.edge_source, second.edge_source))
        self.assertTrue(torch.equal(first.edge_target, second.edge_target))
        self.assertTrue(torch.equal(first.laplacian_source, second.laplacian_source))
        self.assertTrue(torch.equal(first.laplacian_target, second.laplacian_target))
        self.assertTrue(torch.equal(first.interior_node_mask, second.interior_node_mask))
        self.assertTrue(torch.equal(first.edge_weight, second.edge_weight))
        self.assertTrue(torch.allclose(first.laplacian_weight, second.laplacian_weight))

        material_ids = values[2]
        interface_side = values[3]
        source = first.edge_source.numpy()
        target = first.edge_target.numpy()
        self.assertTrue(np.all(material_ids[source] == material_ids[target]))
        self.assertTrue(np.all(interface_side[source] == 0))
        self.assertTrue(np.all(interface_side[target] == 0))
        self.assertFalse(bool(first.interior_node_mask[5]))
        self.assertFalse(bool(first.interior_node_mask[10]))

    def test_boundary_nodes_smooth_but_are_not_laplacian_centers(self):
        coordinates, node_ids, _, interface_side = self._grid()
        material_ids = np.ones(len(node_ids), dtype=np.int64)
        operator = PointPhysicsOperator(
            coordinates,
            node_ids,
            material_ids,
            interface_side,
        )
        endpoints = torch.cat((operator.edge_source, operator.edge_target))
        boundary_indices = torch.nonzero(operator.boundary_node_mask).flatten()
        self.assertTrue(bool(torch.isin(boundary_indices, endpoints).any()))
        self.assertFalse(
            bool((operator.interior_node_mask & operator.boundary_node_mask).any())
        )
        self.assertTrue(bool(operator.interior_node_mask.any()))

    def test_losses_are_zero_for_constant_field_and_differentiable(self):
        coordinates, node_ids, material_ids, interface_side = self._grid()
        operator = PointPhysicsOperator(
            coordinates,
            node_ids,
            material_ids,
            interface_side,
        )
        constant = torch.full((2, 16), 3.5, requires_grad=True)
        constant_loss = (
            operator.smoothness_loss(constant)
            + operator.laplacian_loss(constant)
        )
        self.assertAlmostEqual(float(constant_loss.detach()), 0.0, places=7)

        prediction = torch.randn((2, 16), requires_grad=True)
        loss = operator.smoothness_loss(prediction) + operator.laplacian_loss(
            prediction
        )
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))

        spike = torch.zeros((1, 16))
        spike[0, 6] = 10.0
        self.assertGreater(
            float(operator.smoothness_loss(spike)),
            float(operator.smoothness_loss(torch.zeros_like(spike))),
        )

    def test_zero_distance_edges_and_empty_graph_are_safe(self):
        coordinates = np.asarray(
            [(0.0, 0.0), (0.0, 0.0), (1.0, 0.0)],
            dtype=np.float64,
        )
        operator = PointPhysicsOperator(
            coordinates,
            np.asarray([3, 1, 2]),
            np.ones(3),
            np.zeros(3),
        )
        for source, target in zip(
            operator.edge_source.tolist(),
            operator.edge_target.tolist(),
        ):
            self.assertGreater(
                float(np.linalg.norm(coordinates[source] - coordinates[target])),
                0.0,
            )

        empty = PointPhysicsOperator(
            coordinates,
            np.arange(3),
            np.ones(3),
            np.ones(3),
        )
        prediction = torch.randn((1, 3), requires_grad=True)
        loss = empty.smoothness_loss(prediction) + empty.laplacian_loss(prediction)
        self.assertEqual(float(loss.detach()), 0.0)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(prediction.grad).all()))


if __name__ == "__main__":
    unittest.main()
