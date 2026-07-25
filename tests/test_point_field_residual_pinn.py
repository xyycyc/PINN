from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ai_model.config import AIModelConfig
from ai_model.model.point_physics import (
    POINT_SMOOTHNESS_COEFFICIENT,
    PointPhysicsOperator,
)
from ai_model.model.trainer import (
    OnlineUpdater,
    ReconstructionTrainer,
    _point_field_objective,
)


class PointFieldResidualPinnTests(unittest.TestCase):
    def test_objective_modes_and_default_are_compatible(self):
        self.assertEqual(AIModelConfig().training_mode, "normal")
        coordinates = np.asarray(
            [(x, y) for y in range(3) for x in range(3)],
            dtype=np.float64,
        )
        operator = PointPhysicsOperator(
            coordinates,
            np.arange(9),
            np.ones(9),
            np.zeros(9),
        )
        prediction = torch.linspace(0.0, 1.0, 9).unsqueeze(0)
        target = torch.zeros_like(prediction)
        weights = torch.ones_like(prediction)

        normal_total, normal_parts = _point_field_objective(
            prediction,
            target,
            weights,
            training_mode="normal",
            physics_residual_weight=0.7,
            point_physics=None,
        )
        self.assertTrue(torch.equal(normal_total, normal_parts["field_loss"]))
        self.assertEqual(float(normal_parts["smoothness_loss"]), 0.0)
        self.assertEqual(float(normal_parts["physics_residual_loss"]), 0.0)

        residual_total, residual_parts = _point_field_objective(
            prediction,
            target,
            weights,
            training_mode="residual_pinn",
            physics_residual_weight=0.7,
            point_physics=operator,
        )
        expected = (
            residual_parts["field_loss"]
            + POINT_SMOOTHNESS_COEFFICIENT * residual_parts["smoothness_loss"]
            + 0.7 * residual_parts["physics_residual_loss"]
        )
        self.assertTrue(torch.allclose(residual_total, expected))

    @staticmethod
    def _write_dataset(root: Path) -> tuple[Path, Path]:
        (root / "waveforms").mkdir(parents=True)
        (root / "temperature_fields").mkdir()
        coordinates = np.asarray(
            [(x, y) for y in range(3) for x in range(3)],
            dtype=np.float32,
        )
        node_ids = np.arange(101, 110, dtype=np.int64)
        material_ids = np.ones(9, dtype=np.int32)
        interface_side = np.zeros(9, dtype=np.int32)
        sample_weights = np.ones(9, dtype=np.float32)
        metadata = {
            "sampling_version": "point-physics-test",
            "source_mesh_fingerprint": "fixed-grid-fingerprint",
        }
        np.savez_compressed(
            root / "sampling_index.npz",
            node_ids=node_ids,
            coordinates_m=coordinates,
            material_ids=material_ids,
            interface_side=interface_side,
            sample_weights=sample_weights,
            metadata_json=np.array(json.dumps(metadata)),
        )
        records: list[dict[str, object]] = []
        for index, base_temperature in enumerate((300.0, 330.0, 360.0)):
            waveform_path = root / "waveforms" / f"wave_{index}.npy"
            field_path = root / "temperature_fields" / f"field_{index}.npz"
            np.save(
                waveform_path,
                np.linspace(0.0, 1.0 + index, 16, dtype=np.float32),
            )
            temperature = (
                base_temperature + np.arange(9, dtype=np.float32)
            )
            np.savez_compressed(
                field_path,
                temperature_k=temperature,
                coordinates_m=coordinates,
                node_ids=node_ids,
                material_ids=material_ids,
                interface_side=interface_side,
                sample_weights=sample_weights,
            )
            records.append(
                {
                    "sample_id": f"sample-{index}",
                    "source": "simulation",
                    "waveform_path": str(waveform_path.relative_to(root)),
                    "field_path": str(field_path.relative_to(root)),
                    "temperature_k": base_temperature,
                    "material_key": "test_material",
                    "dimension": "2d",
                    "mode": "steady",
                }
            )
        normalization = {
            "fit_split": "train",
            "mean_k": 334.0,
            "std_k": 25.0,
        }
        common = {
            "schema_version": 2,
            "sampling_index": "sampling_index.npz",
            "normalization": normalization,
        }
        train_manifest = root / "train_manifest.json"
        validation_manifest = root / "validation_manifest.json"
        train_manifest.write_text(
            json.dumps({**common, "records": records[:2]}),
            encoding="utf-8",
        )
        validation_manifest.write_text(
            json.dumps({**common, "records": records[2:]}),
            encoding="utf-8",
        )
        return train_manifest, validation_manifest

    def test_training_validation_checkpoint_and_incremental_modes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_manifest, validation_manifest = self._write_dataset(root)
            checkpoints: dict[str, Path] = {}
            for mode in ("normal", "residual_pinn"):
                config = AIModelConfig(
                    data_root=root / f"database-{mode}",
                    result_root=root / f"result-{mode}",
                    training_mode=mode,
                    physics_residual_weight=0.3,
                    epochs=1,
                    online_epochs=1,
                    early_stopping_patience=1,
                    batch_size=1,
                    hidden_dim=4,
                    latent_dim=4,
                    device="cpu",
                )
                checkpoint = ReconstructionTrainer(config).train(
                    train_manifest,
                    validation_manifest_path=validation_manifest,
                    checkpoint_name=f"{mode}.pt",
                    train_name=mode,
                )
                checkpoints[mode] = checkpoint
                bundle = torch.load(checkpoint, map_location="cpu")
                row = bundle["history"][0]
                self.assertEqual(row["training_mode"], mode)
                self.assertEqual(row["acoustic_loss"], 0.0)
                self.assertEqual(row["temperature_loss"], 0.0)
                self.assertIn("validation_field_loss", row)
                expected_validation = (
                    row["validation_field_loss"]
                    + POINT_SMOOTHNESS_COEFFICIENT
                    * row["validation_smoothness_loss"]
                    + config.physics_residual_weight
                    * row["validation_physics_residual_loss"]
                )
                self.assertAlmostEqual(
                    row["validation_loss"],
                    expected_validation,
                    places=6,
                )
                if mode == "normal":
                    self.assertEqual(row["total_loss"], row["field_loss"])
                    self.assertEqual(row["smoothness_loss"], 0.0)
                    self.assertEqual(row["physics_residual_loss"], 0.0)
                    self.assertIsNone(bundle["point_physics"])
                else:
                    self.assertEqual(
                        bundle["point_physics"]["operator_version"],
                        1,
                    )
                    self.assertEqual(
                        bundle["point_physics"]["neighbor_count"],
                        4,
                    )

                incremental = OnlineUpdater(config).update(
                    train_manifest,
                    checkpoint,
                    output_name=f"{mode}-incremental.pt",
                )
                incremental_bundle = torch.load(incremental, map_location="cpu")
                incremental_row = incremental_bundle["history"][0]
                self.assertEqual(incremental_row["training_mode"], mode)
                self.assertIn("temperature_mae_k", incremental_row)
                self.assertTrue(np.isfinite(incremental_row["total_loss"]))


if __name__ == "__main__":
    unittest.main()
