from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ai_model.batch.batch_test_modes import _resolve_project_path
from ai_model.batch.plot_batch_test_loss_curves import _discover_histories
from ai_model.batch.rerun_predict import rerun_predictions
from ai_model.batch.search_fixed_weights import _build_search_config, run_fixed_weight_search
from ai_model.config import AIModelConfig


class BatchCompatibilityTests(unittest.TestCase):
    def test_relative_batch_roots_match_main_cli_root(self) -> None:
        config = AIModelConfig()
        self.assertEqual(
            _resolve_project_path("database/raw"),
            config.repo_root / "database" / "raw",
        )
        self.assertEqual(
            _resolve_project_path("ai_model/database/raw"),
            config.repo_root / "database" / "raw",
        )
        absolute = config.repo_root / "custom-result"
        self.assertEqual(_resolve_project_path(absolute), absolute)

    def test_search_config_weights_cnn_and_lstm_independently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _build_search_config(
                Path(tmp),
                device="cpu",
                epochs=1,
                seed=42,
                training_mode="normal",
                physics_residual_weight=0.1,
                cnn_weight=0.5,
                lstm_weight=1.5,
            )
            self.assertEqual((cfg.fixed_weight_cnn, cfg.fixed_weight_lstm), (0.5, 1.5))

    def test_history_discovery_supports_current_and_legacy_layouts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            current = (
                run_dir
                / "normal_fixed"
                / "result"
                / "train"
                / "report"
                / "normal_fixed"
                / "normal_fixed_history.json"
            )
            legacy = (
                run_dir
                / "normal_learnable"
                / "outputs"
                / "reports"
                / "normal_learnable_history.json"
            )
            current.parent.mkdir(parents=True)
            legacy.parent.mkdir(parents=True)
            current.write_text(json.dumps([{"epoch": 1, "mean_epoch_loss": 1.0}]), encoding="utf-8")
            legacy.write_text(json.dumps([{"epoch": 1, "mean_epoch_loss": 2.0}]), encoding="utf-8")

            histories = _discover_histories(run_dir)

            self.assertEqual(set(histories), {"normal_fixed", "normal_learnable"})
            self.assertEqual(histories["normal_fixed"][0]["mean_epoch_loss"], 1.0)

    def test_rerun_predict_explicitly_enables_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "batch_test_current"
            manifest = run_dir / "manifests" / "mse_test_manifest.json"
            mode_dir = run_dir / "normal_fixed"
            checkpoint = (
                mode_dir
                / "result"
                / "train"
                / "checkpoint"
                / "normal_fixed"
                / "normal_fixed.pt"
            )
            manifest.parent.mkdir(parents=True)
            checkpoint.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({"records": []}), encoding="utf-8")
            checkpoint.touch()

            metrics = {
                "samples": 1,
                "temperature_mae": 1.0,
                "temperature_rmse": 2.0,
                "field_loss": 3.0,
                "acoustic_loss": 4.0,
                "total_loss": 5.0,
            }
            with (
                patch("ai_model.batch.rerun_predict.sync_config_for_inference"),
                patch("ai_model.batch.rerun_predict.predict_and_compare", return_value=metrics) as predict,
            ):
                rerun_predictions(run_dir, device="cpu", overwrite=False, num_field_samples=3)

            self.assertTrue(predict.call_args.kwargs["enable_plots"])
            self.assertEqual(predict.call_args.kwargs["num_field_samples"], 3)

    def test_removed_physical_weight_search_fails_instead_of_repeating_noop_runs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, "旧版无效搜索维度"):
                run_fixed_weight_search(
                    data_root=root,
                    result_root=root,
                    test_ratio=0.2,
                    epochs=1,
                    device="cpu",
                    seed=42,
                    training_mode="normal",
                    physics_residual_weight=0.1,
                    network_weights=[1.0],
                    physical_weights=[2.0],
                )


if __name__ == "__main__":
    unittest.main()
