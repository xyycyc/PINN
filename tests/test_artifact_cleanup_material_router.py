from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import torch

from ai_model.config import AIModelConfig
from ai_model.model.artifact_cleanup import build_cleanup_plan, execute_cleanup
from ai_model.model.rule_registry import (
    build_rule_record,
    ensure_rule_csv,
    register_checkpoint_rule,
)
from ai_model.model.trainer import MATERIAL_ROUTER_KIND


class MaterialRouterCleanupTests(unittest.TestCase):
    def test_cleanup_rejects_checkpoint_from_another_result_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = AIModelConfig(data_root=root / "data", result_root=root / "result_a")
            second = AIModelConfig(data_root=root / "data", result_root=root / "result_b")
            checkpoint = first.train_checkpoint_root / "run" / "model.pt"
            checkpoint.parent.mkdir(parents=True)
            torch.save({"model_state": {}}, checkpoint)

            with self.assertRaisesRegex(ValueError, "不在当前输出根"):
                build_cleanup_plan(
                    data_root=second.data_root,
                    result_root=second.result_root,
                    checkpoint_path=checkpoint,
                )
            self.assertTrue(checkpoint.exists())

    def test_router_cannot_expand_cleanup_outside_checkpoint_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AIModelConfig(data_root=root / "data", result_root=root / "result")
            checkpoint_dir = config.train_checkpoint_root / "run"
            checkpoint_dir.mkdir(parents=True)
            target = checkpoint_dir / "inside.pt"
            outside = root / "outside.pt"
            torch.save({"model_state": {}}, target)
            torch.save({"model_state": {}}, outside)
            (checkpoint_dir / "field__material_router.json").write_text(
                json.dumps(
                    {
                        "router_kind": MATERIAL_ROUTER_KIND,
                        "checkpoints": [
                            {"checkpoint": target.name},
                            {"checkpoint": str(outside)},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "分材料 checkpoint"):
                build_cleanup_plan(
                    data_root=config.data_root,
                    result_root=config.result_root,
                    checkpoint_path=target,
                )
            self.assertTrue(outside.exists())

    def test_cleaning_one_material_checkpoint_removes_the_complete_router_group(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = AIModelConfig(
                data_root=root / "database",
                result_root=root / "result",
                device="cpu",
            )
            config.ensure_dirs()
            checkpoint_dir = config.train_checkpoint_root / "material_run"
            report_dir = config.train_report_root / "material_run"
            checkpoint_dir.mkdir(parents=True)
            report_dir.mkdir(parents=True)

            checkpoints = [checkpoint_dir / "field__wumu.pt", checkpoint_dir / "field__steel.pt"]
            for path in checkpoints:
                torch.save({"model_state": {}, "config": {"device": "cpu"}}, path)
                (report_dir / f"{path.stem}_history.json").write_text("{}", encoding="utf-8")
            summary = report_dir / "training_summary.json"
            summary.write_text("{}", encoding="utf-8")

            router = checkpoint_dir / "field__material_router.json"
            router.write_text(
                json.dumps(
                    {
                        "router_kind": MATERIAL_ROUTER_KIND,
                        "checkpoints": [
                            {"material_key": material, "checkpoint": path.name}
                            for material, path in zip(("wumu", "steel"), checkpoints)
                        ],
                    }
                ),
                encoding="utf-8",
            )

            registry = ensure_rule_csv(config.data_root)
            for material, path in zip(("wumu", "steel"), checkpoints):
                register_checkpoint_rule(
                    registry,
                    build_rule_record(
                        dimension="two",
                        mode="steady",
                        material=material,
                        checkpoint_path=path,
                        training_mode="normal",
                        physics_residual_weight=0.1,
                        learnable_branch_weights=False,
                        fixed_weight_cnn=0.75,
                        fixed_weight_lstm=0.75,
                        fixed_weight_material=1.0,
                        fixed_weight_dimension=1.0,
                        fixed_weight_mode=1.0,
                    ),
                )

            predict_dir = config.predict_inference_root / "material_prediction"
            predict_dir.mkdir(parents=True)
            (predict_dir / "metrics.json").write_text(
                json.dumps({"checkpoint": str(router.resolve())}),
                encoding="utf-8",
            )

            plan = build_cleanup_plan(
                data_root=config.data_root,
                result_root=config.result_root,
                checkpoint_path=checkpoints[0],
            )
            self.assertEqual(set(plan.material_group_checkpoints), set(checkpoints))
            self.assertEqual(plan.material_router_paths, [router.resolve()])
            self.assertEqual(plan.csv_rows_for_target, 2)
            self.assertIn(summary, plan.target_report_paths)
            self.assertIn(predict_dir, plan.predict_dirs[str(router.resolve())])

            execute_cleanup(
                data_root=config.data_root,
                result_root=config.result_root,
                checkpoint_path=checkpoints[0],
                include_dependents=False,
                promote_dependents=False,
            )
            self.assertFalse(router.exists())
            self.assertFalse(predict_dir.exists())
            self.assertTrue(all(not path.exists() for path in checkpoints))
            with registry.open("r", encoding="utf-8", newline="") as handle:
                self.assertEqual(list(csv.DictReader(handle)), [])


if __name__ == "__main__":
    unittest.main()
