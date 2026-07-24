from __future__ import annotations
import json, unittest
import tempfile
import tkinter as tk
import os, time
import threading
from tkinter import ttk
from pathlib import Path
from unittest.mock import patch
from ai_model.artifact_paths import normalize_pt_filename, validate_artifact_basename
from ai_model.cli import (
    _build_parser,
    _find_material_router_for_checkpoint,
    _parse_material_split_values,
    _reject_legacy_noop_weights,
    _resolve_project_path,
)
from ai_model.config import AIModelConfig
from ai_model.data_process import split_manifest_file, write_case_split_files
from ai_model.model import OnlineUpdater, ReconstructionTrainer
from ai_model.window.app import TAB_GROUPS
from ai_model.window.tabs.base import BaseCommandTab
from ai_model.window.tabs import (
    BuildDbTab,
    DemoTab,
    OnlineUpdateTab,
    PathSettingsTab,
    PredictTab,
    TrainTab,
    ValidateTab,
)
from ai_model.window.settings import DEFAULT_SETTINGS, Settings
from ai_model.window.runner import CommandRunner, python_module_command
from ai_model.window.rules import (
    _checkpoint_model_kind_cached,
    checkpoint_model_kind,
    resolve_gui_project_path,
)

class CompatibilityBaselineTests(unittest.TestCase):
    def test_managed_artifact_names_reject_paths_and_wrong_suffixes(self):
        for unsafe in ("../escape", "nested/name", r"nested\name", "C:escape", "NUL.pt"):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                validate_artifact_basename(unsafe, label="artifact")
        with self.assertRaises(ValueError):
            validate_artifact_basename("manifest.txt", label="manifest", suffix=".json")
        self.assertEqual(normalize_pt_filename("checkpoint", label="checkpoint"), "checkpoint.pt")
        self.assertEqual(normalize_pt_filename("checkpoint.PT", label="checkpoint"), "checkpoint.PT")
        with self.assertRaises(ValueError):
            normalize_pt_filename("checkpoint.bin", label="checkpoint")

    def test_split_manifest_names_cannot_collide_or_overwrite_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest = Path(tmp) / "combined_manifest.json"
            manifest.write_text(
                json.dumps({"dataset_name": "test", "records": []}),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "互不相同"):
                split_manifest_file(
                    manifest,
                    train_output_name="same.json",
                    test_output_name="SAME.json",
                )
            with self.assertRaisesRegex(ValueError, "保留文件"):
                split_manifest_file(
                    manifest,
                    train_output_name="combined_manifest.json",
                    test_output_name="test.json",
                )
            with self.assertRaisesRegex(ValueError, "互不相同"):
                write_case_split_files(
                    manifest,
                    train_name="same.json",
                    validation_name="same.json",
                    test_name="test.json",
                )

    def test_non_positive_training_epochs_are_rejected_by_backend(self):
        with self.assertRaisesRegex(ValueError, "epochs must be > 0"):
            ReconstructionTrainer(AIModelConfig(epochs=0, device="cpu"))
        with self.assertRaisesRegex(ValueError, "early_stopping_patience must be > 0"):
            ReconstructionTrainer(AIModelConfig(early_stopping_patience=0, device="cpu"))
        with self.assertRaisesRegex(ValueError, "online_epochs must be > 0"):
            OnlineUpdater(AIModelConfig(online_epochs=-1, device="cpu"))

    def test_registered_sample_material_checkpoint_discovers_its_router(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "field__wumu.pt"
            checkpoint.write_bytes(b"checkpoint placeholder")
            router = root / "field__material_router.json"
            router.write_text(
                json.dumps(
                    {
                        "router_kind": "sample_material_checkpoints",
                        "checkpoints": [{"material_key": "wumu", "checkpoint": checkpoint.name}],
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(_find_material_router_for_checkpoint(checkpoint), router.resolve())
            unrelated = root / "unrelated.pt"
            unrelated.write_bytes(b"checkpoint placeholder")
            self.assertIsNone(_find_material_router_for_checkpoint(unrelated))

    def test_checkpoint_kind_cache_reuses_unchanged_file_and_falls_back_without_mmap(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.pt"
            checkpoint.write_bytes(b"placeholder")
            _checkpoint_model_kind_cached.cache_clear()
            with patch(
                "ai_model.window.rules.torch.load",
                return_value={"model_kind": "direct_point_field"},
            ) as mocked_load:
                self.assertEqual(checkpoint_model_kind(checkpoint), "direct_point_field")
                self.assertEqual(checkpoint_model_kind(checkpoint), "direct_point_field")
                self.assertEqual(mocked_load.call_count, 1)
                self.assertTrue(mocked_load.call_args.kwargs["mmap"])

            _checkpoint_model_kind_cached.cache_clear()
            with patch(
                "ai_model.window.rules.torch.load",
                side_effect=[TypeError("mmap unsupported"), {"model_kind": "legacy_grid"}],
            ) as mocked_load:
                self.assertEqual(checkpoint_model_kind(checkpoint), "legacy_grid")
                self.assertEqual(mocked_load.call_count, 2)

    def test_cli_commands_and_defaults(self):
        parser = _build_parser(); sub = next(a for a in parser._actions if a.dest == "command")
        self.assertEqual(set(sub.choices), {"build-db", "train", "validate", "online-update", "predict", "demo"})
        train = sub.choices["train"].parse_args([])
        self.assertEqual((train.device, train.training_mode, train.epochs), ("cuda", "normal", 20))
        self.assertEqual(train.early_stopping_patience, 10)
        build = sub.choices["build-db"].parse_args([])
        self.assertEqual(
            (build.skip_simulation, build.split_seed, build.multi_material_input),
            (True, 42, False),
        )
        multi_build = sub.choices["build-db"].parse_args(
            [
                "--multi-material-input",
                "--material-split", "wumu=0.6,0.1,0.3",
                "--material-split", "steel=0.7,0.1,0.2",
            ]
        )
        self.assertTrue(multi_build.multi_material_input)
        self.assertEqual(
            _parse_material_split_values(multi_build.material_split),
            {"wumu": (0.6, 0.1, 0.3), "steel": (0.7, 0.1, 0.2)},
        )
        with self.assertRaisesRegex(ValueError, "比例之和"):
            _parse_material_split_values(["wumu=0.8,0.1,0.2"])
        predict = sub.choices["predict"].parse_args(["--manifest", "x.json"])
        update = sub.choices["online-update"].parse_args(["--manifest", "x.json"])
        self.assertEqual((predict.clip_quantile, predict.smooth_window), (None, None))
        self.assertEqual((update.clip_quantile, update.smooth_window), (None, None))
        obsolete = sub.choices["train"].parse_args(["--fixed-weight-material", "2"])
        with self.assertRaisesRegex(ValueError, "旧版无效参数"):
            _reject_legacy_noop_weights(obsolete)
        demo = sub.choices["demo"].parse_args(
            [
                "--rule-dimension", "two",
                "--rule-mode", "steady",
                "--rule-material", "layer_rule",
            ]
        )
        self.assertEqual(
            (demo.rule_dimension, demo.rule_mode, demo.rule_material),
            ("two", "steady", "layer_rule"),
        )

    def test_cli_paths_are_independent_of_subprocess_cwd(self):
        cfg = AIModelConfig()
        expected = cfg.repo_root / "database" / "train_manifest.json"
        self.assertEqual(_resolve_project_path(cfg, "database/train_manifest.json"), expected)
        self.assertEqual(_resolve_project_path(cfg, "ai_model/database/train_manifest.json"), expected)
        self.assertEqual(
            resolve_gui_project_path(
                "ai_model/database/train_manifest.json",
                repo_root=cfg.repo_root,
            ),
            expected,
        )

    def test_runner_keeps_package_importable_from_custom_launch_root(self):
        with tempfile.TemporaryDirectory() as tmp:
            finished = threading.Event()
            result: list[int | None] = []
            logs: list[str] = []
            runner = CommandRunner(log_callback=logs.append)
            started = runner.run(
                python_module_command("ai_model", "--help"),
                cwd=Path(tmp),
                on_finish=lambda code: (result.append(code), finished.set()),
            )
            self.assertTrue(started)
            self.assertTrue(finished.wait(15), "custom launch_root subprocess timed out")
            self.assertEqual(result, [0], "".join(logs))
    def test_visible_tab_order_and_settings_schema(self):
        self.assertEqual([c.title for c in TAB_GROUPS[0][1]], ["路径设置", "构建数据库", "训练模型", "预测对比", "校验数据", "增量训练", "一键演示", "项目清理"])
        path = Path(__file__).parents[1]/"window"/"settings.json"
        saved = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(saved["$schema_version"], 1)
        for section, defaults in DEFAULT_SETTINGS.items():
            if isinstance(defaults, dict):
                self.assertFalse(set(defaults) - set(saved.get(section, {})), section)

    def test_window_discovers_legacy_and_point_split_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); legacy=root/"data_process"/"legacy"; point=root/"case_temperature_field"
            legacy.mkdir(parents=True); point.mkdir()
            (legacy/"legacy.json").write_text('{"records":[{}]}', encoding="utf-8")
            (legacy/"split_config.json").write_text(json.dumps({"manifests":{"train":"legacy.json"}}),encoding="utf-8")
            tab=object.__new__(BaseCommandTab); tab._resolve_data_root_path=lambda: root
            self.assertEqual(tab.resolve_latest_split_manifest("train"),str((legacy/"legacy.json").resolve()))
            point_config=point/"split_config.json"
            (point/"point.json").write_text('{"records":[{}]}', encoding="utf-8")
            point_config.write_text(json.dumps({"manifests":{"train":"point.json"}}),encoding="utf-8")
            os.utime(point_config,(time.time()+2,time.time()+2))
            self.assertEqual(tab.resolve_latest_split_manifest("train"),str((point/"point.json").resolve()))
            empty_dir=root/"data_process"/"newer_empty"; empty_dir.mkdir()
            (empty_dir/"empty.json").write_text('{"records":[]}', encoding="utf-8")
            empty_config=empty_dir/"split_config.json"
            empty_config.write_text(json.dumps({"manifests":{"train":"empty.json"}}),encoding="utf-8")
            os.utime(empty_config,(time.time()+4,time.time()+4))
            self.assertEqual(tab.resolve_latest_split_manifest("train"),str((point/"point.json").resolve()))

    def test_manual_manifest_is_not_replaced_when_auto_selection_is_disabled(self):
        try:
            root = tk.Tk(); root.withdraw()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        parser = _build_parser(); notebook = ttk.Notebook(root)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                repo_root = Path(tmp)
                split_dir = repo_root / "database" / "data_process" / "latest"
                split_dir.mkdir(parents=True)
                (split_dir / "split_config.json").write_text(
                    json.dumps(
                        {
                            "manifests": {
                                "train": "latest_train.json",
                                "test": "latest_test.json",
                                "combined": "latest_combined.json",
                            }
                        }
                    ),
                    encoding="utf-8",
                )
                kwargs = {
                    "repo_root": repo_root,
                    "settings": Settings(),
                    "run_callback": lambda _cmd: None,
                    "stop_callback": lambda: None,
                }
                path_tab = PathSettingsTab(notebook, **kwargs)
                cases = (
                    (TrainTab, "manual_train.json"),
                    (PredictTab, "manual_test.json"),
                    (ValidateTab, "manual_combined_validate.json"),
                    (OnlineUpdateTab, "manual_combined_update.json"),
                )
                for tab_cls, manual_manifest in cases:
                    tab = tab_cls(notebook, path_settings_tab=path_tab, **kwargs)
                    tab.auto_manifest.set(False)
                    tab.manifest.set(manual_manifest)
                    command = tab.compose_command()
                    module_index = command.index("ai_model")
                    parsed = parser.parse_args(command[module_index + 1:])
                    self.assertEqual(parsed.manifest, manual_manifest, tab_cls.__name__)
                    self.assertEqual(tab.manifest.get(), manual_manifest, tab_cls.__name__)
                    self.assertFalse(tab.to_settings_section()["auto_manifest"])
        finally:
            root.destroy()

    def test_auto_manifest_fallback_uses_configured_data_root(self):
        try:
            root = tk.Tk(); root.withdraw()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        parser = _build_parser(); notebook = ttk.Notebook(root)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                repo_root = Path(tmp)
                data_root = repo_root / "custom_database"
                data_root.mkdir()
                kwargs = {
                    "repo_root": repo_root,
                    "settings": Settings(),
                    "run_callback": lambda _cmd: None,
                    "stop_callback": lambda: None,
                }
                path_tab = PathSettingsTab(notebook, **kwargs)
                path_tab.io_roots["data_root"].set(str(data_root))
                cases = (
                    (TrainTab, None),
                    (PredictTab, data_root / "test_manifest.json"),
                    (ValidateTab, None),
                    (OnlineUpdateTab, data_root / "combined_manifest.json"),
                )
                for tab_cls, expected in cases:
                    tab = tab_cls(notebook, path_settings_tab=path_tab, **kwargs)
                    command = tab.compose_command()
                    parsed = parser.parse_args(command[command.index("ai_model") + 1:])
                    self.assertEqual(parsed.manifest, str(expected) if expected else None)
        finally:
            root.destroy()

    def test_every_command_tab_composes_cli_accepted_arguments(self):
        try:
            root=tk.Tk(); root.withdraw()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        parser=_build_parser(); notebook=ttk.Notebook(root); path_ref=None; seen=[]
        try:
            with tempfile.TemporaryDirectory() as tmp:
                for cls in TAB_GROUPS[0][1]:
                    kwargs={"repo_root":Path(tmp),"settings":Settings(),"run_callback":lambda _cmd:None,"stop_callback":lambda:None}
                    if path_ref is not None and cls is not PathSettingsTab:
                        kwargs["path_settings_tab"]=path_ref
                    tab=cls(notebook,**kwargs)
                    if isinstance(tab,PathSettingsTab): path_ref=tab
                    if cls.title in {"路径设置","项目清理"}: continue
                    command=tab.compose_command(); module_index=command.index("ai_model")
                    if cls in {PredictTab, OnlineUpdateTab}:
                        self.assertNotIn("--preprocess", command)
                        self.assertNotIn("--clip-quantile", command)
                        self.assertNotIn("--smooth-window", command)
                    parsed=parser.parse_args(command[module_index+1:])
                    seen.append((cls.title,parsed.command))
            self.assertEqual(seen,[("构建数据库","build-db"),("训练模型","train"),("预测对比","predict"),
                                   ("校验数据","validate"),("增量训练","online-update"),("一键演示","demo")])
        finally:
            root.destroy()

    def test_new_gui_controls_reach_cli_arguments(self):
        try:
            root = tk.Tk(); root.withdraw()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        parser = _build_parser(); notebook = ttk.Notebook(root)
        kwargs = {
            "repo_root": Path(__file__).parents[1],
            "settings": Settings(),
            "run_callback": lambda _cmd: None,
            "stop_callback": lambda: None,
        }
        try:
            path_tab = PathSettingsTab(notebook, **kwargs)
            for tab_cls in (BuildDbTab, TrainTab, DemoTab):
                tab = tab_cls(notebook, path_settings_tab=path_tab, **kwargs)
                if isinstance(tab, TrainTab):
                    tab.runtime["fixed_weight_cnn"].set(1.2)
                    tab.runtime["fixed_weight_lstm"].set(1.3)
                command = tab.compose_command()
                parsed = parser.parse_args(command[command.index("ai_model") + 1:])
                if isinstance(tab, BuildDbTab):
                    self.assertEqual(parsed.waveform_crop_length, 1097)
                    self.assertEqual(parsed.split_validation_ratio, 0.1)
                    self.assertEqual(parsed.validation_manifest_name, "validation_manifest.json")
                    tab.validation_manifest_name.set(tab.train_manifest_name.get())
                    with self.assertRaisesRegex(ValueError, "互不相同"):
                        tab.validate_form()
                    tab.validation_manifest_name.set("validation_manifest.json")
                elif isinstance(tab, TrainTab):
                    self.assertEqual(
                        (parsed.fixed_weight_cnn, parsed.fixed_weight_lstm),
                        (1.2, 1.3),
                    )
                    self.assertEqual(parsed.early_stopping_patience, 10)
                else:
                    self.assertEqual(parsed.waveform_crop_length, 1097)
                    self.assertEqual(parsed.split_validation_ratio, 0.1)
                    tab.experiment_material.set("dataset_label")
                    tab.rule_material.set("checkpoint_rule")
                    command = tab.compose_command()
                    parsed = parser.parse_args(command[command.index("ai_model") + 1:])
                    self.assertEqual(parsed.experiment_material, "dataset_label")
                    self.assertEqual(parsed.rule_material, "checkpoint_rule")
        finally:
            root.destroy()

    def test_build_gui_scans_sibling_materials_and_emits_independent_splits(self):
        try:
            root = tk.Tk(); root.withdraw()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        parser = _build_parser(); notebook = ttk.Notebook(root)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                repo_root = Path(tmp)
                source = repo_root / "source"
                (source / "wumu" / "worker_01" / "case_0000_T0300p000K").mkdir(parents=True)
                (source / "wumu" / "layer_1").mkdir()
                (source / "wumu" / "layer_2").mkdir()
                (source / "steel" / "case_0000_T0300p000K").mkdir(parents=True)
                kwargs = {
                    "repo_root": repo_root,
                    "settings": Settings(),
                    "run_callback": lambda _cmd: None,
                    "stop_callback": lambda: None,
                }
                path_tab = PathSettingsTab(notebook, **kwargs)
                build = BuildDbTab(
                    notebook,
                    path_settings_tab=path_tab,
                    **kwargs,
                )
                build.experiment_dir.set(str(source))
                build.multi_material_input.set(True)
                build.material_splits.refresh()
                self.assertEqual(set(build.material_splits._rows), {"wumu", "steel"})
                self.assertNotIn("layer_1", build.material_splits._rows)
                train, validation, test = build.material_splits._rows["wumu"]
                train.set("0.6"); validation.set("0.1"); test.set("0.3")
                build.validate_form()
                command = build.compose_command()
                parsed = parser.parse_args(command[command.index("ai_model") + 1:])
                self.assertTrue(parsed.multi_material_input)
                self.assertEqual(
                    _parse_material_split_values(parsed.material_split),
                    {"steel": (0.7, 0.1, 0.2), "wumu": (0.6, 0.1, 0.3)},
                )

                external_source = repo_root / "wumu_exp"
                external_source.mkdir()
                train.set("0.9"); validation.set("0.1"); test.set("0")
                build.external_test_enabled.set(True)
                build.external_test_dir.set(str(external_source))
                build.external_test_material.set("wumu")
                build.external_test_dataset_name.set("wumu_exp")
                build.validate_form()
                command = build.compose_command()
                parsed = parser.parse_args(command[command.index("ai_model") + 1:])
                self.assertEqual(parsed.external_test_dir, str(external_source))
                self.assertEqual(parsed.external_test_material, "wumu")
                self.assertEqual(
                    _parse_material_split_values(parsed.material_split)["wumu"],
                    (0.9, 0.1, 0.0),
                )
        finally:
            root.destroy()

    def test_online_validation_preserves_user_runtime_and_predict_passes_auto_device(self):
        try:
            root = tk.Tk(); root.withdraw()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        parser = _build_parser(); notebook = ttk.Notebook(root)
        kwargs = {
            "repo_root": Path(__file__).parents[1],
            "settings": Settings(),
            "run_callback": lambda _cmd: None,
            "stop_callback": lambda: None,
        }
        try:
            path_tab = PathSettingsTab(notebook, **kwargs)
            online = OnlineUpdateTab(notebook, path_settings_tab=path_tab, **kwargs)
            online.runtime["epochs"].set(17)
            online.runtime["device"].set("cpu")
            online._rule_context = online._rule_context_key()
            with patch.object(online, "_selected_checkpoint_path", return_value="selected.pt"):
                online.validate_form()
                online.runtime["epochs"].set(0)
                with self.assertRaisesRegex(ValueError, "轮数"):
                    online.validate_form()
                online.runtime["epochs"].set(17)
            self.assertEqual(online.runtime["epochs"].get(), 17)
            self.assertEqual(online.runtime["device"].get(), "cpu")

            predict = PredictTab(notebook, path_settings_tab=path_tab, **kwargs)
            predict.infer_device.set("auto")
            predict.rule_dimension.set("one")
            predict._update_prediction_dimension_options()
            with patch.object(predict, "_selected_checkpoint_path", return_value="selected.pt"):
                command = predict.compose_command()
            parsed = parser.parse_args(command[command.index("ai_model") + 1:])
            self.assertEqual(parsed.device, "auto")
            self.assertEqual(parsed.prediction_dimension, "one")
            predict.rule_dimension.set("two")
            predict._update_prediction_dimension_options()
            self.assertEqual(
                tuple(predict.prediction_dimension.combo.cget("values")),
                ("two", "one"),
            )
            predict.prediction_dimension.set("one")
            with patch.object(predict, "_selected_checkpoint_path", return_value="selected.pt"):
                command = predict.compose_command()
            parsed = parser.parse_args(command[command.index("ai_model") + 1:])
            self.assertEqual(parsed.prediction_dimension, "one")
        finally:
            root.destroy()
