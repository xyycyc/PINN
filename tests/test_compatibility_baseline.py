from __future__ import annotations
import json, unittest
import tempfile
import tkinter as tk
import os, time
from tkinter import ttk
from pathlib import Path
from ai_model.cli import _build_parser
from ai_model.window.app import TAB_GROUPS
from ai_model.window.tabs.base import BaseCommandTab
from ai_model.window.tabs import PathSettingsTab
from ai_model.window.settings import Settings

class CompatibilityBaselineTests(unittest.TestCase):
    def test_cli_commands_and_defaults(self):
        parser = _build_parser(); sub = next(a for a in parser._actions if a.dest == "command")
        self.assertEqual(set(sub.choices), {"build-db", "train", "validate", "online-update", "predict", "demo"})
        train = sub.choices["train"].parse_args([])
        self.assertEqual((train.device, train.training_mode, train.epochs), ("cuda", "normal", 20))
        build = sub.choices["build-db"].parse_args([])
        self.assertEqual((build.skip_simulation, build.split_seed), (True, 42))
    def test_visible_tab_order_and_settings_schema(self):
        self.assertEqual([c.title for c in TAB_GROUPS[0][1]], ["路径设置", "构建数据库", "训练模型", "预测对比", "校验数据", "增量训练", "一键演示", "项目清理"])
        path = Path(__file__).parents[1]/"window"/"settings.json"
        self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["$schema_version"], 1)

    def test_window_discovers_legacy_and_point_split_configs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); legacy=root/"data_process"/"legacy"; point=root/"case_temperature_field"
            legacy.mkdir(parents=True); point.mkdir()
            (legacy/"split_config.json").write_text(json.dumps({"manifests":{"train":"legacy.json"}}),encoding="utf-8")
            tab=object.__new__(BaseCommandTab); tab._resolve_data_root_path=lambda: root
            self.assertEqual(tab.resolve_latest_split_manifest("train"),"legacy.json")
            point_config=point/"split_config.json"
            point_config.write_text(json.dumps({"manifests":{"train":"point.json"}}),encoding="utf-8")
            os.utime(point_config,(time.time()+2,time.time()+2))
            self.assertEqual(tab.resolve_latest_split_manifest("train"),"point.json")

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
                    parsed=parser.parse_args(command[module_index+1:])
                    seen.append((cls.title,parsed.command))
            self.assertEqual(seen,[("构建数据库","build-db"),("训练模型","train"),("预测对比","predict"),
                                   ("校验数据","validate"),("增量训练","online-update"),("一键演示","demo")])
        finally:
            root.destroy()
