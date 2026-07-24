from __future__ import annotations

import copy
import tkinter as tk
import unittest
from pathlib import Path
from tkinter import ttk
from unittest.mock import patch

from ai_model.window.settings import DEFAULT_SETTINGS, Settings
from ai_model.window.tabs.manage_artifacts import (
    ManageArtifactsTab,
    _standalone_root_default,
)
from ai_model.window.tabs.path_settings import PathSettingsTab


class ManageArtifactsPathTests(unittest.TestCase):
    def test_standalone_defaults_do_not_hide_custom_common_roots(self) -> None:
        local = copy.deepcopy(DEFAULT_SETTINGS["manage_artifacts"])
        common = {"data_root": "D:/shared-data", "result_root": "D:/shared-result"}

        self.assertEqual(
            _standalone_root_default(common, local, "data_root", "database"),
            "D:/shared-data",
        )
        self.assertEqual(
            _standalone_root_default(common, local, "result_root", "result"),
            "D:/shared-result",
        )

        local["data_root"] = "D:/standalone-data"
        self.assertEqual(
            _standalone_root_default(common, local, "data_root", "database"),
            "D:/standalone-data",
        )

    def test_cleanup_tab_tracks_path_settings_roots_live(self) -> None:
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError as exc:
            self.skipTest(str(exc))

        settings_data = copy.deepcopy(DEFAULT_SETTINGS)
        settings_data["common"]["data_root"] = "D:/initial-data"
        settings_data["common"]["result_root"] = "D:/initial-result"
        settings = Settings(settings_data)
        notebook = ttk.Notebook(root)
        no_run = lambda _cmd: None
        no_stop = lambda: None
        repo_root = Path(__file__).resolve().parents[1]

        try:
            path_tab = PathSettingsTab(
                notebook,
                run_callback=no_run,
                stop_callback=no_stop,
                repo_root=repo_root,
                settings=settings,
            )
            with patch(
                "ai_model.window.tabs.manage_artifacts.load_rule_rows",
                return_value=[],
            ):
                cleanup_tab = ManageArtifactsTab(
                    notebook,
                    run_callback=no_run,
                    stop_callback=no_stop,
                    repo_root=repo_root,
                    settings=settings,
                    path_settings_tab=path_tab,
                )

            self.assertEqual(cleanup_tab._cfg_data_root(), "D:/initial-data")
            self.assertEqual(cleanup_tab._cfg_result_root(), "D:/initial-result")

            cleanup_tab._checkpoint_label_to_path["old-root-model"] = "D:/initial-result/model.pt"
            cleanup_tab.checkpoint.combo.configure(values=["old-root-model"], state="readonly")
            cleanup_tab.checkpoint.set("old-root-model")
            cleanup_tab._latest_plan = object()
            path_tab.io_roots["data_root"].set("E:/live-data")
            path_tab.io_roots["result_root"].set("E:/live-result")
            self.assertEqual(cleanup_tab.data_root.get(), "E:/live-data")
            self.assertEqual(cleanup_tab.result_root.get(), "E:/live-result")
            self.assertEqual(cleanup_tab._cfg_data_root(), "E:/live-data")
            self.assertEqual(cleanup_tab._cfg_result_root(), "E:/live-result")
            self.assertEqual(cleanup_tab._selected_checkpoint_path(), "")
            self.assertIsNone(cleanup_tab._latest_plan)

            cleanup_tab.data_root.set("F:/edited-from-cleanup")
            self.assertEqual(
                path_tab.io_roots["data_root"].get(),
                "F:/edited-from-cleanup",
            )
        finally:
            root.destroy()


if __name__ == "__main__":
    unittest.main()
