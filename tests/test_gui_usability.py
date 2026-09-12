from __future__ import annotations

import tempfile
import tkinter as tk
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ai_model.paths import resolve_source_path
from ai_model.window.theme import configure_styles
from ai_model.window.widgets import (
    FileEntry,
    FormValidationError,
    LabeledCheck,
    LabeledNumber,
)


class SourcePathTests(unittest.TestCase):
    def test_import_paths_match_cli_and_gui_with_custom_data_root(self):
        from ai_model.commands import _resolve_source_dir
        from ai_model.config import AIModelConfig

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "moved project"
            data = Path(temporary) / "external data"
            config = AIModelConfig(repo_root=root)
            config.data_root = data
            for raw, expected in (
                ("raw/material", data / "raw/material"),
                ("database/raw/material", root / "database/raw/material"),
                ("~/measurements", Path.home() / "measurements"),
                (data / "absolute", data / "absolute"),
            ):
                with self.subTest(raw=raw):
                    self.assertEqual(
                        resolve_source_path(raw, data_root=data, repo_root=root),
                        expected.resolve(),
                    )
                    self.assertEqual(
                        _resolve_source_dir(config, raw), expected.resolve()
                    )


class FormInteractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.root = tk.Tk()
            cls.root.withdraw()
        except tk.TclError as error:
            raise unittest.SkipTest(str(error))
        configure_styles(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def tearDown(self):
        for child in self.root.winfo_children():
            child.destroy()

    def test_invalid_and_nonfinite_numbers_mark_the_field_and_clear_on_edit(self):
        control = LabeledNumber(self.root, "学习率", 0.1, is_float=True)
        for invalid in ("abc", "nan", "inf", "-inf"):
            control.var.set(invalid)
            with self.assertRaises(FormValidationError) as caught:
                control.get()
            self.assertIs(caught.exception.widget, control.entry)
            self.assertEqual(control.entry.cget("style"), "Invalid.TEntry")
            control.set(0.25)
            self.assertEqual(control.get(), 0.25)
            self.assertEqual(control.entry.cget("style"), "TEntry")
        control.var.set("")
        self.assertIsNone(control.get())

    def test_browse_uses_project_root_and_nearest_existing_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.root.repo_root = Path(temporary)
            data = Path(temporary) / "database"
            data.mkdir()
            control = FileEntry(
                self.root, "数据集", "database/missing/nested", directory=True
            )
            with patch(
                "ai_model.window.widgets.filedialog.askdirectory", return_value=""
            ) as choose:
                control.browse_button.invoke()
            self.assertEqual(Path(choose.call_args.kwargs["initialdir"]), data)
            self.assertIs(choose.call_args.kwargs["parent"], self.root)
            self.assertEqual(control.get(), "database/missing/nested")
            del self.root.repo_root

    def test_source_browser_follows_the_current_root_and_keeps_save_filename(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "first").mkdir()
            (root / "second").mkdir()
            current = [root / "first"]
            control = FileEntry(
                self.root,
                "导出文件",
                "new/result.json",
                save=True,
                resolver=lambda value: current[0] / value,
            )
            current[0] = root / "second"
            with patch(
                "ai_model.window.widgets.filedialog.asksaveasfilename", return_value=""
            ) as choose:
                control.browse_button.invoke()
            self.assertEqual(
                Path(choose.call_args.kwargs["initialdir"]), root / "second"
            )
            self.assertEqual(choose.call_args.kwargs["initialfile"], "result.json")

    def test_wrapped_checkbox_caption_uses_the_same_callback_and_disabled_state(self):
        control = LabeledCheck(self.root, "开关", "一段可换行的说明")
        callback = Mock()
        control.check.configure(command=callback)
        control._toggle(None)
        self.assertTrue(control.get())
        callback.assert_called_once()
        control.check.configure(state="disabled")
        control._toggle(None)
        self.assertTrue(control.get())
        callback.assert_called_once()
