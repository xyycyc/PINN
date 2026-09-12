from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
import unittest
from pathlib import Path
from tkinter import ttk
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ai_model.paths import PACKAGE_ROOT, resolve_project_path
from ai_model.window.app import AiModelApp
from ai_model.window.runner import CommandRunner, format_command
from ai_model.window.settings import DEFAULT_SETTINGS, Settings, load_settings
from ai_model.window.widgets import ScrollableFrame


class EntryPointTests(unittest.TestCase):
    def test_help_works_without_site_packages_from_both_entrypoints(self):
        for command, cwd in (
            ([sys.executable, "-S", "-m", "ai_model", "--help"], PACKAGE_ROOT.parent),
            (
                [sys.executable, "-S", str(PACKAGE_ROOT / "run.py"), "train", "--help"],
                tempfile.gettempdir(),
            ),
        ):
            with self.subTest(command=command):
                result = subprocess.run(
                    command, cwd=cwd, capture_output=True, timeout=10
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(b"train", result.stdout)

    def test_lightweight_imports_do_not_load_training_or_gui(self):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                "import sys; import ai_model; import ai_model.window.runner; "
                "assert 'torch' not in sys.modules; assert 'ai_model.window.app' not in sys.modules; "
                "assert 'ReconstructionTrainer' in dir(ai_model)",
            ],
            cwd=PACKAGE_ROOT.parent,
            capture_output=True,
            timeout=10,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_project_paths_and_user_directory_are_shared(self):
        from ai_model.config import AIModelConfig
        from ai_model.cli import _resolve_project_path
        from ai_model.window.rules import resolve_gui_project_path

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "moved project"
            config = AIModelConfig(repo_root=root)
            for raw in (
                "database/test.json",
                "moved project/database/test.json",
                "~/data",
                root / "database/test.json",
            ):
                with self.subTest(raw=raw):
                    expected = resolve_project_path(raw, repo_root=root)
                    self.assertEqual(config.resolve_path(raw), expected)
                    self.assertEqual(_resolve_project_path(config, raw), expected)
                    self.assertEqual(
                        resolve_gui_project_path(raw, repo_root=root), expected
                    )
            self.assertEqual(config.resolve_path("~/data"), Path.home() / "data")


class SettingsTests(unittest.TestCase):
    def test_manual_default_preserves_explicit_opt_in(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "settings.json"
            target.write_text('{"train":{"auto_manifest":true}}', encoding="utf-8-sig")
            loaded = load_settings(target)
            self.assertTrue(loaded.get("train", "auto_manifest"))
            for section in ("train", "predict", "validate", "online_update"):
                self.assertFalse(Settings().get(section, "auto_manifest"))

    def test_nested_form_values_cannot_mutate_defaults_or_settings(self):
        original = {"build_db": {"material_splits": {"steel": {"train": 0.7}}}}
        settings = Settings(original)
        original["build_db"]["material_splits"]["steel"]["train"] = 0.1
        section = settings.section("build_db")
        section["material_splits"]["steel"]["train"] = 0.2
        self.assertEqual(
            settings.get("build_db", "material_splits")["steel"]["train"], 0.7
        )
        self.assertEqual(DEFAULT_SETTINGS["build_db"]["material_splits"], {})

    def test_failed_save_preserves_disk_and_memory_and_removes_temp(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.json"
            settings = Settings({"common": {"data_root": "old"}}, path=path)
            settings.save()
            before = path.read_bytes()
            with patch(
                "ai_model.window.settings.os.replace",
                side_effect=PermissionError("locked"),
            ):
                with self.assertRaises(PermissionError):
                    settings.save_sections({"common": {"data_root": "new"}})
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(settings.get("common", "data_root"), "old")
            self.assertEqual(list(path.parent.glob("*.tmp")), [])
            settings.save_sections({"common": {"data_root": "新路径"}})
            self.assertEqual(load_settings(path).get("common", "data_root"), "新路径")

    def test_malformed_section_falls_back_to_defaults(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "settings.json"
            path.write_text('{"common":null}', encoding="utf-8")
            self.assertEqual(load_settings(path).get("common", "data_root"), "database")

    def test_save_all_forms_does_not_partly_update_after_bad_form(self):
        app = AiModelApp.__new__(AiModelApp)
        app.settings = Settings()
        app._tabs = [
            SimpleNamespace(
                settings_section="common",
                to_settings_section=lambda: {"data_root": "new"},
            ),
            SimpleNamespace(
                settings_section="train",
                to_settings_section=Mock(side_effect=ValueError("bad number")),
            ),
        ]
        with patch("ai_model.window.app.messagebox.showerror") as error:
            app.save_settings_from_forms()
        error.assert_called_once()
        self.assertEqual(app.settings.get("common", "data_root"), "database")


class RunnerTests(unittest.TestCase):
    def test_launch_errors_are_reported_and_do_not_occupy_runner(self):
        logs, codes = [], []
        runner = CommandRunner(log_callback=logs.append)
        with patch(
            "ai_model.window.runner.subprocess.Popen",
            side_effect=PermissionError("denied"),
        ):
            self.assertFalse(runner.run([sys.executable], on_finish=codes.append))
        self.assertEqual(codes, [-1])
        self.assertFalse(runner.is_running)
        self.assertIn("启动失败", "".join(logs))

    def test_exited_process_stays_busy_until_its_output_is_drained(self):
        reading, release, finished = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def log(line):
            if line.strip() == "drain marker":
                reading.set()
                release.wait(5)

        runner = CommandRunner(log_callback=log)
        try:
            self.assertTrue(
                runner.run(
                    [sys.executable, "-c", "print('drain marker')"],
                    on_finish=lambda code: finished.set(),
                )
            )
            self.assertTrue(reading.wait(5))
            self.assertIsNotNone(runner._proc)
            runner._proc.wait(timeout=5)
            self.assertTrue(runner.is_running)
            self.assertFalse(runner.run([sys.executable, "-c", "pass"]))
        finally:
            release.set()
            runner.stop()
        self.assertTrue(finished.wait(5))
        self.assertFalse(runner.is_running)

    def test_request_stop_does_not_wait_on_termination(self):
        entered, release = threading.Event(), threading.Event()
        runner = CommandRunner()
        process = Mock()
        runner._proc = process

        def blocking_stop(proc):
            self.assertIs(proc, process)
            entered.set()
            release.wait(5)

        with patch.object(runner, "_stop_process", side_effect=blocking_stop):
            try:
                runner.request_stop()
                self.assertTrue(entered.wait(2))
                self.assertFalse(release.is_set())
                runner.request_stop()
            finally:
                release.set()
                runner._stop_thread.join(timeout=5)

    def test_stop_real_child_reports_completion(self):
        ready, finished = threading.Event(), threading.Event()
        codes = []
        runner = CommandRunner(
            log_callback=lambda line: ready.set() if line.strip() == "ready" else None
        )
        try:
            runner.run(
                [sys.executable, "-c", "import time; print('ready'); time.sleep(30)"],
                on_finish=lambda code: (codes.append(code), finished.set()),
            )
            self.assertTrue(ready.wait(5), "unbuffered child output was not delivered")
            runner.request_stop()
            self.assertTrue(finished.wait(10))
            self.assertFalse(runner.is_running)
            self.assertEqual(len(codes), 1)
        finally:
            runner.stop()

    @unittest.skipUnless(os.name == "nt", "PowerShell quoting is Windows-specific")
    def test_copyable_command_preserves_special_characters(self):
        values = ["a b", "O'Brien", "$value; & literal", "中文"]
        command = format_command(
            [
                sys.executable,
                "-c",
                "import json,sys; print(json.dumps(sys.argv[1:]))",
                *values,
            ]
        )
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            timeout=10,
            creationflags=subprocess.CREATE_NO_WINDOW,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), values)


class GuiEventTests(unittest.TestCase):
    def test_complete_app_reloads_custom_file_and_preserves_logs(self):
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        try:
            with tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                config_path = base / "custom-settings.json"
                settings = Settings(path=config_path)
                settings.update_section(
                    "common",
                    {
                        "data_root": str(base / "data"),
                        "result_root": str(base / "result"),
                    },
                )
                for section in ("train", "predict", "validate", "online_update"):
                    settings.update_section(
                        section,
                        {"manifest": str(base / "missing.json"), "checkpoint": ""},
                    )
                settings.save()
                errors = []
                root.report_callback_exception = lambda *args: errors.append(args)
                app = AiModelApp(root, settings=settings)
                root.update_idletasks()
                app._append_to_text("persist this log\n")
                with patch(
                    "ai_model.window.app.messagebox.askyesno", return_value=True
                ):
                    app.reload_settings()
                root.update_idletasks()
                self.assertIn("persist this log", app.log_text.get("1.0", "end"))
                self.assertEqual(app.settings.path, config_path)
                self.assertEqual(len(app._tabs), 8)
                self.assertEqual(errors, [])
                app._on_close()
        finally:
            try:
                root.destroy()
            except tk.TclError:
                pass

    def test_finish_callback_never_calls_tk_and_stale_events_are_ignored(self):
        app = AiModelApp.__new__(AiModelApp)
        app.root = Mock()
        app._log_queue = queue.Queue()
        app._run_id = 2
        app._closing = False
        app._set_status = Mock()
        app.task_status = Mock()
        app._set_busy = Mock()
        app._log_visible = True
        app._append_to_text = Mock()
        app._on_finish(9, 1)
        app.root.after.assert_not_called()
        app._log_queue.put("first\n")
        app._log_queue.put("second\n")
        app._on_finish(0, 2)
        app._poll_log_queue()
        app._set_status.assert_called_once_with("已结束，退出码：0")
        app._append_to_text.assert_called_once_with("first\nsecond\n")

    def test_unsaved_working_directory_is_used(self):
        app = AiModelApp.__new__(AiModelApp)
        app.package_root = PACKAGE_ROOT
        app.settings = Settings({"common": {"launch_root": "old"}})
        app._path_settings_tab = SimpleNamespace(
            launch_env={"launch_root": SimpleNamespace(get=lambda: "current")}
        )
        self.assertEqual(app._resolve_launch_root(), PACKAGE_ROOT / "current")

    def test_mousewheel_is_scoped_and_bindings_are_cleaned_up(self):
        try:
            root = tk.Tk()
            root.withdraw()
        except tk.TclError as exc:
            self.skipTest(str(exc))
        try:
            first, second = ScrollableFrame(root), ScrollableFrame(root)
            inside = ttk.Entry(first.inner)
            outside = tk.Text(root)
            with patch.object(
                first.canvas, "yview", return_value=(0.1, 0.5)
            ), patch.object(first.canvas, "yview_scroll") as scroll:
                self.assertEqual(
                    first._on_mousewheel(SimpleNamespace(widget=inside, delta=-120)),
                    "break",
                )
                self.assertIsNone(
                    first._on_mousewheel(SimpleNamespace(widget=outside, delta=-120))
                )
                self.assertIsNone(
                    first._on_mousewheel(
                        SimpleNamespace(widget=second.inner, delta=-120)
                    )
                )
                scroll.assert_called_once_with(1, "units")
            second_bindings = dict(second._wheel_bindings)
            first.destroy()
            for sequence, binding in second_bindings.items():
                self.assertIn(binding, root.bind(sequence))
        finally:
            root.destroy()
