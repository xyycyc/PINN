"""Exercise the real Tk UI with temporary data, never production datasets.

Run: python tools/gui_smoke.py [--screenshots] [--output audit/gui_usability/smoke]
Screenshots need Windows, Pillow and pywin32. The interaction checks use Tk only.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import ttk
from unittest.mock import patch

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE_ROOT.parent))

from ai_model.window.app import AiModelApp
from ai_model.window.settings import Settings
from ai_model.window.widgets import FileEntry


def capture(window: tk.Misc, path: Path) -> None:
    """Render only our own Tk window; do not capture the user's desktop."""
    import ctypes
    import win32gui
    import win32ui
    from PIL import Image

    window.update()
    hwnd = win32gui.GetParent(window.winfo_id()) or window.winfo_id()
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    width, height = right - left, bottom - top
    source = win32gui.GetWindowDC(hwnd)
    dc = win32ui.CreateDCFromHandle(source)
    memory = dc.CreateCompatibleDC()
    bitmap = win32ui.CreateBitmap()
    bitmap.CreateCompatibleBitmap(dc, width, height)
    memory.SelectObject(bitmap)
    try:
        if not ctypes.windll.user32.PrintWindow(hwnd, memory.GetSafeHdc(), 2):
            raise RuntimeError("PrintWindow failed")
        Image.frombuffer(
            "RGB", (width, height), bitmap.GetBitmapBits(True), "raw", "BGRX", 0, 1
        ).save(path)
    finally:
        win32gui.DeleteObject(bitmap.GetHandle())
        memory.DeleteDC()
        dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, source)


def fixture(base: Path) -> Settings:
    dataset = base / "database" / "sample_dataset"
    dataset.mkdir(parents=True)
    for name in ("train", "validation", "test"):
        (dataset / f"{name}_manifest.json").write_text(
            json.dumps({"dataset_name": "GUI scenario", "records": []}),
            encoding="utf-8",
        )
    settings = Settings(path=base / "settings.json")
    settings.update_section(
        "common",
        {"data_root": str(base / "database"), "result_root": str(base / "result")},
    )
    settings.update_section("train", {"manifest": str(dataset), "auto_manifest": False})
    for section in ("predict", "validate", "online_update"):
        settings.update_section(
            section,
            {
                "manifest": str(dataset / "test_manifest.json"),
                "checkpoint": "",
                "auto_manifest": False,
            },
        )
    settings.save()
    return settings


def descendants(widget):
    for child in widget.winfo_children():
        yield child
        yield from descendants(child)


def exercise(root: tk.Tk, app: AiModelApp, output: Path, screenshots: bool) -> dict:
    scenarios = []
    pages = []

    def snapshot(name, window=root):
        if screenshots:
            capture(window, output / f"{name}.png")

    def pump_until(predicate, timeout=20):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            root.update()
            if predicate():
                return
            time.sleep(0.01)
        raise AssertionError("Timed out while driving the Tk event loop")

    train = next(tab for tab in app._tabs if tab.settings_section == "train")
    validate = next(tab for tab in app._tabs if tab.settings_section == "validate")
    cleanup = next(
        tab for tab in app._tabs if tab.settings_section == "manage_artifacts"
    )
    root.deiconify()
    for width, height in ((1000, 640), (1280, 820), (1440, 900)):
        root.geometry(f"{width}x{height}+40+20")
        for tab in app._tabs:
            tab.master.select(tab)
            root.update()
            clipped = []
            for control in descendants(tab):
                if isinstance(control, FileEntry) and control.entry.winfo_ismapped():
                    assert control.entry.winfo_width() >= 260, (
                        tab.title,
                        control.entry.winfo_width(),
                    )
                if (
                    isinstance(
                        control, (ttk.Entry, ttk.Combobox, ttk.Label, ttk.Button)
                    )
                    and control.winfo_ismapped()
                ):
                    if (
                        control.winfo_rootx() + control.winfo_width()
                        > root.winfo_rootx() + root.winfo_width()
                    ):
                        clipped.append(str(control))
            assert not clipped, (width, height, tab.title, clipped)
            button = tab._delete_button if tab is cleanup else tab._run_button
            assert (
                button.winfo_rooty() + button.winfo_height()
                < root.winfo_rooty() + root.winfo_height()
            )
            pages.append(
                {
                    "width": width,
                    "height": height,
                    "tab": tab.title,
                    "clipped_controls": len(clipped),
                }
            )
            if width == 1280 or (width == 1000 and tab is train):
                snapshot(f"{width}-{tab.settings_section}")
        # Pane visibility must be measured, not inferred from the toggle label.
        train.master.select(train)
        app.toggle_log()
        root.update()
        assert app._log_frame.winfo_ismapped() and app.log_text.winfo_height() >= 30
        assert (
            train._run_button.winfo_rooty() + train._run_button.winfo_height()
            < root.winfo_rooty() + root.winfo_height()
        )
        app.toggle_log()
        root.update()
    scenarios.append(
        "24 page/size combinations: readable path fields and visible actions"
    )

    root.geometry("1280x820+40+20")
    train.master.select(train)
    root.update()
    before = app.settings.path.read_bytes()
    train.runtime["epochs"].var.set("wrong")
    train._run_button.invoke()
    root.update()
    assert not app.runner.is_running
    assert "整数" in train._feedback.get()
    entry = train.runtime["epochs"].entry
    assert entry.cget("style") == "Invalid.TEntry"
    assert (
        train.scroll.canvas.winfo_rooty()
        <= entry.winfo_rooty()
        < train.scroll.canvas.winfo_rooty() + train.scroll.canvas.winfo_height()
    )
    snapshot("validation-error")
    validate.master.select(validate)
    app.save_settings_from_forms()
    root.update()
    assert train.master.select() == str(train)
    assert app.settings.path.read_bytes() == before
    train.runtime["epochs"].set(20)
    scenarios.append(
        "invalid numeric input focuses and scrolls to field; failed Save All preserves file"
    )

    train._preview_button.invoke()
    root.update()
    dialogs = [
        widget for widget in root.winfo_children() if isinstance(widget, tk.Toplevel)
    ]
    assert len(dialogs) == 1 and not app.runner.is_running
    dialog = dialogs[0]
    command_text = next(
        widget for widget in descendants(dialog) if isinstance(widget, tk.Text)
    ).get("1.0", "end")
    assert "train" in command_text and "--manifest" in command_text
    snapshot("command-preview", dialog)
    dialog.destroy()
    scenarios.append("command preview creates no child process")

    with patch(
        "ai_model.window.widgets.filedialog.askdirectory", return_value=""
    ) as choose:
        app._path_settings_tab.io_roots["data_root"].browse_button.invoke()
    assert Path(choose.call_args.kwargs["initialdir"]) == Path(
        app.settings.get("common", "data_root")
    )
    scenarios.append("file chooser starts at the resolved current data directory")

    validate.master.select(validate)
    validate._run_button.invoke()
    assert app.task_status.phase == "running" and app._log_visible
    assert train._run_button.instate(["disabled"]) and cleanup._delete_button.instate(
        ["disabled"]
    )
    root.update()
    assert app._log_frame.winfo_ismapped() and app.log_text.winfo_height() >= 30
    snapshot("task-running")
    pump_until(lambda: app.task_status.phase in ("success", "error"))
    assert app.task_status.phase == "success", app.log_text.get("1.0", "end")
    assert "exit_code = 0" in app.log_text.get("1.0", "end")
    assert not train._run_button.instate(["disabled"])
    scenarios.append(
        "actual validate CLI subprocess completes, logs arrive and actions recover"
    )

    app.run_command(
        [sys.executable, "-c", "import time; print('SMOKE_READY'); time.sleep(30)"]
    )
    pump_until(lambda: "\nSMOKE_READY\n" in app.log_text.get("1.0", "end"))
    started = time.monotonic()
    with patch("ai_model.window.tabs.base.messagebox.askyesno", return_value=True):
        validate._stop_button.invoke()
    stop_return_ms = (time.monotonic() - started) * 1000
    pump_until(lambda: app.task_status.phase == "stopped")
    assert not app.runner.is_running and validate._stop_button.instate(["disabled"])
    scenarios.append(
        "Stop returns without waiting for the temporary child; stopped state is explicit"
    )

    app.run_command(
        [sys.executable, "-c", "raise RuntimeError('SMOKE_EXPECTED_ERROR')"]
    )
    pump_until(lambda: app.task_status.phase == "error")
    assert "SMOKE_EXPECTED_ERROR" in app.log_text.get("1.0", "end")
    assert not cleanup._delete_button.instate(["disabled"])
    snapshot("task-error")
    scenarios.append("failed subprocess shows error and restores controls")

    # The production delete function must never run during the smoke check.
    cleanup.set_busy(True)
    with patch("ai_model.window.tabs.manage_artifacts.execute_cleanup") as delete:
        cleanup._execute_cleanup()
        delete.assert_not_called()
    cleanup.set_busy(False)
    scenarios.append("cleanup cannot delete while a task is active")

    app._autoscroll_var.set(False)
    if app._log_visible:
        app.toggle_log()
    train.master.select(train)
    app._append_to_text("PRESERVE_ON_RELOAD\n")
    with patch("ai_model.window.app.messagebox.askyesno", return_value=True):
        app.reload_settings()
    root.update()
    assert not app._log_visible and not app._autoscroll_var.get()
    selected = next(tab for tab in app._tabs if tab.settings_section == "train")
    assert selected.master.select() == str(selected)
    assert "PRESERVE_ON_RELOAD" in app.log_text.get("1.0", "end")
    snapshot("reloaded")
    (output / "session.log").write_text(
        app.log_text.get("1.0", "end"), encoding="utf-8"
    )
    scenarios.append(
        "reload preserves selected page, log content, collapse state and scrolling preference"
    )
    return {
        "scenarios": scenarios,
        "layouts": pages,
        "stop_callback_ms": round(stop_return_ms, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--screenshots", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=PACKAGE_ROOT / "audit/gui_usability/smoke"
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    errors = []
    report = {}
    root = tk.Tk()
    root.withdraw()
    root.report_callback_exception = lambda *error: errors.append(
        "".join(traceback.format_exception(*error))
    )
    app = None
    try:
        with tempfile.TemporaryDirectory(prefix="ai-model-gui-") as temporary:
            app = AiModelApp(root, fixture(Path(temporary)))
            report = exercise(root, app, args.output, args.screenshots)
            assert not errors, errors
            report["passed"] = True
    except Exception:
        report["passed"] = False
        report["failure"] = traceback.format_exc()
        raise
    finally:
        if app is not None:
            app.runner.stop()
            app._destroy_window()
        else:
            root.destroy()
        report["tk_callback_errors"] = errors
        (args.output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(
        json.dumps(
            {
                "passed": True,
                "scenarios": len(report["scenarios"]),
                "layouts": len(report["layouts"]),
                "report": str(args.output / "report.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
