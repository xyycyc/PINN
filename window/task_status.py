"""Visible task state and elapsed time, updated only by the Tk main loop."""

from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk


class TaskStatus(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.phase = "idle"
        self._started_at: float | None = None
        self._elapsed = -1
        self.badge = ttk.Label(self, text="就绪", style="Idle.Badge.TLabel")
        self.badge.pack(side="left", padx=(0, 12))
        self.text = tk.StringVar(value="选择功能页，准备开始")
        ttk.Label(self, textvariable=self.text, style="Muted.TLabel").pack(side="left")
        self.elapsed = tk.StringVar(value="")
        ttk.Label(self, textvariable=self.elapsed, style="Hint.TLabel").pack(
            side="left", padx=(14, 0)
        )
        self.progress = ttk.Progressbar(
            self, mode="indeterminate", length=92, style="Task.Horizontal.TProgressbar"
        )

    def start(self, title: str) -> None:
        self.phase = "running"
        self._started_at = time.monotonic()
        self._elapsed = -1
        self.badge.configure(text="运行中", style="Running.Badge.TLabel")
        self.text.set(title)
        self.progress.pack(side="left", padx=(16, 0))
        self.progress.start(14)
        self.tick()

    def stopping(self) -> None:
        self.phase = "stopping"
        self.badge.configure(text="正在停止", style="Running.Badge.TLabel")

    def finish(self, exit_code: int | None) -> None:
        self.tick()
        self._started_at = None
        stopped = self.phase == "stopping"
        self.phase = "stopped" if stopped else "success" if exit_code == 0 else "error"
        self.progress.stop()
        self.progress.pack_forget()
        label, style = (
            ("已停止", "Idle")
            if stopped
            else ("已完成", "Success")
            if exit_code == 0
            else ("未完成", "Error")
        )
        self.badge.configure(text=label, style=f"{style}.Badge.TLabel")
        if stopped:
            self.text.set("任务已停止")
        elif exit_code != 0:
            self.text.set(
                "任务已停止"
                if exit_code is None
                else f"退出码 {exit_code} · 请查看日志"
            )

    def tick(self) -> None:
        if self._started_at is None:
            return
        elapsed = int(time.monotonic() - self._started_at)
        if elapsed != self._elapsed:
            self._elapsed = elapsed
            minutes, seconds = divmod(elapsed, 60)
            self.elapsed.set(f"{minutes:02d}:{seconds:02d}")
