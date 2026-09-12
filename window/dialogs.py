"""Small, non-destructive dialogs shared by command pages."""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import ttk

from .runner import format_command
from .theme import COLORS
from .widgets import wrapped_label


def show_command_preview(
    parent: tk.Misc, title: str, command: list[str], cwd: Path
) -> tk.Toplevel:
    dialog = tk.Toplevel(parent)
    dialog.title(f"预览命令 · {title}")
    dialog.transient(parent)
    dialog.geometry("880x370")
    dialog.minsize(600, 300)
    body = ttk.Frame(dialog, padding=22)
    body.pack(fill="both", expand=True)
    ttk.Label(body, text="执行前预览", style="PageTitle.TLabel").pack(anchor="w")
    wrapped_label(body, f"工作目录：{cwd}", style="Muted.TLabel").pack(
        fill="x", pady=(8, 12)
    )
    frame = ttk.Frame(body)
    frame.pack(fill="both", expand=True)
    preview = tk.Text(
        frame,
        wrap="word",
        height=7,
        font=("Consolas", 10),
        relief="flat",
        background=COLORS["log"],
        foreground=COLORS["ink"],
        padx=12,
        pady=10,
    )
    scrollbar = ttk.Scrollbar(frame, command=preview.yview)
    scrollbar.pack(side="right", fill="y")
    preview.configure(yscrollcommand=scrollbar.set)
    preview.pack(fill="both", expand=True)
    text = format_command(command)
    preview.insert("1.0", text)
    preview.configure(state="disabled")
    actions = ttk.Frame(body)
    actions.pack(fill="x", pady=(16, 0))
    ttk.Label(actions, text="预览不会启动任务。", style="Hint.TLabel").pack(side="left")

    def copy() -> None:
        dialog.clipboard_clear()
        dialog.clipboard_append(text)
        copy_button.configure(text="已复制")

    ttk.Button(actions, text="关闭", command=dialog.destroy).pack(side="right")
    copy_button = ttk.Button(
        actions, text="复制命令", command=copy, style="Primary.TButton"
    )
    copy_button.pack(side="right", padx=(0, 10))
    dialog.bind("<Escape>", lambda _event: dialog.destroy())
    return dialog
