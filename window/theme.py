"""Shared palette and ttk styles for the desktop workspace."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk

COLORS = {
    "workspace": "#f2f5f9",
    "surface": "#ffffff",
    "ink": "#172b45",
    "muted": "#63758b",
    "border": "#dbe3ed",
    "accent": "#087f8c",
    "accent_hover": "#066774",
    "accent_soft": "#e8f5f5",
    "disabled": "#94a3b8",
    "danger": "#b83c48",
    "danger_soft": "#fff0f1",
    "success": "#1f795d",
    "success_soft": "#eaf6f0",
    "log": "#f8fafc",
}


def configure_styles(root: tk.Misc) -> None:
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    base = ("Microsoft YaHei UI", 10)
    title = ("Microsoft YaHei UI", 12, "bold")
    root.configure(background=COLORS["workspace"])
    root.option_add("*Menu.font", base)
    root.option_add("*TCombobox*Listbox.background", COLORS["surface"])
    root.option_add("*TCombobox*Listbox.foreground", COLORS["ink"])
    root.option_add("*TCombobox*Listbox.selectBackground", COLORS["accent_soft"])
    root.option_add("*TCombobox*Listbox.selectForeground", COLORS["ink"])
    style.configure(
        ".", font=base, background=COLORS["surface"], foreground=COLORS["ink"]
    )
    style.configure("Workspace.TFrame", background=COLORS["workspace"])
    style.configure("TLabel", padding=0)
    style.configure("Muted.TLabel", foreground=COLORS["muted"])
    style.configure("Hint.TLabel", font=(base[0], 9), foreground=COLORS["muted"])
    style.configure("PageTitle.TLabel", font=(base[0], 16, "bold"))
    style.configure("AppTitle.TLabel", font=(base[0], 18, "bold"))
    style.configure(
        "Brand.TLabel",
        background=COLORS["accent"],
        foreground="white",
        font=(base[0], 16, "bold"),
        padding=(12, 8),
    )
    style.configure("Title.TLabel", font=title)
    style.configure(
        "Section.TLabelframe",
        padding=(14, 10),
        borderwidth=1,
        relief="solid",
        bordercolor=COLORS["border"],
    )
    style.configure(
        "Section.TLabelframe.Label",
        font=(base[0], 11, "bold"),
        foreground=COLORS["ink"],
    )
    style.configure(
        "TEntry",
        padding=(8, 6),
        fieldbackground="white",
        bordercolor=COLORS["border"],
        lightcolor=COLORS["border"],
        darkcolor=COLORS["border"],
    )
    style.map("TEntry", bordercolor=[("focus", COLORS["accent"])])
    style.configure(
        "Invalid.TEntry",
        bordercolor=COLORS["danger"],
        fieldbackground=COLORS["danger_soft"],
    )
    style.map("Invalid.TEntry", bordercolor=[("focus", COLORS["danger"])])
    style.configure(
        "TCombobox", padding=(8, 6), arrowsize=14, bordercolor=COLORS["border"]
    )
    style.map(
        "TCombobox",
        fieldbackground=[("readonly", "white"), ("disabled", COLORS["workspace"])],
        foreground=[("disabled", COLORS["disabled"])],
        bordercolor=[("focus", COLORS["accent"])],
        selectbackground=[("readonly", "white")],
        selectforeground=[("readonly", COLORS["ink"])],
    )
    style.configure(
        "TCheckbutton",
        padding=(0, 3),
        indicatorbackground="white",
        indicatorcolor=COLORS["accent"],
    )
    style.map("TCheckbutton", background=[("active", "white")])
    style.configure(
        "TButton",
        padding=(12, 7),
        borderwidth=1,
        relief="flat",
        background="white",
        bordercolor=COLORS["border"],
        focusthickness=1,
        focuscolor=COLORS["accent"],
    )
    style.map(
        "TButton",
        background=[("active", COLORS["workspace"])],
        foreground=[("disabled", COLORS["disabled"])],
    )
    style.configure(
        "Primary.TButton",
        padding=(20, 8),
        background=COLORS["accent"],
        foreground="white",
        bordercolor=COLORS["accent"],
        font=(base[0], 10, "bold"),
    )
    style.map(
        "Primary.TButton",
        background=[("disabled", "#c7d8dc"), ("active", COLORS["accent_hover"])],
        foreground=[("disabled", "#6b858b"), ("!disabled", "white")],
        bordercolor=[("disabled", "#c7d8dc"), ("active", COLORS["accent_hover"])],
    )
    style.configure(
        "Danger.TButton",
        padding=(14, 8),
        foreground=COLORS["danger"],
        bordercolor="#edcdd1",
    )
    style.map("Danger.TButton", background=[("active", COLORS["danger_soft"])])
    style.configure(
        "Quiet.TButton", padding=(10, 5), borderwidth=0, foreground=COLORS["muted"]
    )
    style.configure(
        "TNotebook",
        background=COLORS["workspace"],
        borderwidth=0,
        tabmargins=(0, 0, 0, 0),
    )
    style.layout(
        "TNotebook.Tab",
        [
            (
                "Notebook.padding",
                {
                    "sticky": "nswe",
                    "children": [("Notebook.label", {"sticky": "nswe"})],
                },
            )
        ],
    )
    style.configure(
        "TNotebook.Tab",
        padding=(17, 10),
        background=COLORS["workspace"],
        foreground=COLORS["muted"],
        borderwidth=0,
    )
    style.map(
        "TNotebook.Tab",
        padding=[("selected", (17, 10))],
        background=[("selected", "white"), ("active", "#e6edf4")],
        foreground=[("selected", COLORS["accent"]), ("active", COLORS["ink"])],
    )
    style.configure("TPanedwindow", background=COLORS["workspace"], sashwidth=6)
    style.configure(
        "TScrollbar",
        background="#c7d3e0",
        troughcolor=COLORS["workspace"],
        borderwidth=0,
        arrowsize=12,
    )
    style.configure("TSeparator", background=COLORS["border"])
    style.configure(
        "Status.TLabel",
        padding=(16, 6),
        foreground=COLORS["muted"],
        background=COLORS["workspace"],
        font=(base[0], 9),
    )
    style.configure("Error.TLabel", foreground=COLORS["danger"], font=(base[0], 9))
    for name, foreground, background in (
        ("Idle", COLORS["muted"], COLORS["workspace"]),
        ("Running", COLORS["accent"], COLORS["accent_soft"]),
        ("Success", COLORS["success"], COLORS["success_soft"]),
        ("Error", COLORS["danger"], COLORS["danger_soft"]),
    ):
        style.configure(
            f"{name}.Badge.TLabel",
            foreground=foreground,
            background=background,
            padding=(9, 4),
            font=(base[0], 9, "bold"),
        )
    style.configure(
        "Task.Horizontal.TProgressbar",
        background=COLORS["accent"],
        troughcolor=COLORS["workspace"],
        borderwidth=0,
        thickness=3,
    )
    root._mono_font = ("Consolas", 10)
    root._base_font = base
    root._title_font = title
