"""GUI 通用控件库。

集中放置标签输入、文件/目录选择、下拉框、复选项、Section 容器等
组件，方便各 Tab 复用，保持视觉风格一致。
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Iterable
from pathlib import Path
from tkinter import filedialog, ttk
from typing import Any


PADX = 6
PADY = 4
LABEL_WIDTH = 22


def configure_styles(root: tk.Misc) -> None:
    """设置中文字体与 ttk 主题，让所有窗口风格一致。"""

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass

    base_font = ("Microsoft YaHei UI", 10)
    title_font = ("Microsoft YaHei UI", 11, "bold")
    monospace = ("Consolas", 10)

    root.option_add("*Font", base_font)
    style.configure(".", font=base_font)
    style.configure("Title.TLabel", font=title_font)
    style.configure("Section.TLabelframe", padding=8)
    style.configure("Section.TLabelframe.Label", font=title_font)
    style.configure("Primary.TButton", padding=(14, 6))
    style.configure("Danger.TButton", padding=(14, 6), foreground="#b30000")
    style.configure("Status.TLabel", padding=(8, 4), background="#f0f0f0")

    # 把字体常量挂到 root 上，方便 Text 控件复用。
    root._mono_font = monospace  # type: ignore[attr-defined]
    root._base_font = base_font  # type: ignore[attr-defined]
    root._title_font = title_font  # type: ignore[attr-defined]


class Section(ttk.LabelFrame):
    """带统一 padding 的分组容器。"""

    def __init__(self, master: tk.Misc, title: str, **kwargs: Any) -> None:
        super().__init__(master, text=title, style="Section.TLabelframe", **kwargs)


class FormRow(ttk.Frame):
    """单行表单：左侧标签 + 右侧控件。"""

    def __init__(self, master: tk.Misc, label: str) -> None:
        super().__init__(master)
        self.label = ttk.Label(self, text=label, width=LABEL_WIDTH, anchor="w")
        self.label.pack(side="left", padx=(0, PADX))
        self.body = ttk.Frame(self)
        self.body.pack(side="left", fill="x", expand=True)


class LabeledEntry(FormRow):
    """标签 + 文本输入。"""

    def __init__(
        self,
        master: tk.Misc,
        label: str,
        default: str = "",
        width: int = 32,
        hint: str | None = None,
    ) -> None:
        super().__init__(master, label)
        self.var = tk.StringVar(value=default)
        self.entry = ttk.Entry(self.body, textvariable=self.var, width=width)
        self.entry.pack(side="left", fill="x", expand=True)
        if hint:
            ttk.Label(self.body, text=hint, foreground="#666").pack(side="left", padx=(PADX, 0))

    def get(self) -> str:
        return self.var.get().strip()

    def set(self, value: str) -> None:
        self.var.set(str(value))


class LabeledNumber(FormRow):
    """整数 / 浮点数输入。"""

    def __init__(
        self,
        master: tk.Misc,
        label: str,
        default: float | int,
        is_float: bool = False,
        width: int = 12,
        hint: str | None = None,
    ) -> None:
        super().__init__(master, label)
        self.is_float = is_float
        self.var = tk.StringVar(value=str(default))
        self.entry = ttk.Entry(self.body, textvariable=self.var, width=width)
        self.entry.pack(side="left")
        if hint:
            ttk.Label(self.body, text=hint, foreground="#666").pack(side="left", padx=(PADX, 0))

    def get(self) -> float | int | None:
        text = self.var.get().strip()
        if not text:
            return None
        try:
            return float(text) if self.is_float else int(text)
        except ValueError as exc:
            raise ValueError(f"{self.label.cget('text')} 不是合法数字: {text}") from exc

    def set(self, value: float | int) -> None:
        self.var.set(str(value))


class LabeledCombobox(FormRow):
    """标签 + 下拉框（默认只读）。"""

    def __init__(
        self,
        master: tk.Misc,
        label: str,
        values: Iterable[str],
        default: str | None = None,
        readonly: bool = True,
        width: int = 24,
        hint: str | None = None,
    ) -> None:
        super().__init__(master, label)
        values_list = list(values)
        self.var = tk.StringVar(value=default or (values_list[0] if values_list else ""))
        state = "readonly" if readonly else "normal"
        self.combo = ttk.Combobox(
            self.body,
            textvariable=self.var,
            values=values_list,
            state=state,
            width=width,
        )
        self.combo.pack(side="left")
        if hint:
            ttk.Label(self.body, text=hint, foreground="#666").pack(side="left", padx=(PADX, 0))

    def get(self) -> str:
        return self.var.get().strip()

    def set(self, value: str) -> None:
        self.var.set(str(value))


class LabeledCheck(FormRow):
    """标签 + 复选框（带说明文字）。"""

    def __init__(
        self,
        master: tk.Misc,
        label: str,
        text: str,
        default: bool = False,
    ) -> None:
        super().__init__(master, label)
        self.var = tk.BooleanVar(value=bool(default))
        self.check = ttk.Checkbutton(self.body, text=text, variable=self.var)
        self.check.pack(side="left")

    def get(self) -> bool:
        return bool(self.var.get())

    def set(self, value: bool) -> None:
        self.var.set(bool(value))


class FileEntry(FormRow):
    """标签 + 文件路径输入 + 浏览按钮。"""

    def __init__(
        self,
        master: tk.Misc,
        label: str,
        default: str = "",
        width: int = 48,
        filetypes: Iterable[tuple[str, str]] | None = None,
        save: bool = False,
        directory: bool = False,
    ) -> None:
        super().__init__(master, label)
        self.var = tk.StringVar(value=default)
        self.entry = ttk.Entry(self.body, textvariable=self.var, width=width)
        self.entry.pack(side="left", fill="x", expand=True)
        self._filetypes = list(filetypes) if filetypes else [("所有文件", "*.*")]
        self._save = save
        self._directory = directory
        ttk.Button(self.body, text="浏览…", command=self._browse, width=8).pack(
            side="left", padx=(PADX, 0)
        )

    def _browse(self) -> None:
        initial = self.var.get().strip() or "."
        try:
            initial_path = Path(initial).expanduser()
        except Exception:
            initial_path = Path(".")
        initial_file = ""
        if initial_path.exists():
            if initial_path.is_dir():
                initial_dir = str(initial_path)
            else:
                initial_dir = str(initial_path.parent)
                initial_file = initial_path.name
        else:
            if initial_path.suffix:
                initial_dir = str(initial_path.parent) if str(initial_path.parent) else "."
                initial_file = initial_path.name
            else:
                initial_dir = str(initial_path)
        if self._directory:
            picked = filedialog.askdirectory(title="选择目录", initialdir=initial_dir)
        elif self._save:
            picked = filedialog.asksaveasfilename(
                title="保存为",
                initialdir=initial_dir,
                initialfile=initial_file,
                filetypes=self._filetypes,
            )
        else:
            picked = filedialog.askopenfilename(
                title="选择文件",
                initialdir=initial_dir,
                initialfile=initial_file,
                filetypes=self._filetypes,
            )
        if picked:
            self.var.set(picked)

    def get(self) -> str:
        return self.var.get().strip()

    def set(self, value: str) -> None:
        self.var.set(str(value))


class ScrollableFrame(ttk.Frame):
    """带垂直滚动条的容器，里面通过 ``inner`` 放置实际控件。"""

    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master)
        self.canvas = tk.Canvas(self, highlightthickness=0)
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        self.inner = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window((0, 0), window=self.inner, anchor="nw")

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel(self.canvas)

    def _on_inner_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window_id, width=event.width)

    def _bind_mousewheel(self, widget: tk.Misc) -> None:
        def _on_wheel(event: tk.Event) -> None:
            delta = -1 if getattr(event, "delta", 0) > 0 else 1
            self.canvas.yview_scroll(delta, "units")

        widget.bind("<Enter>", lambda _e: widget.bind_all("<MouseWheel>", _on_wheel))
        widget.bind("<Leave>", lambda _e: widget.unbind_all("<MouseWheel>"))


def add_button_row(
    master: tk.Misc,
    buttons: list[tuple[str, Callable[[], None], str]],
) -> ttk.Frame:
    """构造一行按钮，``buttons`` 元素为 (文本, 回调, 样式名)。"""

    row = ttk.Frame(master)
    for text, command, style in buttons:
        ttk.Button(row, text=text, command=command, style=style or "TButton").pack(
            side="left", padx=(0, PADX)
        )
    return row
