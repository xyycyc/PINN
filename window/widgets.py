"""GUI 通用控件库。

集中放置标签输入、文件/目录选择、下拉框、复选项、Section 容器等
组件，方便各 Tab 复用，保持视觉风格一致。
"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable, Iterable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
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


class MaterialSplitEditor(ttk.Frame):
    """Dynamic per-folder train/validation/test ratio editor."""

    def __init__(
        self,
        master: tk.Misc,
        *,
        source_getter: Callable[[], str],
        source_resolver: Callable[[str], Path],
        initial: Any = None,
    ) -> None:
        super().__init__(master)
        self._source_getter = source_getter
        self._source_resolver = source_resolver
        self._initial = self._normalize_initial(initial)
        self._rows: dict[str, tuple[tk.StringVar, tk.StringVar, tk.StringVar]] = {}

        top = ttk.Frame(self)
        top.pack(fill="x")
        ttk.Button(top, text="扫描/刷新材料文件夹", command=self.refresh_with_message).pack(
            side="left"
        )
        self.status = ttk.Label(
            top,
            text="开启多材料输入后，扫描所选上层目录的直接子文件夹。",
            foreground="#666",
        )
        self.status.pack(side="left", padx=(PADX, 0))
        self.table = ttk.Frame(self)
        self.table.pack(fill="x", pady=(PADY, 0))
        self._render_headers()

    @staticmethod
    def _normalize_initial(value: Any) -> dict[str, tuple[float, float, float]]:
        result: dict[str, tuple[float, float, float]] = {}
        if isinstance(value, dict):
            items = value.items()
        elif isinstance(value, list):
            items = []
            for item in value:
                if not isinstance(item, str):
                    continue
                key, separator, ratios = item.partition("=")
                if separator:
                    items.append((key, ratios.split(",")))
        else:
            items = []
        for key, raw in items:
            try:
                if isinstance(raw, dict):
                    values = (raw["train"], raw["validation"], raw["test"])
                else:
                    values = tuple(raw)
                if len(values) == 3:
                    result[str(key)] = tuple(float(item) for item in values)  # type: ignore[assignment]
            except (KeyError, TypeError, ValueError):
                continue
        return result

    def _render_headers(self) -> None:
        for column, text in enumerate(("材料文件夹/路由字段", "训练", "验证", "测试")):
            ttk.Label(self.table, text=text).grid(
                row=0,
                column=column,
                sticky="w",
                padx=(0, PADX),
                pady=(0, PADY),
            )
        self.table.columnconfigure(0, weight=1)

    def refresh(self) -> None:
        from ..data_process import discover_material_roots

        source = self._source_getter().strip()
        if not source:
            raise ValueError("请先选择多材料上层目录。")
        roots = discover_material_roots(self._source_resolver(source))
        previous = self.as_dict(validate=False)
        for child in self.table.winfo_children():
            child.destroy()
        self._render_headers()
        self._rows.clear()
        for row_index, material_key in enumerate(roots, start=1):
            values = previous.get(
                material_key,
                self._initial.get(material_key, (0.7, 0.1, 0.2)),
            )
            variables = tuple(tk.StringVar(value=str(value)) for value in values)
            self._rows[material_key] = variables  # type: ignore[assignment]
            ttk.Label(self.table, text=material_key).grid(
                row=row_index,
                column=0,
                sticky="w",
                padx=(0, PADX),
                pady=2,
            )
            for column, variable in enumerate(variables, start=1):
                ttk.Entry(self.table, textvariable=variable, width=10).grid(
                    row=row_index,
                    column=column,
                    sticky="w",
                    padx=(0, PADX),
                    pady=2,
                )
        self.status.configure(text=f"已识别 {len(self._rows)} 种样本材料。")

    def refresh_with_message(self) -> None:
        try:
            self.refresh()
        except (OSError, ValueError) as exc:
            messagebox.showerror("扫描材料目录失败", str(exc), parent=self.winfo_toplevel())

    def as_dict(self, *, validate: bool = True) -> dict[str, tuple[float, float, float]]:
        result: dict[str, tuple[float, float, float]] = {}
        for material_key, variables in self._rows.items():
            try:
                values = tuple(float(variable.get().strip()) for variable in variables)
            except ValueError as exc:
                if not validate:
                    continue
                raise ValueError(f"材料 {material_key} 的划分比例不是合法数字。") from exc
            if validate:
                if not all(0.0 <= value < 1.0 for value in values):
                    raise ValueError(f"材料 {material_key} 的各划分比例必须位于 [0,1)。")
                if values[0] <= 0.0 or values[2] <= 0.0:
                    raise ValueError(f"材料 {material_key} 的训练和测试比例必须大于 0。")
                if abs(sum(values) - 1.0) > 1e-9:
                    raise ValueError(f"材料 {material_key} 的训练/验证/测试比例之和必须为 1。")
            result[material_key] = values  # type: ignore[assignment]
        return result

    def specs(self) -> list[str]:
        if not self._rows:
            self.refresh()
        return [
            f"{key}={values[0]:g},{values[1]:g},{values[2]:g}"
            for key, values in self.as_dict().items()
        ]

    def to_settings(self) -> dict[str, dict[str, float]]:
        current = self.as_dict(validate=False) or dict(self._initial)
        return {
            key: {
                "train": values[0],
                "validation": values[1],
                "test": values[2],
            }
            for key, values in current.items()
        }


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
