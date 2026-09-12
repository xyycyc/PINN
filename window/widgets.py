"""Reusable, responsive desktop form controls."""

from __future__ import annotations

import math
import tkinter as tk
from collections.abc import Callable, Iterable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from ..paths import PACKAGE_ROOT, resolve_project_path
from .theme import COLORS, configure_styles as configure_styles

PADX = 8
PADY = 5
LABEL_WIDTH = 22


def wrapped_label(master: tk.Misc, text: str, **kwargs: Any) -> ttk.Label:
    """Wrap explanatory text to its allocated width, including after a resize."""
    label = ttk.Label(master, text=text, wraplength=760, justify="left", **kwargs)
    label.bind(
        "<Configure>", lambda event: label.configure(wraplength=max(80, event.width))
    )
    return label


class FormValidationError(ValueError):
    """A validation error that can lead the user back to the relevant control."""

    def __init__(self, message: str, widget: tk.Misc) -> None:
        super().__init__(message)
        self.widget = widget


class Section(ttk.LabelFrame):
    def __init__(self, master: tk.Misc, title: str, **kwargs: Any) -> None:
        super().__init__(master, text=title, style="Section.TLabelframe", **kwargs)


class FormRow(ttk.Frame):
    """Aligned label, input area and optional explanation on a separate line."""

    def __init__(self, master: tk.Misc, label: str) -> None:
        super().__init__(master)
        self.label = ttk.Label(
            self, text=label, width=LABEL_WIDTH, anchor="w", wraplength=190
        )
        self.label.grid(row=0, column=0, sticky="nw", padx=(0, 14), pady=(6, 0))
        self.columnconfigure(1, weight=1)
        self.body = ttk.Frame(self)
        self.body.grid(row=0, column=1, sticky="ew")
        self.controls = ttk.Frame(self.body)
        self.controls.pack(fill="x")

    def add_hint(self, hint: str | None) -> None:
        if hint:
            self.hint = wrapped_label(self.body, hint, style="Hint.TLabel")
            self.hint.pack(fill="x", pady=(4, 0))

    def _watch_entry(self) -> None:
        self.var.trace_add("write", lambda *_: self.entry.configure(style="TEntry"))

    def invalid(self, message: str) -> FormValidationError:
        self.entry.configure(style="Invalid.TEntry")
        return FormValidationError(message, self.entry)


class LabeledEntry(FormRow):
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
        self.entry = ttk.Entry(self.controls, textvariable=self.var, width=width)
        self.entry.pack(side="left", fill="x", expand=True)
        self.add_hint(hint)
        self._watch_entry()

    def get(self) -> str:
        return self.var.get().strip()

    def set(self, value: str) -> None:
        self.var.set(str(value))


class LabeledNumber(FormRow):
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
        self.entry = ttk.Entry(self.controls, textvariable=self.var, width=width)
        self.entry.pack(side="left")
        self.add_hint(hint)
        self._watch_entry()

    def get(self) -> float | int | None:
        text = self.var.get().strip()
        if not text:
            return None
        try:
            value = float(text) if self.is_float else int(text)
            if self.is_float and not math.isfinite(value):
                raise ValueError("non-finite number")
            return value
        except ValueError as exc:
            raise self.invalid(
                f"{self.label.cget('text')}：请输入有效的{'小数或整数' if self.is_float else '整数'}（当前为 {text}）。"
            ) from exc

    def set(self, value: float | int) -> None:
        self.var.set(str(value))


class LabeledCombobox(FormRow):
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
        choices = list(values)
        self.var = tk.StringVar(value=default or (choices[0] if choices else ""))
        self.combo = ttk.Combobox(
            self.controls,
            textvariable=self.var,
            values=choices,
            state="readonly" if readonly else "normal",
            width=width,
        )
        self.combo.pack(side="left", fill="x", expand=True)
        self.add_hint(hint)

    def get(self) -> str:
        return self.var.get().strip()

    def set(self, value: str) -> None:
        self.var.set(str(value))


class LabeledCheck(FormRow):
    def __init__(
        self, master: tk.Misc, label: str, text: str, default: bool = False
    ) -> None:
        super().__init__(master, label)
        self.var = tk.BooleanVar(value=bool(default))
        self.check = ttk.Checkbutton(self.controls, variable=self.var)
        self.check.pack(side="left", anchor="n", pady=(3, 0))
        self.caption = wrapped_label(self.controls, text)
        self.caption.pack(side="left", fill="x", expand=True, pady=6)
        self.caption.bind("<Button-1>", self._toggle)

    def _toggle(self, _event: tk.Event) -> str:
        if not self.check.instate(["disabled"]):
            self.check.focus_set()
            self.check.invoke()
        return "break"

    def get(self) -> bool:
        return bool(self.var.get())

    def set(self, value: bool) -> None:
        self.var.set(bool(value))


class FileEntry(FormRow):
    """Path input whose browser follows the same root as command execution."""

    def __init__(
        self,
        master: tk.Misc,
        label: str,
        default: str = "",
        width: int = 48,
        filetypes: Iterable[tuple[str, str]] | None = None,
        save: bool = False,
        directory: bool = False,
        resolver: Callable[[str], Path] | None = None,
    ) -> None:
        super().__init__(master, label)
        self.var = tk.StringVar(value=default)
        self.entry = ttk.Entry(self.controls, textvariable=self.var, width=width)
        self.entry.pack(side="left", fill="x", expand=True)
        self._filetypes = list(filetypes) if filetypes else [("所有文件", "*.*")]
        self._save = save
        self._directory = directory
        self._resolver = resolver
        self.browse_button = ttk.Button(
            self.controls, text="浏览…", command=self._browse, width=7
        )
        self.browse_button.pack(side="left", padx=(PADX, 0))
        self._watch_entry()

    def resolve(self, value: str) -> Path:
        if self._resolver is not None:
            return self._resolver(value)
        ancestor = self.master
        while ancestor is not None:
            if hasattr(ancestor, "repo_root"):
                return resolve_project_path(value, repo_root=ancestor.repo_root)
            ancestor = getattr(ancestor, "master", None)
        return resolve_project_path(value, repo_root=PACKAGE_ROOT)

    def _browse(self) -> None:
        try:
            initial = self.resolve(self.get() or ".")
            initial_file = "" if self._directory or initial.is_dir() else initial.name
            initial_dir = initial if not initial_file else initial.parent
            while not initial_dir.is_dir() and initial_dir != initial_dir.parent:
                initial_dir = initial_dir.parent
            options = {"parent": self.winfo_toplevel(), "initialdir": str(initial_dir)}
            if self._directory:
                picked = filedialog.askdirectory(
                    title=f"选择{self.label.cget('text')}", **options
                )
            else:
                picker = (
                    filedialog.asksaveasfilename
                    if self._save
                    else filedialog.askopenfilename
                )
                picked = picker(
                    title=f"{'保存' if self._save else '选择'}{self.label.cget('text')}",
                    initialfile=initial_file,
                    filetypes=self._filetypes,
                    **options,
                )
        except (OSError, ValueError) as exc:
            messagebox.showerror("无法打开路径", str(exc), parent=self.winfo_toplevel())
            return
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
        ttk.Button(
            top, text="扫描/刷新材料文件夹", command=self.refresh_with_message
        ).pack(side="left")
        self.status = ttk.Label(
            top,
            text="测试比例可以为 0；可由独立波形测试集替代。",
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
            messagebox.showerror(
                "扫描材料目录失败", str(exc), parent=self.winfo_toplevel()
            )

    def as_dict(
        self, *, validate: bool = True
    ) -> dict[str, tuple[float, float, float]]:
        result: dict[str, tuple[float, float, float]] = {}
        for material_key, variables in self._rows.items():
            try:
                values = tuple(float(variable.get().strip()) for variable in variables)
            except ValueError as exc:
                if not validate:
                    continue
                raise ValueError(
                    f"材料 {material_key} 的划分比例不是合法数字。"
                ) from exc
            if validate:
                if not all(0.0 <= value <= 1.0 for value in values):
                    raise ValueError(
                        f"材料 {material_key} 的各划分比例必须位于 [0,1]。"
                    )
                if values[0] <= 0.0:
                    raise ValueError(f"case 材料 {material_key} 的训练比例必须大于 0。")
                if abs(sum(values) - 1.0) > 1e-9:
                    raise ValueError(
                        f"材料 {material_key} 的训练/验证/测试比例之和必须为 1。"
                    )
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
        self.canvas = tk.Canvas(
            self,
            highlightthickness=0,
            background=COLORS["surface"],
            yscrollincrement=24,
        )
        self.vbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.vbar.set)
        self.canvas.pack(side="left", fill="both", expand=True)
        self.vbar.pack(side="right", fill="y")

        self.inner = ttk.Frame(self.canvas)
        self._window_id = self.canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )

        self.inner.bind("<Configure>", self._on_inner_configure)
        self.canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel()
        self._focus_binding = self._wheel_owner.bind(
            "<FocusIn>", self._reveal_focused, add="+"
        )

    def _on_inner_configure(self, _event: tk.Event) -> None:
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))

    def _on_canvas_configure(self, event: tk.Event) -> None:
        self.canvas.itemconfigure(self._window_id, width=event.width)

    def _reveal_focused(self, event: tk.Event) -> None:
        ancestor = event.widget
        while ancestor is not None and ancestor is not self.inner:
            ancestor = getattr(ancestor, "master", None)
        if ancestor is self.inner:
            self.reveal(event.widget)

    def reveal(self, widget: tk.Misc) -> None:
        """Keep keyboard focus and validation errors inside the visible viewport."""
        self.update_idletasks()
        top = widget.winfo_rooty() - self.inner.winfo_rooty()
        height = self.canvas.winfo_height()
        current = self.canvas.canvasy(0)
        if top < current:
            target = top - 12
        elif top + widget.winfo_height() > current + height:
            target = top + widget.winfo_height() - height + 12
        else:
            return
        content_height = max(self.inner.winfo_height(), 1)
        self.canvas.yview_moveto(max(0, target) / content_height)

    def _bind_mousewheel(self) -> None:
        self._wheel_owner = self.winfo_toplevel()
        self._wheel_bindings = {
            sequence: self._wheel_owner.bind(sequence, self._on_mousewheel, add="+")
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>")
        }
        self.bind("<Destroy>", self._unbind_mousewheel, add="+")

    def _on_mousewheel(self, event: tk.Event) -> str | None:
        widget = event.widget
        while widget is not None:
            if widget is self:
                break
            # Widgets with their own scrolling keep their native behavior.
            if isinstance(widget, (tk.Text, tk.Listbox, ttk.Treeview, ttk.Combobox)):
                return None
            widget = getattr(widget, "master", None)
        if widget is None:
            return None
        number = getattr(event, "num", None)
        delta = getattr(event, "delta", 0)
        if number in (4, 5):
            units = -1 if number == 4 else 1
        elif delta:
            units = -max(1, abs(int(delta)) // 120) * (1 if delta > 0 else -1)
        else:
            return None
        if self.canvas.yview() == (0.0, 1.0):
            return None
        self.canvas.yview_scroll(units, "units")
        return "break"

    def _unbind_mousewheel(self, event: tk.Event) -> None:
        if event.widget is self:
            if self._focus_binding:
                self._wheel_owner.unbind("<FocusIn>", self._focus_binding)
            for sequence, binding in self._wheel_bindings.items():
                if binding:
                    self._wheel_owner.unbind(sequence, binding)


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
