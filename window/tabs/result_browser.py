"""结果浏览：在窗口内浏览结果目录下的图片与表格、文本类产物。"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from ..settings import Settings
from ..widgets import PADX, PADY


PREVIEW_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}
PREVIEW_TEXT_EXTS = {".json", ".csv", ".txt", ".md", ".log"}
MAX_TEXT_BYTES = 256 * 1024  # 单文件预览最多读 256KB，避免卡死


class ResultBrowserTab(ttk.Frame):
    """左侧目录树 + 右侧预览的产物浏览器。

    - PNG / JPG / GIF 等用 Pillow 缩放后预览；
    - JSON / CSV / TXT 直接以文本展示；
    - 其他类型仅展示元信息与 "用系统打开" 按钮。
    """

    title = "结果浏览"
    settings_section = "result_browser"
    description = (
        "浏览结果目录下的产物：左侧为目录树，选中图片或表格、文本类文件可在右侧预览。"
    )

    def __init__(
        self,
        master: tk.Misc,
        run_callback: Callable[[list[str]], None],
        stop_callback: Callable[[], None],
        *,
        repo_root: Path,
        settings: Settings | None = None,
    ) -> None:
        super().__init__(master)
        self.repo_root = Path(repo_root)
        self.settings = settings or Settings()
        section_data = self.settings.section(self.settings_section)
        common = self.settings.section("common")
        self._image_ref: object | None = None  # 防止 PhotoImage 被 GC

        ttk.Label(
            self,
            text=self.description,
            wraplength=900,
            foreground="#444",
        ).pack(anchor="w", padx=PADX * 2, pady=(PADY * 2, 0))

        toolbar = ttk.Frame(self)
        toolbar.pack(fill="x", padx=PADX, pady=PADY)

        ttk.Label(toolbar, text="结果根目录：").pack(side="left")
        default_root_str = str(
            section_data.get("root_dir") or common.get("result_root", "result")
        )
        # 相对路径解释为相对 ai_model 项目根，便于 settings.json 里写相对路径
        default_root = Path(default_root_str)
        if not default_root.is_absolute():
            default_root = self.repo_root / default_root
        self._root_var = tk.StringVar(value=str(default_root))
        ttk.Entry(toolbar, textvariable=self._root_var, width=72).pack(
            side="left", fill="x", expand=True, padx=(0, PADX)
        )
        ttk.Button(toolbar, text="浏览…", command=self._browse_root).pack(side="left")
        ttk.Button(toolbar, text="刷新", command=self.refresh).pack(side="left", padx=(PADX, 0))
        ttk.Button(toolbar, text="在系统中打开", command=self._open_in_system).pack(
            side="left", padx=(PADX, 0)
        )

        body = ttk.Panedwindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=PADX, pady=PADY)

        # ------- 左侧目录树 -------
        tree_frame = ttk.Frame(body)
        body.add(tree_frame, weight=1)
        self.tree = ttk.Treeview(tree_frame, show="tree")
        self.tree.pack(side="left", fill="both", expand=True)
        tree_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side="right", fill="y")
        self.tree.bind("<<TreeviewOpen>>", self._on_open)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        # ------- 右侧预览 -------
        preview_frame = ttk.Frame(body)
        body.add(preview_frame, weight=3)

        self.info_var = tk.StringVar(value="（请从左侧选择文件）")
        ttk.Label(preview_frame, textvariable=self.info_var, foreground="#444").pack(
            anchor="w", padx=PADX, pady=(0, PADY)
        )

        self.preview_container = ttk.Frame(preview_frame)
        self.preview_container.pack(fill="both", expand=True)
        self._build_text_preview()
        self._build_image_preview()

        self.refresh()

    # ------------------------------------------------------------------
    # 目录树构建
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        root_path = Path(self._root_var.get()).expanduser()
        if not root_path.exists():
            self.info_var.set(f"目录不存在: {root_path}")
            return
        node = self.tree.insert(
            "", "end", text=root_path.name or str(root_path),
            values=(str(root_path),), open=True,
        )
        self._populate_node(node, root_path)
        self.info_var.set(f"已加载: {root_path}")

    def _populate_node(self, node: str, path: Path) -> None:
        for child in self.tree.get_children(node):
            self.tree.delete(child)
        try:
            entries = sorted(
                path.iterdir(),
                key=lambda p: (not p.is_dir(), p.name.lower()),
            )
        except PermissionError:
            return
        for entry in entries:
            label = entry.name + ("/" if entry.is_dir() else "")
            child_node = self.tree.insert(node, "end", text=label, values=(str(entry),))
            if entry.is_dir():
                # 占位子节点，保证显示展开箭头
                self.tree.insert(child_node, "end", text="…", values=("__placeholder__",))

    def _on_open(self, _event: tk.Event) -> None:
        node = self.tree.focus()
        path_str = self._node_path(node)
        if not path_str:
            return
        path = Path(path_str)
        children = self.tree.get_children(node)
        if (
            len(children) == 1
            and self._node_path(children[0]) == "__placeholder__"
        ):
            self._populate_node(node, path)

    def _node_path(self, node: str) -> str:
        if not node:
            return ""
        values = self.tree.item(node, "values")
        return str(values[0]) if values else ""

    def _on_select(self, _event: tk.Event) -> None:
        node = self.tree.focus()
        path_str = self._node_path(node)
        if not path_str or path_str == "__placeholder__":
            return
        path = Path(path_str)
        if path.is_dir():
            self._show_directory_info(path)
            return
        self._show_file_preview(path)

    # ------------------------------------------------------------------
    # 预览面板：文本 / 图片
    # ------------------------------------------------------------------
    def _build_text_preview(self) -> None:
        self.text_widget = tk.Text(
            self.preview_container,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
            background="#fafafa",
        )
        self.text_scroll = ttk.Scrollbar(
            self.preview_container, orient="vertical", command=self.text_widget.yview
        )
        self.text_widget.configure(yscrollcommand=self.text_scroll.set)

    def _build_image_preview(self) -> None:
        self.image_label = ttk.Label(
            self.preview_container, anchor="center", background="#222"
        )

    def _hide_all_previews(self) -> None:
        self.text_widget.pack_forget()
        self.text_scroll.pack_forget()
        self.image_label.pack_forget()

    def _show_text(self, content: str) -> None:
        self._hide_all_previews()
        self.text_widget.pack(side="left", fill="both", expand=True)
        self.text_scroll.pack(side="right", fill="y")
        self.text_widget.configure(state="normal")
        self.text_widget.delete("1.0", "end")
        self.text_widget.insert("1.0", content)
        self.text_widget.configure(state="disabled")

    def _show_image(self, path: Path) -> None:
        try:
            from PIL import Image, ImageTk
        except ImportError:
            self._show_text(
                "未安装 Pillow，无法显示图片预览。\n请执行: pip install pillow"
            )
            return
        try:
            with Image.open(path) as image:
                image.load()
                max_w = max(self.preview_container.winfo_width() - 16, 360)
                max_h = max(self.preview_container.winfo_height() - 16, 360)
                image.thumbnail((max_w, max_h), Image.LANCZOS)
                photo = ImageTk.PhotoImage(image)
        except Exception as exc:
            self._show_text(f"图片读取失败: {exc}")
            return
        self._hide_all_previews()
        self.image_label.configure(image=photo, text="")
        self.image_label.pack(fill="both", expand=True)
        self._image_ref = photo  # 防 GC

    # ------------------------------------------------------------------
    # 信息显示
    # ------------------------------------------------------------------
    def _show_directory_info(self, path: Path) -> None:
        try:
            entries = list(path.iterdir())
        except PermissionError:
            self.info_var.set(f"无权限访问: {path}")
            self._show_text("")
            return
        files = [p for p in entries if p.is_file()]
        dirs = [p for p in entries if p.is_dir()]
        self.info_var.set(
            f"目录: {path} | 子目录 {len(dirs)} 个, 文件 {len(files)} 个"
        )

        listing_lines = [f"= {path} ="]
        for d in sorted(dirs):
            listing_lines.append(f"[DIR]  {d.name}/")
        for f in sorted(files):
            try:
                size = f.stat().st_size
            except OSError:
                size = 0
            listing_lines.append(f"[FILE] {f.name}  ({_human_size(size)})")
        self._show_text("\n".join(listing_lines))

    def _show_file_preview(self, path: Path) -> None:
        ext = path.suffix.lower()
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        self.info_var.set(f"文件: {path} | 大小: {_human_size(size)}")

        if ext in PREVIEW_IMAGE_EXTS:
            self._show_image(path)
            return

        if ext in PREVIEW_TEXT_EXTS:
            try:
                with path.open("rb") as fh:
                    raw = fh.read(MAX_TEXT_BYTES + 1)
            except OSError as exc:
                self._show_text(f"读取失败: {exc}")
                return
            truncated = len(raw) > MAX_TEXT_BYTES
            text = raw[:MAX_TEXT_BYTES].decode("utf-8", errors="replace")
            if ext == ".json":
                try:
                    obj = json.loads(text)
                    text = json.dumps(obj, ensure_ascii=False, indent=2)
                except json.JSONDecodeError:
                    pass
            if truncated:
                text += f"\n\n... 已截断（仅显示前 {MAX_TEXT_BYTES // 1024} KB）"
            self._show_text(text)
            return

        self._show_text(
            "该类型暂不支持内嵌预览，可点击右上角 “在系统中打开”。\n"
            f"文件: {path}"
        )

    # ------------------------------------------------------------------
    # 工具栏
    # ------------------------------------------------------------------
    def _browse_root(self) -> None:
        initial = self._root_var.get().strip() or str(self.repo_root)
        picked = filedialog.askdirectory(title="选择结果目录", initialdir=initial)
        if picked:
            self._root_var.set(picked)
            self.refresh()

    def to_settings_section(self) -> dict[str, Any]:
        return {"root_dir": self._root_var.get()}

    def _open_in_system(self) -> None:
        node = self.tree.focus()
        path_str = self._node_path(node) or self._root_var.get()
        if not path_str or path_str == "__placeholder__":
            messagebox.showinfo("提示", "请先在左侧选择文件或目录")
            return
        path = Path(path_str)
        if not path.exists():
            messagebox.showwarning("提示", f"路径不存在: {path}")
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("打开失败", str(exc))


def _human_size(num: float) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    for unit in units:
        if num < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} PB"
