"""ai_model 中文图形化操作窗口主程序。"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

from .runner import (
    CommandRunner,
    ai_model_package_dir,
    default_subprocess_cwd,
    format_command,
)
from .settings import SETTINGS_PATH, Settings, load_settings
from .tabs import (
    BuildDbTab,
    DemoTab,
    ManageArtifactsTab,
    OnlineUpdateTab,
    PathSettingsTab,
    PredictTab,
    TrainTab,
    ValidateTab,
)
from .widgets import PADX, PADY, configure_styles


AI_MODEL_PACKAGE_ROOT = ai_model_package_dir()


TAB_GROUPS: list[tuple[str, list[type]]] = [
    (
        "主流程",
        [
            PathSettingsTab,
            BuildDbTab,
            TrainTab,
            PredictTab,
            ValidateTab,
            OnlineUpdateTab,
            DemoTab,
            ManageArtifactsTab,
        ],
    ),
]


class AiModelApp:
    """应用主类：装配 Notebook、日志面板、状态栏，连接 Runner、settings。"""

    def __init__(self, root: tk.Tk, settings: Settings | None = None) -> None:
        self.root = root
        self.settings = settings or load_settings()
        self.package_root = AI_MODEL_PACKAGE_ROOT
        # 各 Tab 内仍用 ``repo_root`` 指代 ai_model 包目录（解析相对 data/result 路径等）。
        self.repo_root = self.package_root
        self.root.title("ai_model 图形化操作面板")
        self.root.geometry("1280x820")
        self.root.minsize(1080, 720)
        configure_styles(self.root)

        self._log_queue: "queue.Queue[str | None]" = queue.Queue()
        self._poll_after_id: str | None = None
        self.runner = CommandRunner(
            log_callback=self._log_from_thread,
            default_cwd=default_subprocess_cwd(),
        )

        self._build_menu()
        self._build_body()
        self._build_statusbar()

        self._log_line(f"[app] ai_model 包目录: {self.package_root}")
        self._log_line(f"[app] 子进程默认工作目录: {self._resolve_launch_root()}")
        self._log_line(f"[app] 配置文件: {self.settings.path}")
        self._log_line("[app] 选中一个功能页，填好参数后点击右下角「执行」即可。")
        self._poll_log_queue()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _resolve_launch_root(self) -> Path:
        """运行 ``python -m ai_model`` 时的工作目录；空配置则用包目录上一级。"""

        raw = str(self.settings.section("common").get("launch_root", "")).strip()
        if not raw:
            return default_subprocess_cwd()
        path = Path(raw).expanduser()
        if not path.is_absolute():
            return (self.package_root / path).resolve()
        return path.resolve()

    # ------------------------------------------------------------------
    # UI 构建
    # ------------------------------------------------------------------
    def _build_menu(self) -> None:
        menubar = tk.Menu(self.root)

        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="清空日志", command=self.clear_log)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=self._on_close)
        menubar.add_cascade(label="文件", menu=file_menu)

        run_menu = tk.Menu(menubar, tearoff=0)
        run_menu.add_command(label="停止当前任务", command=self.stop_command)
        menubar.add_cascade(label="运行", menu=run_menu)

        cfg_menu = tk.Menu(menubar, tearoff=0)
        cfg_menu.add_command(
            label="重新加载配置文件", command=self.reload_settings
        )
        cfg_menu.add_command(
            label="把当前表单保存为默认值", command=self.save_settings_from_forms
        )
        cfg_menu.add_separator()
        cfg_menu.add_command(
            label="在系统编辑器中打开配置文件", command=self.open_settings_in_editor
        )
        cfg_menu.add_command(
            label="复制配置文件路径到剪贴板", command=self.copy_settings_path
        )
        menubar.add_cascade(label="配置", menu=cfg_menu)

        help_menu = tk.Menu(menubar, tearoff=0)
        help_menu.add_command(label="关于", command=self._show_about)
        menubar.add_cascade(label="帮助", menu=help_menu)
        self.root.config(menu=menubar)

    def _build_body(self) -> None:
        paned = ttk.Panedwindow(self.root, orient="vertical")
        paned.pack(fill="both", expand=True, padx=PADX, pady=PADY)

        notebook_frame = ttk.Frame(paned)
        paned.add(notebook_frame, weight=3)

        self._tabs: list[Any] = []
        use_grouped_notebook = len(TAB_GROUPS) > 1
        if use_grouped_notebook:
            outer_nb = ttk.Notebook(notebook_frame)
            outer_nb.pack(fill="both", expand=True)
        for group_name, tab_classes in TAB_GROUPS:
            if use_grouped_notebook:
                inner_nb = ttk.Notebook(outer_nb)
                outer_nb.add(inner_nb, text=group_name)
            else:
                inner_nb = ttk.Notebook(notebook_frame)
                inner_nb.pack(fill="both", expand=True)
            path_settings_ref: Any | None = None
            for tab_cls in tab_classes:
                tab_kwargs: dict[str, Any] = {}
                if group_name == "主流程" and path_settings_ref is not None:
                    if tab_cls is not PathSettingsTab:
                        tab_kwargs["path_settings_tab"] = path_settings_ref
                tab = tab_cls(
                    inner_nb,
                    run_callback=self.run_command,
                    stop_callback=self.stop_command,
                    repo_root=self.repo_root,
                    settings=self.settings,
                    **tab_kwargs,
                )
                if group_name == "主流程" and isinstance(tab, PathSettingsTab):
                    path_settings_ref = tab
                inner_nb.add(tab, text=getattr(tab_cls, "title", tab_cls.__name__))
                self._tabs.append(tab)

        log_frame = ttk.Labelframe(paned, text="运行日志", padding=4)
        paned.add(log_frame, weight=2)

        log_toolbar = ttk.Frame(log_frame)
        log_toolbar.pack(fill="x", pady=(0, PADY))
        ttk.Button(log_toolbar, text="清空日志", command=self.clear_log).pack(side="left")
        ttk.Button(log_toolbar, text="复制日志", command=self.copy_log).pack(
            side="left", padx=(PADX, 0)
        )
        ttk.Button(
            log_toolbar, text="保存日志…", command=self.save_log
        ).pack(side="left", padx=(PADX, 0))
        self._autoscroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            log_toolbar, text="自动滚动到底部", variable=self._autoscroll_var
        ).pack(side="left", padx=(PADX * 2, 0))

        text_frame = ttk.Frame(log_frame)
        text_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(
            text_frame,
            wrap="none",
            font=("Consolas", 10),
            background="#111",
            foreground="#d8d8d8",
            insertbackground="#d8d8d8",
            state="disabled",
        )
        yscroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.log_text.yview)
        xscroll = ttk.Scrollbar(text_frame, orient="horizontal", command=self.log_text.xview)
        self.log_text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)

    def _build_statusbar(self) -> None:
        self._status_var = tk.StringVar(value="空闲")
        status = ttk.Label(
            self.root,
            textvariable=self._status_var,
            style="Status.TLabel",
            anchor="w",
        )
        status.pack(side="bottom", fill="x")

    # ------------------------------------------------------------------
    # 命令执行
    # ------------------------------------------------------------------
    def run_command(self, cmd: list[str]) -> None:
        if self.runner.is_running:
            messagebox.showwarning("任务进行中", "已有命令在运行，请先停止当前任务。")
            return
        self._set_status(f"运行中: {format_command(cmd)[:80]}…")
        started = self.runner.run(
            cmd,
            cwd=self._resolve_launch_root(),
            on_finish=self._on_finish,
        )
        if not started:
            self._set_status("启动失败")

    def stop_command(self) -> None:
        if not self.runner.is_running:
            self._log_line("[app] 当前没有正在运行的任务。")
            return
        self.runner.stop()

    def _on_finish(self, exit_code: int | None) -> None:
        # 工作线程回调，转到主线程刷新状态栏。
        self._log_queue.put(None)
        self.root.after(
            0,
            lambda: self._set_status(
                f"已结束，退出码：{exit_code}" if exit_code is not None else "已结束"
            ),
        )

    # ------------------------------------------------------------------
    # 配置菜单回调
    # ------------------------------------------------------------------
    def reload_settings(self) -> None:
        """从磁盘重新加载配置文件并重建所有功能页。"""

        if self.runner.is_running:
            messagebox.showwarning("无法重载", "请先停止正在运行的任务。")
            return
        if not messagebox.askyesno(
            "重新加载配置",
            "重新加载会丢弃当前未保存的表单内容并重建所有功能页，继续吗？",
        ):
            return
        try:
            self.settings = load_settings()
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc))
            return
        self._rebuild_tabs()
        self._log_line(f"[settings] 已从 {self.settings.path} 重新加载配置。")
        self._set_status("配置已重载")

    def save_settings_from_forms(self) -> None:
        """收集所有功能页的表单状态，按配置段写回配置文件。"""

        try:
            for tab in self._tabs:
                section = getattr(tab, "settings_section", "")
                if not section:
                    continue
                if not hasattr(tab, "to_settings_section"):
                    continue
                payload = tab.to_settings_section()
                if not payload:
                    continue
                self.settings.update_section(section, payload)
            target = self.settings.save()
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self._log_line(f"[settings] 当前表单已写回 {target}")
        self._set_status(f"配置已保存: {target}")
        messagebox.showinfo("已保存", f"当前表单状态已保存到\n{target}")

    def open_settings_in_editor(self) -> None:
        path = self.settings.path
        if not path.exists():
            try:
                self.settings.save()
            except Exception as exc:
                messagebox.showerror("写入失败", str(exc))
                return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))  # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
        except Exception as exc:
            messagebox.showerror("打开失败", str(exc))

    def copy_settings_path(self) -> None:
        self.root.clipboard_clear()
        self.root.clipboard_append(str(self.settings.path))
        self._set_status(f"配置文件路径已复制: {self.settings.path}")

    def _rebuild_tabs(self) -> None:
        """配置变化时重新装配整套 Notebook。"""

        # 先把日志面板下方的旧 PanedWindow 拆掉，重建一个新的
        for child in list(self.root.children.values()):
            if isinstance(child, ttk.Panedwindow):
                child.destroy()
        # 状态栏要保留，body 重新搭起来
        self._build_body()
        # status bar 需要被推到底层；重建 body 不会覆盖它。
        # 重新填一条提示日志
        self._log_line("[app] 功能页已根据新配置重建。")

    # ------------------------------------------------------------------
    # 日志区
    # ------------------------------------------------------------------
    def _log_from_thread(self, text: str) -> None:
        self._log_queue.put(text)

    def _log_line(self, line: str) -> None:
        if not line.endswith("\n"):
            line += "\n"
        self._log_queue.put(line)

    def _poll_log_queue(self) -> None:
        try:
            while True:
                item = self._log_queue.get_nowait()
                if item is None:
                    continue
                self._append_to_text(item)
        except queue.Empty:
            pass
        self._poll_after_id = self.root.after(80, self._poll_log_queue)

    def _append_to_text(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text)
        if self._autoscroll_var.get():
            self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def clear_log(self) -> None:
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def copy_log(self) -> None:
        text = self.log_text.get("1.0", "end").strip()
        if not text:
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._set_status("日志已复制到剪贴板")

    def save_log(self) -> None:
        text = self.log_text.get("1.0", "end")
        if not text.strip():
            messagebox.showinfo("提示", "日志为空。")
            return
        default_name = datetime.now().strftime("ai_model_log_%Y%m%d_%H%M%S.log")
        path = filedialog.asksaveasfilename(
            title="保存日志",
            defaultextension=".log",
            initialfile=default_name,
            filetypes=[("日志/文本", "*.log;*.txt"), ("所有文件", "*.*")],
        )
        if not path:
            return
        try:
            Path(path).write_text(text, encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self._set_status(f"日志已保存: {path}")

    # ------------------------------------------------------------------
    # 状态栏
    # ------------------------------------------------------------------
    def _set_status(self, text: str) -> None:
        self._status_var.set(text)

    # ------------------------------------------------------------------
    # 关闭
    # ------------------------------------------------------------------
    def _on_close(self) -> None:
        if self.runner.is_running:
            if not messagebox.askyesno(
                "确认退出", "仍有命令在运行，确定要退出吗？退出会终止子进程。"
            ):
                return
            self.runner.stop()
            # 给子进程一点时间退出
            threading.Event().wait(0.2)
        if self._poll_after_id is not None:
            try:
                self.root.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
        self.root.destroy()

    def _show_about(self) -> None:
        messagebox.showinfo(
            "关于本程序",
            "温度场模型中文图形化操作面板。\n"
            "包含建库、训练、预测、校验、增量训练、一键演示、项目清理等主流程功能。\n\n"
            f"配置文件：{SETTINGS_PATH}\n"
            "源代码位于本项目的图形面板目录。",
        )


def launch() -> None:
    """模块入口：``python -m ai_model.window``。"""

    root = tk.Tk()
    AiModelApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
