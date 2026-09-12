"""ai_model 中文图形化操作窗口主程序。"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
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
from .settings import Settings, load_settings
from ..paths import resolve_project_path
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
from .widgets import FormValidationError
from .theme import COLORS, configure_styles
from .task_status import TaskStatus


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
        width = min(1280, max(1000, self.root.winfo_screenwidth() - 80))
        height = min(820, max(640, self.root.winfo_screenheight() - 120))
        self.root.geometry(f"{width}x{height}")
        self.root.minsize(1000, 640)
        configure_styles(self.root)

        self._log_queue: queue.Queue[str | tuple[int, int | None]] = queue.Queue()
        self._run_id = 0
        self._closing = False
        self._poll_after_id: str | None = None
        self.runner = CommandRunner(
            log_callback=self._log_from_thread,
            default_cwd=default_subprocess_cwd(),
        )

        self._build_menu()
        self._build_header()
        self._build_statusbar()
        self._build_body()
        self.toggle_log()

        self._log_line(f"[app] ai_model 包目录: {self.package_root}")
        self._log_line(f"[app] 子进程默认工作目录: {self._resolve_launch_root()}")
        self._log_line(f"[app] 配置文件: {self.settings.path}")
        self._log_line("[app] 选中功能页后可预览命令，或点击右下角按钮开始任务。")
        self._poll_log_queue()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _resolve_launch_root(self) -> Path:
        """Use the current form immediately; saving controls the next session."""
        tab = getattr(self, "_path_settings_tab", None)
        raw = (
            tab.launch_env["launch_root"].get()
            if tab is not None
            else self.settings.get("common", "launch_root", "")
        )
        raw = str(raw).strip()
        return (
            resolve_project_path(raw, repo_root=self.package_root)
            if raw
            else self.package_root.parent
        )

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
        cfg_menu.add_command(label="重新加载配置文件", command=self.reload_settings)
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

    def _build_header(self) -> None:
        header = ttk.Frame(self.root, padding=(22, 14))
        header.pack(fill="x")
        ttk.Label(header, text="AI", style="Brand.TLabel").pack(
            side="left", padx=(0, 14)
        )
        titles = ttk.Frame(header)
        titles.pack(side="left")
        ttk.Label(titles, text="温度场重构", style="AppTitle.TLabel").pack(anchor="w")
        ttk.Label(
            titles, text="数据准备  /  模型训练  /  预测与校验", style="Hint.TLabel"
        ).pack(anchor="w", pady=(2, 0))
        ttk.Button(
            header, text="打开输出目录", command=self.open_result_directory
        ).pack(side="right")
        ttk.Button(
            header,
            text="保存表单",
            command=self.save_settings_from_forms,
            style="Quiet.TButton",
        ).pack(side="right", padx=(0, 10))
        taskbar = ttk.Frame(self.root, padding=(22, 8))
        taskbar.pack(fill="x", pady=(1, 0))
        self.task_status = TaskStatus(taskbar)
        self.task_status.pack(side="left")
        self._log_toggle_button = ttk.Button(
            taskbar, text="收起日志", style="Quiet.TButton", command=self.toggle_log
        )
        self._log_toggle_button.pack(side="right")

    def _build_body(self) -> None:
        self._paned = ttk.Panedwindow(self.root, orient="vertical")
        self._paned.pack(fill="both", expand=True, padx=16, pady=(10, 0))
        notebook_frame = ttk.Frame(self._paned)
        self._paned.add(notebook_frame, weight=1)

        self._tabs: list[Any] = []
        grouped = len(TAB_GROUPS) > 1
        if grouped:
            outer = ttk.Notebook(notebook_frame)
            outer.pack(fill="both", expand=True)
        for group_name, tab_classes in TAB_GROUPS:
            notebook = ttk.Notebook(outer if grouped else notebook_frame)
            if grouped:
                outer.add(notebook, text=group_name)
            else:
                notebook.pack(fill="both", expand=True)
            path_settings = None
            for tab_class in tab_classes:
                kwargs = (
                    {"path_settings_tab": path_settings}
                    if path_settings is not None
                    else {}
                )
                tab = tab_class(
                    notebook,
                    run_callback=self.run_command,
                    stop_callback=self.stop_command,
                    repo_root=self.repo_root,
                    settings=self.settings,
                    **kwargs,
                )
                if group_name == "主流程" and isinstance(tab, PathSettingsTab):
                    path_settings = self._path_settings_tab = tab
                notebook.add(tab, text=tab.title)
                self._tabs.append(tab)

        self._log_frame = ttk.Frame(self._paned, padding=(12, 8))
        self._paned.add(self._log_frame, weight=0)
        self._log_visible = True
        self._log_height = 175
        toolbar = ttk.Frame(self._log_frame)
        toolbar.pack(fill="x", pady=(0, 6))
        ttk.Label(toolbar, text="运行日志", style="Title.TLabel").pack(
            side="left", padx=(0, 16)
        )
        self._autoscroll_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            toolbar, text="跟随最新输出", variable=self._autoscroll_var
        ).pack(side="left")
        for label, action in (
            ("保存日志…", self.save_log),
            ("复制", self.copy_log),
            ("清空", self.clear_log),
        ):
            ttk.Button(toolbar, text=label, command=action, style="Quiet.TButton").pack(
                side="right", padx=(6, 0)
            )

        text_frame = ttk.Frame(self._log_frame)
        text_frame.pack(fill="both", expand=True)
        self.log_text = tk.Text(
            text_frame,
            wrap="none",
            height=5,
            font=("Consolas", 10),
            background=COLORS["log"],
            foreground=COLORS["ink"],
            insertbackground=COLORS["ink"],
            selectbackground="#d5e8ed",
            relief="flat",
            padx=10,
            pady=6,
            state="disabled",
        )
        yscroll = ttk.Scrollbar(
            text_frame, orient="vertical", command=self.log_text.yview
        )
        xscroll = ttk.Scrollbar(
            text_frame, orient="horizontal", command=self.log_text.xview
        )
        self.log_text.configure(yscrollcommand=yscroll.set, xscrollcommand=xscroll.set)
        self.log_text.grid(row=0, column=0, sticky="nsew")
        yscroll.grid(row=0, column=1, sticky="ns")
        xscroll.grid(row=1, column=0, sticky="ew")
        text_frame.rowconfigure(0, weight=1)
        text_frame.columnconfigure(0, weight=1)
        self.root.after_idle(self._position_log_sash)

    def _position_log_sash(self) -> None:
        if self._log_visible:
            # Adding a forgotten pane schedules another geometry pass. Finish it
            # before setting the sash, or Tk can overwrite the requested height.
            self._paned.update_idletasks()
            self._paned.sashpos(
                0, max(220, self._paned.winfo_height() - self._log_height)
            )

    def toggle_log(self) -> None:
        if self._log_visible:
            height = self._log_frame.winfo_height()
            if height > 1:
                self._log_height = max(100, height)
            self._paned.forget(self._log_frame)
        else:
            self._paned.add(self._log_frame, weight=0)
        self._log_visible = not self._log_visible
        self._log_toggle_button.configure(
            text="收起日志" if self._log_visible else "展开日志"
        )
        if self._log_visible:
            self.root.after_idle(self._position_log_sash)

    def open_result_directory(self) -> None:
        path = self._path_settings_tab._resolve_result_root_path()
        if not path.is_dir():
            messagebox.showinfo(
                "输出目录尚未创建",
                f"运行任务生成结果后即可打开：\n{path}",
                parent=self.root,
            )
            return
        try:
            if sys.platform.startswith("win"):
                os.startfile(str(path))
            else:
                subprocess.Popen(
                    ["open" if sys.platform == "darwin" else "xdg-open", str(path)]
                )
        except OSError as exc:
            messagebox.showerror("无法打开目录", str(exc), parent=self.root)

    def _set_busy(self, busy: bool) -> None:
        for tab in self._tabs:
            tab.set_busy(busy)

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
        launch_root = self._resolve_launch_root()
        if not launch_root.is_dir():
            messagebox.showwarning(
                "路径有误", f"子进程工作目录不存在或不是目录: {launch_root}"
            )
            return
        titles = {
            "build-db": "构建数据库",
            "train": "训练模型",
            "predict": "温度场预测",
            "validate": "校验数据",
            "online-update": "增量训练",
            "demo": "一键演示",
        }
        title = next((titles[part] for part in cmd if part in titles), "当前任务")
        if not self._log_visible:
            self.toggle_log()
        self.task_status.start(title)
        self._set_busy(True)
        self._run_id += 1
        run_id = self._run_id
        self._set_status(f"运行中: {format_command(cmd)[:80]}…")
        started = self.runner.run(
            cmd,
            cwd=launch_root,
            on_finish=lambda code: self._on_finish(code, run_id),
        )
        if not started:
            self._set_status("启动失败")
            self.task_status.finish(-1)
            self._set_busy(False)

    def stop_command(self) -> None:
        if not self.runner.is_running:
            self._log_line("[app] 当前没有正在运行的任务。")
            return
        self.task_status.stopping()
        self._set_status("正在停止任务…")
        self.runner.request_stop()

    def _on_finish(self, exit_code: int | None, run_id: int) -> None:
        # Worker callbacks only enqueue data; all Tk calls stay on the main thread.
        self._log_queue.put((run_id, exit_code))

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
            self.settings.reload()
        except Exception as exc:
            messagebox.showerror("加载失败", str(exc))
            return
        self._rebuild_tabs()
        self._log_line(f"[settings] 已从 {self.settings.path} 重新加载配置。")
        self._set_status("配置已重载")

    def save_settings_from_forms(self) -> None:
        """Collect every form before committing either memory or the file."""
        try:
            sections = {}
            for tab in self._tabs:
                section = getattr(tab, "settings_section", "")
                if section and hasattr(tab, "to_settings_section"):
                    payload = tab.to_settings_section()
                    if payload:
                        sections[section] = payload
            target = self.settings.save_sections(sections)
        except FormValidationError as exc:
            tab.master.select(tab)
            tab.show_validation_error(exc)
            self._set_status("表单尚未保存，请修正标记的输入。")
            return
        except Exception as exc:
            messagebox.showerror("保存失败", str(exc))
            return
        self._log_line(f"[settings] 当前表单已写回 {target}")
        self._set_status(f"配置已保存: {target}")

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
        """Keep the active page and log layout when applying reloaded settings."""
        previous_log = self.log_text.get("1.0", "end-1c")
        autoscroll = self._autoscroll_var.get()
        selected = next(
            (
                tab.settings_section
                for tab in self._tabs
                if str(tab) == tab.master.select()
            ),
            "common",
        )
        visible = self._log_visible
        height = self._log_frame.winfo_height() if visible else self._log_height
        self._paned.destroy()
        self._build_body()
        self._log_height = max(100, height)
        self._autoscroll_var.set(autoscroll)
        self._append_to_text(previous_log)
        for tab in self._tabs:
            if tab.settings_section == selected:
                tab.master.select(tab)
                break
        if not visible:
            self.toggle_log()
            self._log_height = height
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
        lines = []
        # Bound work per tick so verbose training cannot starve UI events.
        for _ in range(500):
            try:
                item = self._log_queue.get_nowait()
            except queue.Empty:
                break
            if isinstance(item, tuple):
                run_id, exit_code = item
                if run_id == self._run_id:
                    self._set_status(
                        "启动失败"
                        if exit_code == -1
                        else f"已结束，退出码：{exit_code}"
                    )
                    self.task_status.finish(exit_code)
                    self._set_busy(False)
                    if exit_code != 0 and not self._log_visible:
                        self.toggle_log()
            else:
                lines.append(item)
        if lines:
            self._append_to_text("".join(lines))
        self.task_status.tick()
        if self._closing and not self.runner.is_running:
            self._destroy_window()
            return
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
        if self._closing:
            return
        if self.runner.is_running:
            if not messagebox.askyesno(
                "确认退出", "仍有命令在运行，确定要退出吗？退出会终止子进程。"
            ):
                return
            self._closing = True
            self._set_status("正在停止任务，结束后关闭窗口…")
            self.runner.request_stop()
            return
        self._destroy_window()

    def _destroy_window(self) -> None:
        if self._poll_after_id is not None:
            self.root.after_cancel(self._poll_after_id)
            self._poll_after_id = None
        self.root.destroy()

    def _show_about(self) -> None:
        messagebox.showinfo(
            "关于本程序",
            "温度场模型中文图形化操作面板。\n"
            "包含建库、训练、预测、校验、增量训练、一键演示、项目清理等主流程功能。\n\n"
            f"配置文件：{self.settings.path}\n"
            "源代码位于本项目的图形面板目录。",
        )


def launch() -> None:
    """模块入口：``python -m ai_model.window``。"""

    root = tk.Tk()
    AiModelApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch()
