"""主流程共用的输入/输出根目录（写入配置文件中的共用段）。"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from tkinter import messagebox

from ...paths import resolve_project_path
from .base import BaseCommandTab


class PathSettingsTab(BaseCommandTab):
    """主流程共用的路径与运行环境（写入 ``settings.json`` 的 ``common``）。"""

    title = "路径设置"
    description = (
        "在这里统一设置数据与结果的存放位置。路径修改会立即用于下一次任务；"
        "点击「保存共用路径」可供下次启动时继续使用。"
    )
    settings_section = "common"
    primary_button_text = "保存共用路径"

    def build_form(self, parent: tk.Misc) -> None:
        self.io_roots = self.add_io_root_section(parent)
        self.launch_env = self.add_launch_environment_section(parent)

    def compose_command(self) -> list[str]:
        # 本页的「保存」走 _on_run 覆盖逻辑，不启动子进程。
        return []

    def to_settings_section(self) -> dict[str, Any]:
        data = dict(self.io_root_to_dict(self.io_roots))
        data.update(self.launch_env_to_dict(self.launch_env))
        return data

    def validate_form(self) -> None:
        cwd = str(self.launch_env["launch_root"].get()).strip()  # type: ignore[index]
        if cwd:
            resolved = resolve_project_path(cwd, repo_root=self.repo_root)
            if not resolved.is_dir():
                raise self.launch_env["launch_root"].invalid(
                    f"子进程工作目录不存在或不是目录: {resolved}"
                )

    def _on_run(self) -> None:
        self.clear_validation_error()
        try:
            self.validate_form()
            payload = self.to_settings_section()
            if not payload:
                return
            path = self.settings.save_sections({"common": payload})
        except ValueError as exc:
            self.show_validation_error(exc)
            return
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("保存失败", str(exc))
            return
        messagebox.showinfo(
            "已保存",
            f"共用路径已写入配置文件的 common 段：\n{path}",
        )
