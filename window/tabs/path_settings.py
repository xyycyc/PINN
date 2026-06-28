"""主流程共用的输入/输出根目录（写入配置文件中的共用段）。"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Any

from tkinter import messagebox

from .base import BaseCommandTab


class PathSettingsTab(BaseCommandTab):
    """主流程共用的路径与运行环境（写入 ``settings.json`` 的 ``common``）。"""

    title = "路径设置"
    description = (
        "设置主流程共用的「输入根目录」「输出根目录」以及子进程工作目录；"
        "命令始终用启动本窗口的 Python 解释器执行。"
        "设备、训练轮数、权重、预处理等请在各自使用的功能页中配置。"
        "点「保存共用路径」只写回共用段；也可用菜单「把当前表单保存为默认值」与其它页一并保存。"
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
        if cwd and not Path(cwd).is_dir():
            raise ValueError(f"子进程工作目录不存在或不是目录: {cwd}")

    def _on_run(self) -> None:
        try:
            self.validate_form()
            payload = self.to_settings_section()
            if not payload:
                return
            self.settings.update_section("common", payload)
            path = self.settings.save()
        except ValueError as exc:
            messagebox.showwarning("参数有误", str(exc))
            return
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("保存失败", str(exc))
            return
        messagebox.showinfo(
            "已保存",
            f"共用路径已写入配置文件的 common 段：\n{path}",
        )
