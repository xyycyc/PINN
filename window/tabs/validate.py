"""``python -m ai_model validate`` 的可视化封装。"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from ..widgets import PADX, PADY, FileEntry, Section
from .base import BaseCommandTab


class ValidateTab(BaseCommandTab):
    title = "校验数据"
    description = (
        "校验当前清单是否满足文档对数据规模与数值范围等要求，结果写入运行日志。"
    )
    settings_section = "validate"

    def build_form(self, parent: tk.Misc) -> None:
        section = Section(parent, "清单文件")
        section.pack(fill="x", padx=PADX, pady=PADY)

        self.manifest = FileEntry(
            section,
            "待校验清单路径",
            default=str(
                self._cfg_value(
                    "manifest", "database/combined_manifest.json"
                )
            ),
            filetypes=[("JSON 清单", "*.json"), ("所有文件", "*.*")],
        )
        self.manifest.pack(fill="x", padx=PADX, pady=PADY)

    def compose_command(self) -> list[str]:
        args: list[str] = ["validate"]
        args.extend(self.shared_io_root_args())
        manifest = self.resolve_latest_split_manifest("combined") or self.manifest.get()
        if manifest:
            self.manifest.set(manifest)
        if manifest:
            args.extend(["--manifest", manifest])
        return self.python_module_cmd("ai_model", *args)

    def to_settings_section(self) -> dict[str, Any]:
        return {"manifest": self.manifest.get()}
