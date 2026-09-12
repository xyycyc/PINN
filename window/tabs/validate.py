"""``python -m ai_model validate`` 的可视化封装。"""

from __future__ import annotations

import tkinter as tk
from pathlib import Path
from typing import Any

from ..widgets import PADX, PADY, FileEntry, LabeledCheck, Section
from .base import BaseCommandTab


class ValidateTab(BaseCommandTab):
    title = "校验数据"
    description = (
        "校验当前清单是否满足文档对数据规模与数值范围等要求，结果写入运行日志。"
    )
    settings_section = "validate"

    def _latest_validation_manifest(self) -> str:
        candidates = [
            value
            for value in (
                self.resolve_latest_split_manifest("combined"),
                self.resolve_latest_material_collection(),
            )
            if value
        ]
        if not candidates:
            return ""
        return max(candidates, key=lambda value: Path(value).stat().st_mtime_ns)

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
        self.auto_manifest = LabeledCheck(
            section,
            "清单选择",
            "自动使用最新的合并清单或多材料集合",
            default=bool(self._cfg_value("auto_manifest", False)),
        )
        self.auto_manifest.pack(fill="x", padx=PADX, pady=PADY)
        if self.auto_manifest.get():
            auto_manifest = self._latest_validation_manifest()
            self.manifest.set(
                auto_manifest or str(self._resolve_data_root_path() / "combined_manifest.json")
            )

    def compose_command(self) -> list[str]:
        args: list[str] = ["validate"]
        args.extend(self.shared_io_root_args())
        latest_manifest = self._latest_validation_manifest() if self.auto_manifest.get() else ""
        manifest = latest_manifest if self.auto_manifest.get() else self.manifest.get()
        if self.auto_manifest.get():
            self.manifest.set(
                latest_manifest or str(self._resolve_data_root_path() / "combined_manifest.json")
            )
        if manifest:
            args.extend(["--manifest", manifest])
        return self.python_module_cmd("ai_model", *args)

    def to_settings_section(self) -> dict[str, Any]:
        return {
            "manifest": self.manifest.get(),
            "auto_manifest": bool(self.auto_manifest.get()),
        }
