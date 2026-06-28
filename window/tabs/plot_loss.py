"""``python -m ai_model.batch.plot_batch_test_loss_curves`` 的可视化封装。"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from ..widgets import PADX, PADY, FileEntry, LabeledNumber, Section
from .base import BaseCommandTab


class PlotLossTab(BaseCommandTab):
    title = "损失曲线绘制"
    description = (
        "扫描批量测试目录中的训练历史文件，将各模式的损失曲线绘制到一张图中。"
    )
    primary_button_text = "绘制曲线"
    settings_section = "plot_loss"

    def build_form(self, parent: tk.Misc) -> None:
        section = Section(parent, "输入 / 输出")
        section.pack(fill="x", padx=PADX, pady=PADY)

        self.run_dir = FileEntry(
            section,
            "批量测试运行目录",
            default=str(self._cfg_value("run_dir", "")),
            directory=True,
        )
        self.run_dir.pack(fill="x", padx=PADX, pady=PADY)

        self.output = FileEntry(
            section,
            "输出图片路径",
            default=str(self._cfg_value("output", "")),
            filetypes=[("PNG 图片", "*.png"), ("所有文件", "*.*")],
            save=True,
        )
        self.output.pack(fill="x", padx=PADX, pady=PADY)

        self.dpi = LabeledNumber(section, "图片分辨率（每英寸点数）", int(self._cfg_value("dpi", 150)))
        self.dpi.pack(fill="x", padx=PADX, pady=PADY)

    def validate_form(self) -> None:
        if not self.run_dir.get():
            raise ValueError("请填写批量测试运行目录")

    def compose_command(self) -> list[str]:
        args: list[str] = ["--run-dir", self.run_dir.get()]
        if (out := self.output.get()):
            args.extend(["-o", out])
        if (dpi := self.dpi.get()) is not None:
            args.extend(["--dpi", str(dpi)])
        return self.python_module_cmd("ai_model.batch.plot_batch_test_loss_curves", *args)

    def to_settings_section(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir.get(),
            "output": self.output.get(),
            "dpi": self.dpi.get(),
        }
