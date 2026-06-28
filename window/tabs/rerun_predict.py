"""``python -m ai_model.batch.rerun_predict`` 的可视化封装。"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from ..widgets import (
    PADX,
    PADY,
    FileEntry,
    LabeledCheck,
    LabeledCombobox,
    LabeledNumber,
    Section,
)
from .base import BaseCommandTab, DeviceChoices


class RerunPredictTab(BaseCommandTab):
    title = "重跑预测"
    description = (
        "在已完成的批量测试运行目录上重新做推理，不重新训练，"
        "生成预测结果与汇总表。"
    )
    primary_button_text = "重跑预测"
    settings_section = "rerun_predict"

    def build_form(self, parent: tk.Misc) -> None:
        section = Section(parent, "输入")
        section.pack(fill="x", padx=PADX, pady=PADY)

        self.run_dir = FileEntry(
            section,
            "批量测试运行目录",
            default=str(self._cfg_value("run_dir", "")),
            directory=True,
        )
        self.run_dir.pack(fill="x", padx=PADX, pady=PADY)

        opts_section = Section(parent, "选项")
        opts_section.pack(fill="x", padx=PADX, pady=PADY)

        self.device = LabeledCombobox(
            opts_section,
            "设备",
            DeviceChoices,
            default=str(self._cfg_value("device", "cuda")),
        )
        self.device.pack(fill="x", padx=PADX, pady=PADY)

        self.num_field_samples = LabeledNumber(
            opts_section,
            "三联图样本数",
            int(self._cfg_value("num_field_samples", 6)),
        )
        self.num_field_samples.pack(fill="x", padx=PADX, pady=PADY)

        self.overwrite = LabeledCheck(
            opts_section,
            "强制重算",
            "若预测目录已存在仍强制重算",
            default=bool(self._cfg_value("overwrite", False)),
        )
        self.overwrite.pack(fill="x", padx=PADX, pady=PADY)

    def validate_form(self) -> None:
        if not self.run_dir.get():
            raise ValueError("请填写批量测试运行目录")

    def compose_command(self) -> list[str]:
        args: list[str] = ["--run-dir", self.run_dir.get()]
        if (val := self.device.get()):
            args.extend(["--device", val])
        if (val := self.num_field_samples.get()) is not None:
            args.extend(["--num-field-samples", str(val)])
        if self.overwrite.get():
            args.append("--overwrite")
        return self.python_module_cmd("ai_model.batch.rerun_predict", *args)

    def to_settings_section(self) -> dict[str, Any]:
        return {
            "run_dir": self.run_dir.get(),
            "device": self.device.get(),
            "num_field_samples": self.num_field_samples.get(),
            "overwrite": self.overwrite.get(),
        }
