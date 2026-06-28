"""``python -m ai_model demo`` 的可视化封装。"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from ..widgets import PADX, PADY, LabeledEntry, LabeledNumber, Section
from .base import BaseCommandTab


class DemoTab(BaseCommandTab):
    title = "一键演示"
    description = (
        "依次完成建库、训练与校验，适合第一次跑通整条流程。"
    )
    settings_section = "demo"

    def build_form(self, parent: tk.Misc) -> None:
        section = Section(parent, "数据规模")
        section.pack(fill="x", padx=PADX, pady=PADY)

        self.sim_per_material = LabeledNumber(
            section,
            "每材料仿真样本数",
            int(self._cfg_value("sim_per_material", 400)),
        )
        self.sim_per_material.pack(fill="x", padx=PADX, pady=PADY)

        self.experiment_limit = LabeledNumber(
            section,
            "实验样本上限",
            int(self._cfg_value("experiment_limit", 2000)),
        )
        self.experiment_limit.pack(fill="x", padx=PADX, pady=PADY)

        self.train_name = LabeledEntry(
            section,
            "训练名称",
            default=str(self._cfg_value("train_name", "")),
        )
        self.train_name.pack(fill="x", padx=PADX, pady=PADY)

        self.runtime = self.add_runtime_section(
            parent,
            include_epochs=True,
            epochs_default=int(self._cfg_value("epochs", 20)),
        )
        self.preprocess = self.add_preprocess_section(parent)

    def compose_command(self) -> list[str]:
        args: list[str] = ["demo"]
        args.extend(self.shared_io_root_args())
        sim = self.sim_per_material.get()
        if sim is not None:
            args.extend(["--sim-per-material", str(sim)])
        limit = self.experiment_limit.get()
        if limit is not None:
            args.extend(["--experiment-limit", str(limit)])
        if (val := self.train_name.get()):
            args.extend(["--train-name", val])
        args.extend(self.runtime_args(self.runtime, include_epochs=True))
        args.extend(self.preprocess_args(self.preprocess))
        return self.python_module_cmd("ai_model", *args)

    def to_settings_section(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "sim_per_material": self.sim_per_material.get(),
            "experiment_limit": self.experiment_limit.get(),
            "train_name": self.train_name.get(),
        }
        data.update(self.runtime_to_dict(self.runtime))
        data.update(self.preprocess_to_dict(self.preprocess))
        return data
