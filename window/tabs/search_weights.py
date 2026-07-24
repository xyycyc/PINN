"""``python -m ai_model.batch.search_fixed_weights`` 的可视化封装。"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from ..widgets import (
    PADX,
    PADY,
    FileEntry,
    LabeledCombobox,
    LabeledEntry,
    LabeledNumber,
    Section,
)
from .base import BaseCommandTab, DeviceChoices, TrainingModes


class SearchWeightsTab(BaseCommandTab):
    title = "固定权重搜索"
    description = (
        "在固定分支权重模式下，对当前模型实际存在的 CNN/LSTM 网络分支权重做网格搜索，"
        "按验证集误差排序输出汇总结果。"
    )
    settings_section = "search_weights"

    def build_form(self, parent: tk.Misc) -> None:
        section = Section(parent, "数据 / 输出")
        section.pack(fill="x", padx=PADX, pady=PADY)

        self.data_root = FileEntry(
            section,
            "数据目录",
            default=str(self._cfg_value("data_root", "database/raw")),
            directory=True,
        )
        self.data_root.pack(fill="x", padx=PADX, pady=PADY)

        self.result_root = FileEntry(
            section,
            "结果根目录",
            default=str(self._cfg_value("result_root", "result")),
            directory=True,
        )
        self.result_root.pack(fill="x", padx=PADX, pady=PADY)

        self.test_ratio = LabeledNumber(
            section,
            "测试集占比",
            float(self._cfg_value("test_ratio", 0.2)),
            is_float=True,
        )
        self.test_ratio.pack(fill="x", padx=PADX, pady=PADY)

        train_section = Section(parent, "训练参数")
        train_section.pack(fill="x", padx=PADX, pady=PADY)

        self.epochs = LabeledNumber(
            train_section,
            "训练轮数",
            int(self._cfg_value("epochs", 1000)),
            hint="须不小于 1000",
        )
        self.epochs.pack(fill="x", padx=PADX, pady=PADY)

        self.device = LabeledCombobox(
            train_section,
            "设备",
            DeviceChoices,
            default=str(self._cfg_value("device", "cuda")),
        )
        self.device.pack(fill="x", padx=PADX, pady=PADY)

        self.seed = LabeledNumber(
            train_section, "随机种子", int(self._cfg_value("seed", 42))
        )
        self.seed.pack(fill="x", padx=PADX, pady=PADY)

        self.training_mode = LabeledCombobox(
            train_section,
            "训练模式",
            TrainingModes,
            default=str(self._cfg_value("training_mode", "residual_pinn")),
        )
        self.training_mode.pack(fill="x", padx=PADX, pady=PADY)

        self.physics_residual = LabeledNumber(
            train_section,
            "物理残差权重",
            float(self._cfg_value("physics_residual_weight", 0.1)),
            is_float=True,
        )
        self.physics_residual.pack(fill="x", padx=PADX, pady=PADY)

        weight_section = Section(parent, "搜索空间")
        weight_section.pack(fill="x", padx=PADX, pady=PADY)

        self.network_weights = LabeledEntry(
            weight_section,
            "网络分支权重",
            default=str(self._cfg_value("network_weights", "0.75,1.0,1.25")),
            hint="逗号分隔；CNN 与 LSTM 共用候选列表并执行笛卡尔积组合",
        )
        self.network_weights.pack(fill="x", padx=PADX, pady=PADY)

        self.preprocess = self.add_preprocess_section(parent)

    def validate_form(self) -> None:
        ratio = self.test_ratio.get()
        if ratio is None or not (0.0 < float(ratio) < 1.0):
            raise ValueError("测试集占比必须在 0 与 1 之间")
        epochs = self.epochs.get()
        if epochs is None or int(epochs) < 1000:
            raise ValueError("训练轮数不能低于 1000")
        if not self.network_weights.get():
            raise ValueError("网络分支权重不能为空")

    def compose_command(self) -> list[str]:
        args: list[str] = []
        if (val := self.data_root.get()):
            args.extend(["--data-root", val])
        if (val := self.result_root.get()):
            args.extend(["--result-root", val])
        if (val := self.test_ratio.get()) is not None:
            args.extend(["--test-ratio", str(val)])
        if (val := self.epochs.get()) is not None:
            args.extend(["--epochs", str(val)])
        if (val := self.device.get()):
            args.extend(["--device", val])
        if (val := self.seed.get()) is not None:
            args.extend(["--seed", str(val)])
        if (val := self.training_mode.get()):
            args.extend(["--training-mode", val])
        if (val := self.physics_residual.get()) is not None:
            args.extend(["--physics-residual-weight", str(val)])
        if (val := self.network_weights.get()):
            args.extend(["--network-weights", val])
        args.extend(self.preprocess_args(self.preprocess))
        return self.python_module_cmd("ai_model.batch.search_fixed_weights", *args)

    def to_settings_section(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "data_root": self.data_root.get(),
            "result_root": self.result_root.get(),
            "test_ratio": self.test_ratio.get(),
            "epochs": self.epochs.get(),
            "device": self.device.get(),
            "seed": self.seed.get(),
            "training_mode": self.training_mode.get(),
            "physics_residual_weight": self.physics_residual.get(),
            "network_weights": self.network_weights.get(),
        }
        data.update(self.preprocess_to_dict(self.preprocess))
        return data
