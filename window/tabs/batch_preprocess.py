"""``python -m ai_model.batch.batch_preprocess_test`` 的可视化封装。"""

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
from .base import BaseCommandTab, DeviceChoices


class BatchPreprocessTab(BaseCommandTab):
    title = "预处理批量实验"
    description = (
        "一次性跑多组预处理流水线；每组内部再跑四种训练组合并汇总到顶层目录。"
    )
    settings_section = "batch_preprocess"

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

        self.resume_run_dir = FileEntry(
            section,
            "断点续跑目录",
            default=str(self._cfg_value("resume_run_dir", "")),
            directory=True,
        )
        self.resume_run_dir.pack(fill="x", padx=PADX, pady=PADY)

        train_section = Section(parent, "训练参数")
        train_section.pack(fill="x", padx=PADX, pady=PADY)

        self.epochs = LabeledNumber(
            train_section, "训练轮数", int(self._cfg_value("epochs", 5000))
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

        self.physics_residual = LabeledNumber(
            train_section,
            "物理残差权重",
            float(self._cfg_value("physics_residual_weight", 0.1)),
            is_float=True,
        )
        self.physics_residual.pack(fill="x", padx=PADX, pady=PADY)

        self.clip_quantile = LabeledNumber(
            train_section,
            "裁峰分位数",
            float(self._cfg_value("clip_quantile", 1.0)),
            is_float=True,
        )
        self.clip_quantile.pack(fill="x", padx=PADX, pady=PADY)

        self.smooth_window = LabeledNumber(
            train_section,
            "平滑窗口宽度",
            int(self._cfg_value("smooth_window", 11)),
        )
        self.smooth_window.pack(fill="x", padx=PADX, pady=PADY)

        pipeline_section = Section(parent, "预处理流水线列表")
        pipeline_section.pack(fill="x", padx=PADX, pady=PADY)

        self.pipelines = LabeledEntry(
            pipeline_section,
            "分号分隔多组",
            default=str(
                self._cfg_value(
                    "pipelines",
                    "base;clip,smooth;clip,smooth,detrend;smooth,robust_norm",
                )
            ),
            hint="示例：基础；裁峰与平滑；裁峰、平滑、去趋势与标准化等，用分号分隔多组",
            width=72,
        )
        self.pipelines.pack(fill="x", padx=PADX, pady=PADY)

    def validate_form(self) -> None:
        if not (self.pipelines.get()):
            raise ValueError("至少需要一组预处理流水线")
        ratio = self.test_ratio.get()
        if ratio is None or not (0.0 < float(ratio) < 1.0):
            raise ValueError("测试集占比必须在 0 与 1 之间")
        epochs = self.epochs.get()
        if epochs is None or int(epochs) <= 0:
            raise ValueError("训练轮数必须大于 0")

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
        if (val := self.physics_residual.get()) is not None:
            args.extend(["--physics-residual-weight", str(val)])
        if (val := self.clip_quantile.get()) is not None:
            args.extend(["--clip-quantile", str(val)])
        if (val := self.smooth_window.get()) is not None:
            args.extend(["--smooth-window", str(val)])
        if (val := self.pipelines.get()):
            args.extend(["--pipelines", val])
        if (val := self.resume_run_dir.get()):
            args.extend(["--resume-run-dir", val])
        return self.python_module_cmd("ai_model.batch.batch_preprocess_test", *args)

    def to_settings_section(self) -> dict[str, Any]:
        return {
            "data_root": self.data_root.get(),
            "result_root": self.result_root.get(),
            "test_ratio": self.test_ratio.get(),
            "resume_run_dir": self.resume_run_dir.get(),
            "epochs": self.epochs.get(),
            "device": self.device.get(),
            "seed": self.seed.get(),
            "physics_residual_weight": self.physics_residual.get(),
            "clip_quantile": self.clip_quantile.get(),
            "smooth_window": self.smooth_window.get(),
            "pipelines": self.pipelines.get(),
        }
