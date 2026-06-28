"""``python -m ai_model train`` 的可视化封装。"""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from typing import Any

from ...model import default_parameter_base_name, validate_rule_triplet
from ..widgets import PADX, PADY, FileEntry, LabeledCombobox, LabeledEntry, Section
from .base import BaseCommandTab


class TrainTab(BaseCommandTab):
    title = "训练模型"
    description = (
        "在指定清单上训练温度场重构模型；支持普通与残差物理两种训练模式，"
        "以及可学习与固定分支权重。"
    )
    settings_section = "train"

    def build_form(self, parent: tk.Misc) -> None:
        section = Section(parent, "数据与清单")
        section.pack(fill="x", padx=PADX, pady=PADY)

        self.manifest = FileEntry(
            section,
            "训练清单路径",
            default=str(
                self._cfg_value(
                    "manifest", "database/train_manifest.json"
                )
            ),
            filetypes=[("JSON 清单", "*.json"), ("所有文件", "*.*")],
        )
        self.manifest.pack(fill="x", padx=PADX, pady=PADY)
        auto_manifest = self.resolve_latest_split_manifest("train")
        if auto_manifest:
            self.manifest.set(auto_manifest)

        rule_section = Section(parent, "预训练规则登记（会写入 database/rule/trained_rules.csv）")
        rule_section.pack(fill="x", padx=PADX, pady=PADY)
        self.rule_dimension = LabeledCombobox(
            rule_section,
            "维度",
            ["one", "two"],
            default=str(self._cfg_value("rule_dimension", "one")),
            hint="one=一维，two=二维",
        )
        self.rule_dimension.pack(fill="x", padx=PADX, pady=PADY)
        self.rule_mode = LabeledCombobox(
            rule_section,
            "稳瞬态",
            ["steady", "transient"],
            default=str(self._cfg_value("rule_mode", "steady")),
        )
        self.rule_mode.pack(fill="x", padx=PADX, pady=PADY)
        self.rule_material = LabeledEntry(
            rule_section,
            "材料种类(英文)",
            default=str(self._cfg_value("rule_material", "metal_matrix")),
            hint="仅允许英文、数字、下划线，且必须以英文开头",
        )
        self.rule_material.pack(fill="x", padx=PADX, pady=PADY)

        out_section = Section(parent, "训练产物命名")
        out_section.pack(fill="x", padx=PADX, pady=PADY)
        self.train_name = LabeledEntry(
            out_section,
            "训练任务名称",
            default=str(self._cfg_value("train_name", "")),
            hint="用于 checkpoint/report 子目录命名，与检查点文件名是两个概念",
        )
        self.train_name.pack(fill="x", padx=PADX, pady=PADY)
        self.checkpoint_name = LabeledEntry(
            out_section,
            "检查点文件名",
            default=str(self._cfg_value("checkpoint_name", "ai_model.pt")),
            hint="保存在训练任务目录下",
        )
        self.checkpoint_name.pack(fill="x", padx=PADX, pady=PADY)

        self.preprocess = self.add_preprocess_section(parent)

        self.runtime = self.add_runtime_section(
            parent,
            include_epochs=True,
            epochs_default=int(self._cfg_value("epochs", 20)),
        )

    def compose_command(self) -> list[str]:
        dim, mode, material = validate_rule_triplet(
            self.rule_dimension.get(),
            self.rule_mode.get(),
            self.rule_material.get(),
        )
        now = datetime.now()
        generated_base = default_parameter_base_name(
            dimension=dim,
            mode=mode,
            material=material,
            now=now,
        )
        train_name = self.train_name.get() or generated_base
        checkpoint_name = self.checkpoint_name.get() or f"{generated_base}.pt"

        args: list[str] = ["train"]
        args.extend(self.shared_io_root_args())
        manifest = self.resolve_latest_split_manifest("train") or self.manifest.get()
        if manifest:
            self.manifest.set(manifest)
        if manifest:
            args.extend(["--manifest", manifest])
        args.extend(["--train-name", train_name])
        args.extend(["--checkpoint-name", checkpoint_name])
        args.extend(["--rule-dimension", dim, "--rule-mode", mode, "--rule-material", material])
        args.extend(self.preprocess_args(self.preprocess))
        args.extend(self.runtime_args(self.runtime, include_epochs=True))
        return self.python_module_cmd("ai_model", *args)

    def to_settings_section(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "manifest": self.manifest.get(),
            "train_name": self.train_name.get(),
            "checkpoint_name": self.checkpoint_name.get(),
            "rule_dimension": self.rule_dimension.get(),
            "rule_mode": self.rule_mode.get(),
            "rule_material": self.rule_material.get(),
        }
        data.update(self.preprocess_to_dict(self.preprocess))
        data.update(self.runtime_to_dict(self.runtime))
        return data
