"""``python -m ai_model train`` 的可视化封装。"""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from pathlib import Path
from typing import Any

from ...artifact_paths import normalize_pt_filename, validate_artifact_basename
from ...data_process import resolve_training_manifest_pair
from ...model import default_parameter_base_name, validate_rule_triplet
from ..rules import manifest_model_kind, resolve_gui_project_path
from ..widgets import (
    PADX,
    PADY,
    FileEntry,
    LabeledCheck,
    LabeledCombobox,
    LabeledEntry,
    LabeledNumber,
    Section,
)
from .base import BaseCommandTab


class TrainTab(BaseCommandTab):
    """Configure shared or per-material model training from the GUI."""

    title = "训练模型"
    description = (
        "选择数据集文件夹，使用 train manifest 反向传播并用 validation manifest 逐轮评估；"
        "支持普通与残差物理两种训练模式，"
        "以及可学习与固定分支权重。"
    )
    settings_section = "train"

    def _latest_training_input(self) -> str:
        train_manifest = self.resolve_latest_split_manifest("train")
        collection = self.resolve_latest_material_collection()
        if hasattr(self, "separate_materials") and self.separate_materials.get():
            return str(Path(collection).parent) if collection else ""
        candidates = [value for value in (train_manifest, collection) if value]
        if not candidates:
            return ""
        selected = max(candidates, key=lambda value: Path(value).stat().st_mtime_ns)
        return str(Path(selected).parent)

    def _manifest_for_validation(self) -> str:
        if not self.auto_manifest.get():
            return self.manifest.get()
        latest = self._latest_training_input()
        if latest:
            return latest
        return str(self._resolve_data_root_path())

    def build_form(self, parent: tk.Misc) -> None:
        """Create dataset, routing, rule, runtime, and artifact controls."""
        section = Section(parent, "数据与清单")
        section.pack(fill="x", padx=PADX, pady=PADY)

        self.manifest = FileEntry(
            section,
            "训练数据集文件夹",
            default=str(
                self._cfg_value(
                    "manifest", "database"
                )
            ),
            directory=True,
        )
        self.manifest.pack(fill="x", padx=PADX, pady=PADY)
        self.auto_manifest = LabeledCheck(
            section,
            "清单选择",
            "自动使用当前输入根目录中最新的训练数据集文件夹",
            default=bool(self._cfg_value("auto_manifest", False)),
        )
        self.auto_manifest.pack(fill="x", padx=PADX, pady=PADY)
        if self.auto_manifest.get():
            auto_manifest = self._latest_training_input()
            self.manifest.set(
                auto_manifest or str(self._resolve_data_root_path())
            )
        self.separate_materials = LabeledCheck(
            section,
            "多材料训练策略",
            "对集合中的每个材料文件夹分别训练一个完整温度场 checkpoint",
            default=bool(self._cfg_value("separate_materials", False)),
        )
        self.separate_materials.pack(fill="x", padx=PADX, pady=PADY)
        self.separate_materials.check.configure(command=self._sync_auto_manifest)
        self._sync_auto_manifest()

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
            hint="仅允许英文、数字、下划线；分别训练时实际路由材料取集合中的文件夹名",
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

        early_stopping_section = Section(parent, "验证与早停")
        early_stopping_section.pack(fill="x", padx=PADX, pady=PADY)
        self.early_stopping_patience = LabeledNumber(
            early_stopping_section,
            "早停耐心代数",
            int(self._cfg_value("early_stopping_patience", 10)),
            hint="validation loss 连续多少代未改善后停止训练",
        )
        self.early_stopping_patience.pack(fill="x", padx=PADX, pady=PADY)

    def compose_command(self) -> list[str]:
        """Translate the form into a versioned training command."""
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
        latest_manifest = ""
        if self.auto_manifest.get():
            latest_manifest = self._latest_training_input()
        manifest = latest_manifest if self.auto_manifest.get() else self.manifest.get()
        if self.auto_manifest.get():
            self.manifest.set(latest_manifest or str(self._resolve_data_root_path()))
        if manifest:
            args.extend(["--manifest", manifest])
        args.extend(["--train-name", train_name])
        args.extend(["--checkpoint-name", checkpoint_name])
        if self.separate_materials.get():
            args.append("--separate-materials")
        else:
            args.append("--no-separate-materials")
        args.extend(["--rule-dimension", dim, "--rule-mode", mode, "--rule-material", material])
        args.extend(self.preprocess_args(self.preprocess))
        args.extend(self.runtime_args(self.runtime, include_epochs=True))
        patience = self.early_stopping_patience.get()
        if patience is not None:
            args.extend(["--early-stopping-patience", str(int(patience))])
        return self.python_module_cmd("ai_model", *args)

    def _sync_auto_manifest(self) -> None:
        if not self.auto_manifest.get():
            return
        value = self._latest_training_input()
        if value:
            self.manifest.set(value)

    def validate_form(self) -> None:
        """Validate manifest compatibility and required runtime values."""
        manifest = self._manifest_for_validation()
        if not manifest:
            raise ValueError("请选择训练数据集文件夹。")
        path = resolve_gui_project_path(manifest, repo_root=self.repo_root)
        if not path.exists():
            raise ValueError(f"训练数据集路径不存在: {path}")
        train_manifest, validation_manifest = resolve_training_manifest_pair(path)
        model_kind = manifest_model_kind(train_manifest, repo_root=self.repo_root)
        if model_kind != "material_collection" and validation_manifest is None:
            raise ValueError(f"训练数据集文件夹缺少 validation manifest: {path}")
        if self.separate_materials.get() and model_kind != "material_collection":
            raise ValueError("分别训练需要选择多材料建库生成的 material_collection.json。")
        epochs = self.runtime["epochs"].get()  # type: ignore[union-attr]
        if epochs is None or int(epochs) <= 0:
            raise ValueError("训练轮数必须大于 0。")
        patience = self.early_stopping_patience.get()
        if patience is None or int(patience) <= 0:
            raise ValueError("早停耐心代数必须大于 0。")
        residual = self.runtime["residual_weight"].get()  # type: ignore[union-attr]
        if residual is None or float(residual) < 0:
            raise ValueError("物理残差权重必须大于或等于 0。")
        if str(self.train_name.get() or "").strip():
            validate_artifact_basename(
                self.train_name.get(),
                label="训练任务名称",
            )
        normalize_pt_filename(
            self.checkpoint_name.get() or "ai_model.pt",
            label="checkpoint 文件名",
        )

    def to_settings_section(self) -> dict[str, Any]:
        """Serialize user-visible training options for the next session."""
        data: dict[str, Any] = {
            "manifest": self.manifest.get(),
            "auto_manifest": bool(self.auto_manifest.get()),
            "train_name": self.train_name.get(),
            "checkpoint_name": self.checkpoint_name.get(),
            "separate_materials": bool(self.separate_materials.get()),
            "rule_dimension": self.rule_dimension.get(),
            "rule_mode": self.rule_mode.get(),
            "rule_material": self.rule_material.get(),
            "early_stopping_patience": self.early_stopping_patience.get(),
        }
        data.update(self.preprocess_to_dict(self.preprocess))
        data.update(self.runtime_to_dict(self.runtime))
        return data
