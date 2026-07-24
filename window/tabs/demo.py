"""``python -m ai_model demo`` 的可视化封装。"""

from __future__ import annotations

import tkinter as tk
from datetime import datetime
from pathlib import Path
from typing import Any

from ...artifact_paths import (
    validate_artifact_basename,
    validate_distinct_artifact_basenames,
)
from ...data_process import discover_cases
from ...model import default_parameter_base_name, validate_rule_triplet
from ..widgets import (
    PADX,
    PADY,
    FileEntry,
    LabeledCheck,
    LabeledCombobox,
    LabeledEntry,
    LabeledNumber,
    MaterialSplitEditor,
    Section,
)
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

        self.experiment_dir = FileEntry(
            section,
            "实验材料目录",
            default=str(self._cfg_value("experiment_dir", "raw/calibration_sweep")),
            directory=True,
        )
        self.experiment_dir.pack(fill="x", padx=PADX, pady=PADY)

        self.multi_material_input = LabeledCheck(
            section,
            "材料目录模式",
            "多材料输入：当前目录是上层目录，每个直接子文件夹是一种样本材料",
            default=bool(self._cfg_value("multi_material_input", False)),
        )
        self.multi_material_input.pack(fill="x", padx=PADX, pady=PADY)

        def _resolve_material_parent(value: str) -> Path:
            path = Path(value).expanduser()
            return path if path.is_absolute() else self._resolve_data_root_path() / path

        self.material_splits = MaterialSplitEditor(
            section,
            source_getter=self.experiment_dir.get,
            source_resolver=_resolve_material_parent,
            initial=self._cfg_value("material_splits", {}),
        )
        self.material_splits.pack(fill="x", padx=PADX * 2, pady=PADY)
        self.multi_material_input.check.configure(
            command=lambda: (
                self.material_splits.refresh_with_message()
                if self.multi_material_input.get()
                else None
            )
        )

        self.experiment_limit = LabeledNumber(
            section,
            "实验样本上限",
            int(self._cfg_value("experiment_limit", 2000)),
        )
        self.experiment_limit.pack(fill="x", padx=PADX, pady=PADY)

        self.experiment_material = LabeledEntry(
            section,
            "数据集命名标签",
            default=str(self._cfg_value("experiment_material", "metal_matrix")),
            hint="仅用于目录/任务命名；实际样本材料路由字段取材料文件夹名",
        )
        self.experiment_material.pack(fill="x", padx=PADX, pady=PADY)

        self.waveform_crop_length = LabeledNumber(
            section,
            "case 波形固定前缀点数",
            int(self._cfg_value("waveform_crop_length", 1097)),
            hint="默认保留原始连续前 1097 点，不重采样",
        )
        self.waveform_crop_length.pack(fill="x", padx=PADX, pady=PADY)

        self.train_name = LabeledEntry(
            section,
            "训练名称",
            default=str(self._cfg_value("train_name", "")),
        )
        self.train_name.pack(fill="x", padx=PADX, pady=PADY)
        self.separate_materials = LabeledCheck(
            section,
            "多材料训练策略",
            "多材料上层目录中的每个材料文件夹分别训练完整 checkpoint",
            default=bool(self._cfg_value("separate_materials", False)),
        )
        self.separate_materials.pack(fill="x", padx=PADX, pady=PADY)

        rule_section = Section(parent, "训练规则登记（供预测/增量训练页选择）")
        rule_section.pack(fill="x", padx=PADX, pady=PADY)
        self.rule_dimension = LabeledCombobox(
            rule_section,
            "维度",
            ["one", "two"],
            default=str(self._cfg_value("rule_dimension", "two")),
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
            "规则材料标签(英文)",
            default=str(self._cfg_value("rule_material", "metal_matrix")),
            hint="混合训练用于 checkpoint 登记；分别训练时实际材料取文件夹名",
        )
        self.rule_material.pack(fill="x", padx=PADX, pady=PADY)

        split_section = Section(parent, "数据集划分")
        split_section.pack(fill="x", padx=PADX, pady=PADY)
        self.split_dataset = LabeledCheck(
            split_section,
            "自动划分数据集",
            "固定节点一键演示必须启用；旧网格数据可关闭",
            default=bool(self._cfg_value("split_dataset", True)),
        )
        self.split_dataset.pack(fill="x", padx=PADX, pady=PADY)
        self.split_test_ratio = LabeledNumber(
            split_section,
            "测试集比例",
            float(self._cfg_value("split_test_ratio", 0.2)),
            is_float=True,
        )
        self.split_test_ratio.pack(fill="x", padx=PADX, pady=PADY)
        self.split_validation_ratio = LabeledNumber(
            split_section,
            "验证集比例（固定节点 case）",
            float(self._cfg_value("split_validation_ratio", 0.1)),
            is_float=True,
        )
        self.split_validation_ratio.pack(fill="x", padx=PADX, pady=PADY)
        self.split_seed = LabeledNumber(
            split_section,
            "划分随机种子",
            int(self._cfg_value("split_seed", 42)),
        )
        self.split_seed.pack(fill="x", padx=PADX, pady=PADY)
        self.split_experiment_policy = LabeledCombobox(
            split_section,
            "旧网格实验样本划分策略",
            ["uniform", "all_experiment_train", "all_experiment_test"],
            default=str(self._cfg_value("split_experiment_policy", "uniform")),
        )
        self.split_experiment_policy.pack(fill="x", padx=PADX, pady=PADY)
        self.train_manifest_name = LabeledEntry(
            split_section,
            "训练清单文件名",
            default=str(self._cfg_value("train_manifest_name", "train_manifest.json")),
        )
        self.train_manifest_name.pack(fill="x", padx=PADX, pady=PADY)
        self.validation_manifest_name = LabeledEntry(
            split_section,
            "验证清单文件名（固定节点 case）",
            default=str(self._cfg_value("validation_manifest_name", "validation_manifest.json")),
        )
        self.validation_manifest_name.pack(fill="x", padx=PADX, pady=PADY)
        self.test_manifest_name = LabeledEntry(
            split_section,
            "测试清单文件名",
            default=str(self._cfg_value("test_manifest_name", "test_manifest.json")),
        )
        self.test_manifest_name.pack(fill="x", padx=PADX, pady=PADY)

        self.runtime = self.add_runtime_section(
            parent,
            include_epochs=True,
            epochs_default=int(self._cfg_value("epochs", 20)),
        )
        self.preprocess = self.add_preprocess_section(parent)

    def compose_command(self) -> list[str]:
        dim, mode, rule_material = validate_rule_triplet(
            self.rule_dimension.get(),
            self.rule_mode.get(),
            self.rule_material.get(),
        )
        args: list[str] = ["demo"]
        args.extend(self.shared_io_root_args())
        sim = self.sim_per_material.get()
        if sim is not None:
            args.extend(["--sim-per-material", str(sim)])
        if (experiment_dir := self.experiment_dir.get()):
            args.extend(["--experiment-dir", experiment_dir])
        if self.multi_material_input.get():
            args.append("--multi-material-input")
            for spec in self.material_splits.specs():
                args.extend(["--material-split", spec])
        else:
            args.append("--no-multi-material-input")
        limit = self.experiment_limit.get()
        if limit is not None:
            args.extend(["--experiment-limit", str(limit)])
        if (dataset_label := self.experiment_material.get()):
            args.extend(["--experiment-material", dataset_label])
        if (crop_length := self.waveform_crop_length.get()) is not None:
            args.extend(["--waveform-crop-length", str(crop_length)])
        train_name = self.train_name.get() or default_parameter_base_name(
            dimension=dim,
            mode=mode,
            material=rule_material,
            now=datetime.now(),
        )
        args.extend(["--train-name", train_name])
        args.append(
            "--separate-materials"
            if self.separate_materials.get()
            else "--no-separate-materials"
        )
        args.append("--split-dataset" if self.split_dataset.get() else "--no-split-dataset")
        if (value := self.split_test_ratio.get()) is not None:
            args.extend(["--split-test-ratio", str(value)])
        if (value := self.split_validation_ratio.get()) is not None:
            args.extend(["--split-validation-ratio", str(value)])
        if (value := self.split_seed.get()) is not None:
            args.extend(["--split-seed", str(value)])
        if (value := self.split_experiment_policy.get()):
            args.extend(["--split-experiment-policy", value])
        if (value := self.train_manifest_name.get()):
            args.extend(["--train-manifest-name", value])
        if (value := self.validation_manifest_name.get()):
            args.extend(["--validation-manifest-name", value])
        if (value := self.test_manifest_name.get()):
            args.extend(["--test-manifest-name", value])
        args.extend(
            [
                "--rule-dimension",
                dim,
                "--rule-mode",
                mode,
                "--rule-material",
                rule_material,
            ]
        )
        args.extend(self.runtime_args(self.runtime, include_epochs=True))
        args.extend(self.preprocess_args(self.preprocess))
        return self.python_module_cmd("ai_model", *args)

    def validate_form(self) -> None:
        crop_length = self.waveform_crop_length.get()
        if crop_length is None or int(crop_length) <= 0:
            raise ValueError("case 波形固定前缀点数必须大于 0。")
        validate_rule_triplet(
            self.rule_dimension.get(),
            self.rule_mode.get(),
            self.rule_material.get(),
        )
        raw_source = Path(self.experiment_dir.get()).expanduser()
        source = raw_source if raw_source.is_absolute() else self._resolve_data_root_path() / raw_source
        fixed_node_cases = self.multi_material_input.get() or bool(discover_cases(source))
        if self.multi_material_input.get():
            self.material_splits.specs()
        training_mode = str(self.runtime["training_mode"].get())  # type: ignore[union-attr]
        if fixed_node_cases and training_mode != "normal":
            raise ValueError("固定节点一键演示目前仅支持 normal 模式，请修改训练模式。")
        if self.separate_materials.get() and not self.multi_material_input.get():
            raise ValueError(
                "分别训练仅用于多材料上层目录；单个 wumu 文件夹是一种完整材料。"
            )
        epochs = self.runtime["epochs"].get()  # type: ignore[union-attr]
        if epochs is None or int(epochs) <= 0:
            raise ValueError("训练轮数必须大于 0。")
        residual = self.runtime["residual_weight"].get()  # type: ignore[union-attr]
        if residual is None or float(residual) < 0:
            raise ValueError("物理残差权重必须大于或等于 0。")
        if str(self.train_name.get() or "").strip():
            validate_artifact_basename(
                self.train_name.get(),
                label="训练任务名称",
            )
        if not self.split_dataset.get():
            if fixed_node_cases:
                raise ValueError("固定节点一键演示必须启用自动划分数据集。")
            return
        test_ratio = self.split_test_ratio.get()
        validation_ratio = self.split_validation_ratio.get()
        if test_ratio is None or not (0.0 < float(test_ratio) < 1.0):
            raise ValueError("测试集比例必须在 (0,1) 区间内。")
        if validation_ratio is None or not (0.0 <= float(validation_ratio) < 1.0):
            raise ValueError("验证集比例必须在 [0,1) 区间内。")
        if float(test_ratio) + float(validation_ratio) >= 1.0:
            raise ValueError("测试集比例与验证集比例之和必须小于 1。")
        for label, value in (
            ("训练", self.train_manifest_name.get()),
            ("验证", self.validation_manifest_name.get()),
            ("测试", self.test_manifest_name.get()),
        ):
            if not str(value or "").strip():
                raise ValueError(f"请填写{label}清单文件名。")
            validate_artifact_basename(
                value,
                label=f"{label}清单文件名",
                suffix=".json",
            )
        validate_distinct_artifact_basenames(
            (
                self.train_manifest_name.get(),
                self.validation_manifest_name.get(),
                self.test_manifest_name.get(),
            ),
            label="训练/验证/测试清单文件名",
            reserved=("manifest.json", "combined_manifest.json", "split_config.json"),
        )

    def to_settings_section(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "sim_per_material": self.sim_per_material.get(),
            "experiment_dir": self.experiment_dir.get(),
            "multi_material_input": bool(self.multi_material_input.get()),
            "material_splits": self.material_splits.to_settings(),
            "experiment_limit": self.experiment_limit.get(),
            "experiment_material": self.experiment_material.get(),
            "waveform_crop_length": self.waveform_crop_length.get(),
            "train_name": self.train_name.get(),
            "separate_materials": bool(self.separate_materials.get()),
            "rule_dimension": self.rule_dimension.get(),
            "rule_mode": self.rule_mode.get(),
            "rule_material": self.rule_material.get(),
            "split_dataset": bool(self.split_dataset.get()),
            "split_test_ratio": self.split_test_ratio.get(),
            "split_validation_ratio": self.split_validation_ratio.get(),
            "split_seed": self.split_seed.get(),
            "split_experiment_policy": self.split_experiment_policy.get(),
            "train_manifest_name": self.train_manifest_name.get(),
            "validation_manifest_name": self.validation_manifest_name.get(),
            "test_manifest_name": self.test_manifest_name.get(),
        }
        data.update(self.runtime_to_dict(self.runtime))
        data.update(self.preprocess_to_dict(self.preprocess))
        return data
