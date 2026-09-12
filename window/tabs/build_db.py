"""``python -m ai_model build-db`` 的可视化封装。"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from ...artifact_paths import (
    validate_artifact_basename,
    validate_distinct_artifact_basenames,
)
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


class BuildDbTab(BaseCommandTab):
    """Expose single-material, collection, and external-test database workflows."""

    title = "构建数据库"
    description = (
        "可按需导入内置仿真、实验波形、外部仿真 CSV，最后合并为清单文件；"
        "输入输出根目录在「路径设置」中统一配置，本页负责原始数据导入与数据集划分。"
    )
    settings_section = "build_db"

    def build_form(self, parent: tk.Misc) -> None:
        """Create the database source, split, and external-test controls."""
        section = Section(parent, "数据来源")
        section.pack(fill="x", padx=PADX, pady=PADY)

        self.sim_per_material = LabeledNumber(
            section,
            "每材料仿真样本数",
            int(self._cfg_value("sim_per_material", 400)),
        )
        self.sim_per_material.pack(fill="x", padx=PADX, pady=PADY)
        self.skip_simulation = LabeledCheck(
            section,
            "仿真数据",
            "跳过内置仿真生成（默认启用）",
            default=bool(self._cfg_value("skip_simulation", True)),
        )
        self.skip_simulation.pack(fill="x", padx=PADX, pady=PADY)

        self.experiment_dir = FileEntry(
            section,
            "实验数据目录",
            default=str(self._cfg_value("experiment_dir", "raw/wumu")),
            directory=True,
            resolver=self._resolve_source_path,
        )
        self.experiment_dir.pack(fill="x", padx=PADX, pady=PADY)

        self.multi_material_input = LabeledCheck(
            section,
            "材料目录模式",
            "多材料输入：当前目录是上层目录，每个直接子文件夹是一种样本材料",
            default=bool(self._cfg_value("multi_material_input", False)),
        )
        self.multi_material_input.pack(fill="x", padx=PADX, pady=PADY)

        self.material_splits = MaterialSplitEditor(
            section,
            source_getter=self.experiment_dir.get,
            source_resolver=self._resolve_source_path,
            initial=self._cfg_value("material_splits", {}),
        )
        self.material_splits.pack(fill="x", padx=PADX * 2, pady=PADY)
        self.multi_material_input.check.configure(
            command=self._on_multi_material_mode_changed
        )

        self.experiment_material = LabeledEntry(
            section,
            "实验数据集命名标签",
            default=str(self._cfg_value("experiment_material", "wumu")),
            hint="仅用于输出目录/任务命名；实际样本材料路由字段取材料文件夹名",
        )
        self.experiment_material.pack(fill="x", padx=PADX, pady=PADY)

        self.experiment_limit = LabeledNumber(
            section,
            "实验样本上限",
            int(self._cfg_value("experiment_limit", 2000)),
        )
        self.experiment_limit.pack(fill="x", padx=PADX, pady=PADY)

        self.waveform_crop_length = LabeledNumber(
            section,
            "case 波形固定前缀点数",
            int(self._cfg_value("waveform_crop_length", 1097)),
            hint="默认保留原始连续前 1097 点，不重采样",
        )
        self.waveform_crop_length.pack(fill="x", padx=PADX, pady=PADY)

        self.external_sim_dir = FileEntry(
            section,
            "外部仿真 CSV 目录（可选）",
            default=str(self._cfg_value("external_sim_dir", "")),
            directory=True,
            resolver=self._resolve_source_path,
        )
        self.external_sim_dir.pack(fill="x", padx=PADX, pady=PADY)

        self.external_sim_material = LabeledEntry(
            section,
            "外部仿真材料标签",
            default=str(self._cfg_value("external_sim_material", "")),
            hint="不填则复用实验材料标签",
        )
        self.external_sim_material.pack(fill="x", padx=PADX, pady=PADY)

        self.external_sim_limit = LabeledNumber(
            section,
            "外部仿真样本上限",
            int(self._cfg_value("external_sim_limit", -1)),
            hint="-1 表示不限制，0 表示不导入",
        )
        self.external_sim_limit.pack(fill="x", padx=PADX, pady=PADY)

        external_test_section = Section(parent, "独立波形测试集（不参与训练）")
        external_test_section.pack(fill="x", padx=PADX, pady=PADY)
        self.external_test_enabled = LabeledCheck(
            external_test_section,
            "外部测试导入",
            "启用 post0 波形测试集；全部样本只进入指定材料的测试清单",
            default=bool(self._cfg_value("external_test_enabled", False)),
        )
        self.external_test_enabled.pack(fill="x", padx=PADX, pady=PADY)
        self.external_test_dir = FileEntry(
            external_test_section,
            "post0 CSV 目录",
            default=str(self._cfg_value("external_test_dir", "")),
            directory=True,
            resolver=self._resolve_source_path,
        )
        self.external_test_dir.pack(fill="x", padx=PADX, pady=PADY)
        self.external_test_material = LabeledEntry(
            external_test_section,
            "挂接材料",
            default=str(self._cfg_value("external_test_material", "wumu")),
            hint="例如 wumu；该材料在上方划分表中的测试比例应设为 0",
        )
        self.external_test_material.pack(fill="x", padx=PADX, pady=PADY)
        self.external_test_dataset_name = LabeledEntry(
            external_test_section,
            "测试数据集名称",
            default=str(self._cfg_value("external_test_dataset_name", "wumu_exp")),
        )
        self.external_test_dataset_name.pack(fill="x", padx=PADX, pady=PADY)
        self.external_test_signal_column = LabeledEntry(
            external_test_section,
            "声波列名",
            default=str(
                self._cfg_value(
                    "external_test_signal_column",
                    "amplitude_filtered_residual",
                )
            ),
        )
        self.external_test_signal_column.pack(fill="x", padx=PADX, pady=PADY)
        self.external_test_time_column = LabeledEntry(
            external_test_section,
            "时间列名",
            default=str(self._cfg_value("external_test_time_column", "time_s")),
        )
        self.external_test_time_column.pack(fill="x", padx=PADX, pady=PADY)
        self.external_test_time_min_s = LabeledNumber(
            external_test_section,
            "post0 起始时间（秒）",
            float(self._cfg_value("external_test_time_min_s", 0.0)),
            is_float=True,
        )
        self.external_test_time_min_s.pack(fill="x", padx=PADX, pady=PADY)
        self.external_test_limit = LabeledNumber(
            external_test_section,
            "外部测试样本上限",
            int(self._cfg_value("external_test_limit", -1)),
            hint="-1 表示全部导入",
        )
        self.external_test_limit.pack(fill="x", padx=PADX, pady=PADY)

        split_section = Section(parent, "数据集划分")
        split_section.pack(fill="x", padx=PADX, pady=PADY)
        self.split_dataset = LabeledCheck(
            split_section,
            "自动划分数据集",
            "固定节点写出 train/validation/test；旧网格数据写出 train/test",
            default=bool(self._cfg_value("split_dataset", True)),
        )
        self.split_dataset.pack(fill="x", padx=PADX, pady=PADY)
        self.split_test_ratio = LabeledNumber(
            split_section,
            "测试集比例",
            float(self._cfg_value("split_test_ratio", 0.2)),
            is_float=True,
            hint="范围 (0,1)，例如 0.2 表示 8:2",
        )
        self.split_test_ratio.pack(fill="x", padx=PADX, pady=PADY)
        self.split_validation_ratio = LabeledNumber(
            split_section,
            "验证集比例（固定节点 case）",
            float(self._cfg_value("split_validation_ratio", 0.1)),
            is_float=True,
            hint="固定节点默认 train/validation/test = 0.7/0.1/0.2；旧网格数据忽略此项",
        )
        self.split_validation_ratio.pack(fill="x", padx=PADX, pady=PADY)
        self.split_seed = LabeledNumber(
            split_section,
            "划分随机种子",
            int(self._cfg_value("split_seed", 42)),
            hint="同一数据+同一策略+同一种子可复现实验",
        )
        self.split_seed.pack(fill="x", padx=PADX, pady=PADY)
        self.split_experiment_policy = LabeledCombobox(
            split_section,
            "实验样本划分策略",
            ["uniform", "all_experiment_train", "all_experiment_test"],
            default=str(self._cfg_value("split_experiment_policy", "uniform")),
            hint="uniform=均匀划分；all_experiment_* = 全部实验样本进入某一侧",
        )
        self.split_experiment_policy.pack(fill="x", padx=PADX, pady=PADY)
        self.train_manifest_name = LabeledEntry(
            split_section,
            "训练清单文件名",
            default=str(self._cfg_value("train_manifest_name", "train_manifest.json")),
        )
        self.train_manifest_name.pack(fill="x", padx=PADX, pady=PADY)
        self.test_manifest_name = LabeledEntry(
            split_section,
            "测试清单文件名",
            default=str(self._cfg_value("test_manifest_name", "test_manifest.json")),
        )
        self.test_manifest_name.pack(fill="x", padx=PADX, pady=PADY)
        self.validation_manifest_name = LabeledEntry(
            split_section,
            "验证清单文件名（固定节点 case）",
            default=str(self._cfg_value("validation_manifest_name", "validation_manifest.json")),
        )
        self.validation_manifest_name.pack(fill="x", padx=PADX, pady=PADY)
        self._sync_split_controls()

    def _on_multi_material_mode_changed(self) -> None:
        if self.multi_material_input.get():
            self.material_splits.refresh_with_message()
        self._sync_split_controls()

    def _sync_split_controls(self) -> None:
        """Keep exactly one split-ratio source visible for the selected mode."""
        global_split_controls = (
            self.split_dataset,
            self.split_test_ratio,
            self.split_validation_ratio,
            self.split_experiment_policy,
        )
        if self.multi_material_input.get():
            self.split_dataset.set(True)
            for control in global_split_controls:
                control.pack_forget()
            return

        if not self.split_dataset.winfo_manager():
            self.split_dataset.pack(
                fill="x", padx=PADX, pady=PADY, before=self.split_seed
            )
        if not self.split_test_ratio.winfo_manager():
            self.split_test_ratio.pack(
                fill="x", padx=PADX, pady=PADY, before=self.split_seed
            )
        if not self.split_validation_ratio.winfo_manager():
            self.split_validation_ratio.pack(
                fill="x", padx=PADX, pady=PADY, before=self.split_seed
            )
        if not self.split_experiment_policy.winfo_manager():
            self.split_experiment_policy.pack(
                fill="x", padx=PADX, pady=PADY, before=self.train_manifest_name
            )

    def compose_command(self) -> list[str]:
        """Translate the visible form into one deterministic build-db command."""
        args: list[str] = ["build-db"]
        args.extend(self.shared_io_root_args())
        sim = self.sim_per_material.get()
        if sim is not None:
            args.extend(["--sim-per-material", str(sim)])
        if self.skip_simulation.get():
            args.append("--skip-simulation")
        else:
            args.append("--no-skip-simulation")
        exp_dir = self.experiment_dir.get()
        if exp_dir:
            args.extend(["--experiment-dir", exp_dir])
        if self.multi_material_input.get():
            args.append("--multi-material-input")
            for spec in self.material_splits.specs():
                args.extend(["--material-split", spec])
        else:
            args.append("--no-multi-material-input")
        if self.external_test_enabled.get():
            args.extend(["--external-test-dir", self.external_test_dir.get()])
            args.extend(["--external-test-material", self.external_test_material.get()])
            args.extend(
                ["--external-test-dataset-name", self.external_test_dataset_name.get()]
            )
            args.extend(
                ["--external-test-signal-column", self.external_test_signal_column.get()]
            )
            args.extend(
                ["--external-test-time-column", self.external_test_time_column.get()]
            )
            external_time_min = self.external_test_time_min_s.get()
            if external_time_min is not None:
                args.extend(["--external-test-time-min-s", str(external_time_min)])
            external_test_limit = self.external_test_limit.get()
            if external_test_limit is not None:
                args.extend(["--external-test-limit", str(external_test_limit)])
        material = self.experiment_material.get()
        if material:
            args.extend(["--experiment-material", material])
        limit = self.experiment_limit.get()
        if limit is not None:
            args.extend(["--experiment-limit", str(limit)])
        crop_length = self.waveform_crop_length.get()
        if crop_length is not None:
            args.extend(["--waveform-crop-length", str(crop_length)])
        ext_dir = self.external_sim_dir.get()
        if ext_dir:
            args.extend(["--external-sim-dir", ext_dir])
        ext_material = self.external_sim_material.get()
        if ext_material:
            args.extend(["--external-sim-material", ext_material])
        ext_limit = self.external_sim_limit.get()
        if ext_limit is not None:
            args.extend(["--external-sim-limit", str(ext_limit)])
        if self.split_dataset.get():
            args.append("--split-dataset")
        else:
            args.append("--no-split-dataset")
        if not self.multi_material_input.get():
            ratio = self.split_test_ratio.get()
            if ratio is not None:
                args.extend(["--split-test-ratio", str(ratio)])
            validation_ratio = self.split_validation_ratio.get()
            if validation_ratio is not None:
                args.extend(["--split-validation-ratio", str(validation_ratio)])
        seed = self.split_seed.get()
        if seed is not None:
            args.extend(["--split-seed", str(seed)])
        if not self.multi_material_input.get():
            policy = self.split_experiment_policy.get()
            if policy:
                args.extend(["--split-experiment-policy", str(policy)])
        train_manifest_name = self.train_manifest_name.get()
        if train_manifest_name:
            args.extend(["--train-manifest-name", str(train_manifest_name)])
        test_manifest_name = self.test_manifest_name.get()
        if test_manifest_name:
            args.extend(["--test-manifest-name", str(test_manifest_name)])
        validation_manifest_name = self.validation_manifest_name.get()
        if validation_manifest_name:
            args.extend(["--validation-manifest-name", str(validation_manifest_name)])
        return self.python_module_cmd("ai_model", *args)

    def validate_form(self) -> None:
        """Reject incompatible sources and ambiguous split configurations early."""
        crop_length = self.waveform_crop_length.get()
        if crop_length is None or int(crop_length) <= 0:
            raise ValueError("case 波形固定前缀点数必须大于 0。")
        if self.multi_material_input.get():
            if not self.skip_simulation.get():
                raise ValueError("多材料固定节点建库不能同时生成旧版内置仿真数据。")
            if self.external_sim_dir.get():
                raise ValueError("多材料固定节点建库不能同时导入旧版外部仿真 CSV。")
            if not self.split_dataset.get():
                raise ValueError("多材料建库必须启用数据集划分。")
            self.material_splits.specs()
        if self.external_test_enabled.get():
            if not self.multi_material_input.get():
                raise ValueError("独立波形测试集只能在“多材料输入”模式下导入。")
            if not self.external_test_dir.get():
                raise ValueError("请选择 post0 CSV 测试目录。")
            external_material = self.external_test_material.get()
            if not external_material:
                raise ValueError("请填写外部测试集要挂接的材料。")
            if not self.external_test_dataset_name.get():
                raise ValueError("请填写外部测试数据集名称。")
            if not self.external_test_signal_column.get():
                raise ValueError("请填写外部测试声波列名。")
            if not self.external_test_time_column.get():
                raise ValueError("请填写外部测试时间列名。")
            material_splits = self.material_splits.as_dict()
            if external_material not in material_splits:
                raise ValueError(
                    f"挂接材料 {external_material!r} 不在扫描到的多材料文件夹中。"
                )
            if abs(float(material_splits[external_material][2])) > 1e-9:
                raise ValueError(
                    f"材料 {external_material} 已使用独立测试集，"
                    "请将上方该材料的测试比例设为 0。"
                )
            external_limit = self.external_test_limit.get()
            if external_limit is not None and int(external_limit) == 0:
                raise ValueError("外部测试样本上限不能为 0；使用 -1 表示全部导入。")
        if self.split_dataset.get():
            if not self.multi_material_input.get():
                ratio = self.split_test_ratio.get()
                if ratio is None or not (0.0 < float(ratio) < 1.0):
                    raise ValueError("测试集比例必须在 (0,1) 区间内。")
                validation_ratio = self.split_validation_ratio.get()
                if validation_ratio is None or not (0.0 <= float(validation_ratio) < 1.0):
                    raise ValueError("验证集比例必须在 [0,1) 区间内。")
                if float(ratio) + float(validation_ratio) >= 1.0:
                    raise ValueError("测试集比例与验证集比例之和必须小于 1。")
            if not str(self.train_manifest_name.get() or "").strip():
                raise ValueError("请填写训练清单文件名。")
            if not str(self.test_manifest_name.get() or "").strip():
                raise ValueError("请填写测试清单文件名。")
            if not str(self.validation_manifest_name.get() or "").strip():
                raise ValueError("请填写验证清单文件名。")
            for label, value in (
                ("训练清单文件名", self.train_manifest_name.get()),
                ("验证清单文件名", self.validation_manifest_name.get()),
                ("测试清单文件名", self.test_manifest_name.get()),
            ):
                validate_artifact_basename(
                    value,
                    label=label,
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
        """Serialize the current form without changing project data."""
        data: dict[str, Any] = {
            "sim_per_material": self.sim_per_material.get(),
            "skip_simulation": bool(self.skip_simulation.get()),
            "experiment_dir": self.experiment_dir.get(),
            "multi_material_input": bool(self.multi_material_input.get()),
            "material_splits": self.material_splits.to_settings(),
            "experiment_material": self.experiment_material.get(),
            "experiment_limit": self.experiment_limit.get(),
            "waveform_crop_length": self.waveform_crop_length.get(),
            "external_sim_dir": self.external_sim_dir.get(),
            "external_sim_material": self.external_sim_material.get(),
            "external_sim_limit": self.external_sim_limit.get(),
            "external_test_enabled": bool(self.external_test_enabled.get()),
            "external_test_dir": self.external_test_dir.get(),
            "external_test_material": self.external_test_material.get(),
            "external_test_dataset_name": self.external_test_dataset_name.get(),
            "external_test_signal_column": self.external_test_signal_column.get(),
            "external_test_time_column": self.external_test_time_column.get(),
            "external_test_time_min_s": self.external_test_time_min_s.get(),
            "external_test_limit": self.external_test_limit.get(),
            "split_dataset": bool(self.split_dataset.get()),
            "split_test_ratio": self.split_test_ratio.get(),
            "split_validation_ratio": self.split_validation_ratio.get(),
            "split_seed": self.split_seed.get(),
            "split_experiment_policy": self.split_experiment_policy.get(),
            "train_manifest_name": self.train_manifest_name.get(),
            "test_manifest_name": self.test_manifest_name.get(),
            "validation_manifest_name": self.validation_manifest_name.get(),
        }
        return data
