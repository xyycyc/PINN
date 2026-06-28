"""``python -m ai_model build-db`` 的可视化封装。"""

from __future__ import annotations

import tkinter as tk
from typing import Any

from ..widgets import PADX, PADY, FileEntry, LabeledCheck, LabeledCombobox, LabeledEntry, LabeledNumber, Section
from .base import BaseCommandTab


class BuildDbTab(BaseCommandTab):
    title = "构建数据库"
    description = (
        "可按需导入内置仿真、实验波形、外部仿真 CSV，最后合并为清单文件；"
        "输入输出根目录在「路径设置」中统一配置，预处理在本页配置。"
    )
    settings_section = "build_db"

    def build_form(self, parent: tk.Misc) -> None:
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
            default=str(self._cfg_value("experiment_dir", "raw/10times")),
            directory=True,
        )
        self.experiment_dir.pack(fill="x", padx=PADX, pady=PADY)

        self.experiment_material = LabeledEntry(
            section,
            "实验材料标签",
            default=str(self._cfg_value("experiment_material", "metal_matrix")),
            hint="会自动写入 database/rule/material.csv 并在后续训练中复用",
        )
        self.experiment_material.pack(fill="x", padx=PADX, pady=PADY)

        self.experiment_limit = LabeledNumber(
            section,
            "实验样本上限",
            int(self._cfg_value("experiment_limit", 2000)),
        )
        self.experiment_limit.pack(fill="x", padx=PADX, pady=PADY)

        self.external_sim_dir = FileEntry(
            section,
            "外部仿真 CSV 目录（可选）",
            default=str(self._cfg_value("external_sim_dir", "")),
            directory=True,
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

        split_section = Section(parent, "数据集划分")
        split_section.pack(fill="x", padx=PADX, pady=PADY)
        self.split_dataset = LabeledCheck(
            split_section,
            "自动划分 train/test",
            "建库完成后自动写出 train/test 清单",
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

    def compose_command(self) -> list[str]:
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
        material = self.experiment_material.get()
        if material:
            args.extend(["--experiment-material", material])
        limit = self.experiment_limit.get()
        if limit is not None:
            args.extend(["--experiment-limit", str(limit)])
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
        ratio = self.split_test_ratio.get()
        if ratio is not None:
            args.extend(["--split-test-ratio", str(ratio)])
        seed = self.split_seed.get()
        if seed is not None:
            args.extend(["--split-seed", str(seed)])
        policy = self.split_experiment_policy.get()
        if policy:
            args.extend(["--split-experiment-policy", str(policy)])
        train_manifest_name = self.train_manifest_name.get()
        if train_manifest_name:
            args.extend(["--train-manifest-name", str(train_manifest_name)])
        test_manifest_name = self.test_manifest_name.get()
        if test_manifest_name:
            args.extend(["--test-manifest-name", str(test_manifest_name)])
        return self.python_module_cmd("ai_model", *args)

    def validate_form(self) -> None:
        if self.split_dataset.get():
            ratio = self.split_test_ratio.get()
            if ratio is None or not (0.0 < float(ratio) < 1.0):
                raise ValueError("测试集比例必须在 (0,1) 区间内。")
            if not str(self.train_manifest_name.get() or "").strip():
                raise ValueError("请填写训练清单文件名。")
            if not str(self.test_manifest_name.get() or "").strip():
                raise ValueError("请填写测试清单文件名。")

    def to_settings_section(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "sim_per_material": self.sim_per_material.get(),
            "skip_simulation": bool(self.skip_simulation.get()),
            "experiment_dir": self.experiment_dir.get(),
            "experiment_material": self.experiment_material.get(),
            "experiment_limit": self.experiment_limit.get(),
            "external_sim_dir": self.external_sim_dir.get(),
            "external_sim_material": self.external_sim_material.get(),
            "external_sim_limit": self.external_sim_limit.get(),
            "split_dataset": bool(self.split_dataset.get()),
            "split_test_ratio": self.split_test_ratio.get(),
            "split_seed": self.split_seed.get(),
            "split_experiment_policy": self.split_experiment_policy.get(),
            "train_manifest_name": self.train_manifest_name.get(),
            "test_manifest_name": self.test_manifest_name.get(),
        }
        return data
