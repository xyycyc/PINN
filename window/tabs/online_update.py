"""``python -m ai_model online-update`` 的可视化封装（增量训练）。"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Any

from ...artifact_paths import normalize_pt_filename
from ..rules import (
    NO_CHECKPOINT_LABEL,
    apply_checkpoint_combobox,
    available_dimensions,
    available_modes,
    default_rule_choices,
    filter_rule_rows_for_model_kind,
    load_rule_rows,
    manifest_model_kind,
    resolve_training_runtime_for_ui,
    unique_materials,
)
from ..widgets import PADX, PADY, FileEntry, LabeledCheck, LabeledCombobox, LabeledEntry, Section
from .base import BaseCommandTab

NO_RULE_PLACEHOLDER = "暂无"


class OnlineUpdateTab(BaseCommandTab):
    title = "增量训练"
    description = (
        "在已有模型检查点上，用新的训练清单做少量轮次微调。"
        "权重默认保存到基础 checkpoint 所在目录，文件名为训练时间.pt；"
        "训练报告保存到 report/<基础任务名>/Incremental/<训练时间>/；"
        "完成后自动登记到 trained_rules.csv，可在本页与预测页按三元组匹配选择。"
    )
    settings_section = "online_update"
    primary_button_text = "开始增量训练"

    def _active_manifest(self) -> str:
        if hasattr(self, "auto_manifest") and self.auto_manifest.get():
            latest = self.resolve_latest_split_manifest("combined")
            if latest:
                return latest
            return str(self._resolve_data_root_path() / "combined_manifest.json")
        return self.manifest.get() if hasattr(self, "manifest") else ""

    def _refresh_rule_rows(self, *, sync_runtime: bool = True) -> None:
        rows = load_rule_rows(
            self._resolve_data_root_path(),
            self._resolve_result_root_path(),
        )
        manifest = self._active_manifest()
        expected = manifest_model_kind(manifest, repo_root=self.repo_root) if manifest else "legacy_grid"
        self._rule_rows = filter_rule_rows_for_model_kind(rows, expected)
        self._update_rule_material_options()
        self._update_rule_dimension_options()
        self._update_rule_mode_options()
        self._sync_checkpoint_from_rule(sync_runtime=sync_runtime)
        self._rule_context = self._rule_context_key()

    def _rule_context_key(self) -> tuple[str, str, str]:
        return (
            str(self._resolve_data_root_path().resolve()),
            str(self._resolve_result_root_path().resolve()),
            str(self._active_manifest()),
        )

    def _update_rule_material_options(self) -> None:
        materials = unique_materials(self._rule_rows)
        if not materials:
            self.rule_material.combo.configure(values=[NO_RULE_PLACEHOLDER], state="disabled")
            self.rule_material.set(NO_RULE_PLACEHOLDER)
            return
        self.rule_material.combo.configure(values=materials, state="readonly")
        if materials and self.rule_material.get() not in materials:
            self.rule_material.set(materials[0])

    def _update_rule_dimension_options(self) -> None:
        dimensions = available_dimensions(self._rule_rows, material=self.rule_material.get())
        if not dimensions:
            dimensions = [self.rule_dimension.get()] if self.rule_dimension.get() else []
        self.rule_dimension.combo.configure(values=dimensions)
        if dimensions and self.rule_dimension.get() not in dimensions:
            self.rule_dimension.set(dimensions[0])

    def _update_rule_mode_options(self) -> None:
        modes = available_modes(
            self._rule_rows,
            material=self.rule_material.get(),
            dimension=self.rule_dimension.get(),
        )
        if not modes:
            modes = [self.rule_mode.get()] if self.rule_mode.get() else []
        self.rule_mode.combo.configure(values=modes)
        if modes and self.rule_mode.get() not in modes:
            self.rule_mode.set(modes[0])

    def _selected_checkpoint_path(self) -> str:
        label = self.checkpoint.get()
        if label == NO_CHECKPOINT_LABEL:
            return ""
        return self._checkpoint_label_to_path.get(label, "")

    def _sync_checkpoint_from_rule(self, *, sync_runtime: bool = True) -> None:
        previous_checkpoint = self._selected_checkpoint_path()
        preferred = previous_checkpoint or self._initial_checkpoint_path
        self._initial_checkpoint_path = ""
        apply_checkpoint_combobox(
            self.checkpoint,
            self._rule_rows,
            dimension=self.rule_dimension.get(),
            mode=self.rule_mode.get(),
            material=self.rule_material.get(),
            label_to_path=self._checkpoint_label_to_path,
            preferred_path=preferred,
        )
        # Preserve explicit form edits only while the selected model stays the
        # same. If refresh had to switch models, inherit the new model's runtime
        # rather than applying stale settings from a different checkpoint.
        if sync_runtime or self._selected_checkpoint_path() != previous_checkpoint:
            self._sync_runtime_from_checkpoint()

    def _sync_runtime_from_checkpoint(self) -> None:
        if not hasattr(self, "runtime"):
            return
        checkpoint_path = self._selected_checkpoint_path()
        if not checkpoint_path:
            # Keep values loaded from this tab's settings when no registered
            # checkpoint is available yet. There is nothing to inherit from.
            return
        runtime = resolve_training_runtime_for_ui(self._resolve_data_root_path(), checkpoint_path)
        self.apply_runtime_to_controls(self.runtime, runtime)

    def build_form(self, parent: tk.Misc) -> None:
        section = Section(parent, "输入")
        section.pack(fill="x", padx=PADX, pady=PADY)

        self.manifest = FileEntry(
            section,
            "增量训练清单路径",
            default=str(
                self._cfg_value(
                    "manifest", "database/combined_manifest.json"
                )
            ),
            filetypes=[("JSON 清单", "*.json"), ("所有文件", "*.*")],
        )
        self.manifest.pack(fill="x", padx=PADX, pady=PADY)
        self.auto_manifest = LabeledCheck(
            section,
            "清单选择",
            "自动使用当前输入根目录中最新的合并清单",
            default=bool(self._cfg_value("auto_manifest", True)),
        )
        self.auto_manifest.pack(fill="x", padx=PADX, pady=PADY)
        if self.auto_manifest.get():
            auto_manifest = self.resolve_latest_split_manifest("combined")
            self.manifest.set(
                auto_manifest or str(self._resolve_data_root_path() / "combined_manifest.json")
            )

        dims, modes = default_rule_choices()
        rule_section = Section(parent, "参数规则选择（仅可选择已登记项）")
        rule_section.pack(fill="x", padx=PADX, pady=PADY)
        self.rule_material = LabeledCombobox(
            rule_section,
            "材料种类(英文)",
            [],
            default=str(self._cfg_value("rule_material", "")),
            hint="级联第 1 步：先选择材料，再筛选维度与稳瞬态",
        )
        self.rule_material.pack(fill="x", padx=PADX, pady=PADY)
        self.rule_dimension = LabeledCombobox(
            rule_section,
            "维度",
            list(dims),
            default=str(self._cfg_value("rule_dimension", "one")),
        )
        self.rule_dimension.pack(fill="x", padx=PADX, pady=PADY)
        self.rule_mode = LabeledCombobox(
            rule_section,
            "稳瞬态",
            list(modes),
            default=str(self._cfg_value("rule_mode", "steady")),
        )
        self.rule_mode.pack(fill="x", padx=PADX, pady=PADY)
        self._checkpoint_label_to_path: dict[str, str] = {}
        self._initial_checkpoint_path = str(self._cfg_value("checkpoint", "")).strip()
        self.checkpoint = LabeledCombobox(
            rule_section,
            "基础模型检查点",
            [],
            default="",
            hint="同一规则三元组下若有多次训练，可在此选择；增量权重写入该 checkpoint 所在目录",
            width=56,
        )
        self.checkpoint.pack(fill="x", padx=PADX, pady=PADY)
        ttk.Button(rule_section, text="刷新规则映射", command=self._refresh_rule_rows).pack(
            anchor="w", padx=PADX, pady=PADY
        )

        out_section = Section(parent, "输出（可选）")
        out_section.pack(fill="x", padx=PADX, pady=PADY)
        self.output_name = LabeledEntry(
            out_section,
            "权重文件名",
            default=str(self._cfg_value("output_name", "")),
            hint="留空则使用训练时间.pt；报告目录名为同一时间戳，位于 report/<基础任务名>/Incremental/<时间戳>/",
        )
        self.output_name.pack(fill="x", padx=PADX, pady=PADY)

        self.override_preprocess = LabeledCheck(
            parent,
            "预处理覆盖策略",
            "覆盖基础 checkpoint 中保存的波形预处理（不勾选则完整继承）",
            default=bool(self._cfg_value("override_preprocess", False)),
        )
        self.override_preprocess.pack(fill="x", padx=PADX * 2, pady=PADY)
        self.preprocess = self.add_preprocess_section(parent)

        self.runtime = self.add_runtime_section(
            parent,
            include_epochs=True,
            epochs_label="增量训练轮数",
            epochs_default=int(self._cfg_value("epochs", 5)),
            runtime_hint="默认与所选基础 checkpoint 的训练模式、分支权重及轮数一致；修改后将覆盖默认值。",
        )

        self._rule_rows: list[dict[str, str]] = []
        self._rule_context: tuple[str, str, str] | None = None
        self.rule_material.combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: (
                self._update_rule_dimension_options(),
                self._update_rule_mode_options(),
                self._sync_checkpoint_from_rule(),
            ),
        )
        self.rule_dimension.combo.bind(
            "<<ComboboxSelected>>",
            lambda _e: (self._update_rule_mode_options(), self._sync_checkpoint_from_rule()),
        )
        self.rule_mode.combo.bind("<<ComboboxSelected>>", lambda _e: self._sync_checkpoint_from_rule())
        self.checkpoint.combo.bind("<<ComboboxSelected>>", lambda _e: self._sync_runtime_from_checkpoint())
        self._refresh_rule_rows()

    def validate_form(self) -> None:
        # Refresh checkpoint compatibility without overwriting runtime values the
        # user has just edited. Runtime is synchronized when the checkpoint is
        # selected, not when the Execute button validates the form.
        if self._rule_context != self._rule_context_key():
            self._refresh_rule_rows(sync_runtime=False)
        active_manifest = self._active_manifest()
        if not active_manifest:
            raise ValueError("请填写增量训练清单路径")
        if not self._selected_checkpoint_path():
            raise ValueError("当前规则组合未匹配到模型检查点，请先在训练页登记该组合")
        epochs = self.runtime["epochs"].get()  # type: ignore[union-attr]
        if epochs is None or int(epochs) <= 0:
            raise ValueError("增量训练轮数必须大于 0。")
        residual = self.runtime["residual_weight"].get()  # type: ignore[union-attr]
        if residual is None or float(residual) < 0:
            raise ValueError("物理残差权重必须大于或等于 0。")
        output_name = str(self.output_name.get() or "").strip()
        if output_name:
            normalize_pt_filename(output_name, label="增量 checkpoint 文件名")

    def compose_command(self) -> list[str]:
        manifest = self._active_manifest()
        if self.auto_manifest.get():
            self.manifest.set(manifest)
        args: list[str] = [
            "online-update",
            *self.shared_io_root_args(),
            "--manifest",
            manifest,
            "--checkpoint",
            self._selected_checkpoint_path(),
            "--rule-dimension",
            self.rule_dimension.get(),
            "--rule-mode",
            self.rule_mode.get(),
            "--rule-material",
            self.rule_material.get(),
        ]
        if (val := self.output_name.get()):
            args.extend(["--output-name", val])
        if self.override_preprocess.get():
            args.extend(self.preprocess_args(self.preprocess, include_empty=True))
        args.extend(self.runtime_args(self.runtime, include_epochs=True, explicit_online=True))
        return self.python_module_cmd("ai_model", *args)

    def to_settings_section(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "manifest": self.manifest.get(),
            "auto_manifest": bool(self.auto_manifest.get()),
            "checkpoint": self._selected_checkpoint_path(),
            "rule_dimension": self.rule_dimension.get(),
            "rule_mode": self.rule_mode.get(),
            "rule_material": self.rule_material.get(),
            "output_name": self.output_name.get(),
            "override_preprocess": bool(self.override_preprocess.get()),
        }
        data.update(self.preprocess_to_dict(self.preprocess))
        data.update(self.runtime_to_dict(self.runtime))
        return data
