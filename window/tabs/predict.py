"""``python -m ai_model predict`` 的可视化封装。"""

from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

from ...artifact_paths import validate_artifact_basename
from ...model import default_predict_output_name
from ..rules import (
    NO_CHECKPOINT_LABEL,
    apply_checkpoint_combobox,
    available_dimensions,
    available_modes,
    default_rule_choices,
    filter_rule_rows_for_model_kind,
    load_rule_rows,
    manifest_model_kind,
    resolve_gui_project_path,
    unique_materials,
)
from ..widgets import PADX, PADY, FileEntry, LabeledCheck, LabeledCombobox, LabeledEntry, LabeledNumber, Section
from .base import BaseCommandTab, DeviceChoices

NO_RULE_PLACEHOLDER = "暂无"


class PredictTab(BaseCommandTab):
    """Configure checkpoint or material-router prediction and comparison artifacts."""

    title = "预测对比"
    description = (
        "基于训练好的模型检查点对清单中的样本做推理，在输出目录生成预测结果、"
        "散点对比图、温度场三联图和指标文件，并可选用纯推理方式测量耗时。"
    )
    settings_section = "predict"

    def _active_manifest(self) -> str:
        if hasattr(self, "auto_manifest") and self.auto_manifest.get():
            if (
                hasattr(self, "auto_material_routing")
                and self.auto_material_routing.get()
            ):
                collection = self.resolve_latest_material_collection()
                if collection:
                    return collection
            candidates = [
                value
                for value in (
                    self.resolve_latest_split_manifest("test"),
                    self.resolve_latest_material_collection(),
                )
                if value
            ]
            latest = (
                max(candidates, key=lambda value: Path(value).stat().st_mtime_ns)
                if candidates
                else ""
            )
            if latest:
                return latest
            return str(self._resolve_data_root_path() / "test_manifest.json")
        return self.manifest.get() if hasattr(self, "manifest") else ""

    def _latest_material_router(self) -> str:
        result_root = self._resolve_result_root_path()
        checkpoint_root = result_root / "train" / "checkpoint"
        if not checkpoint_root.is_dir():
            return ""
        candidates: list[tuple[int, Path]] = []
        for path in checkpoint_root.rglob("*__material_router.json"):
            try:
                if path.is_file():
                    candidates.append((path.stat().st_mtime_ns, path))
            except OSError:
                continue
        if not candidates:
            return ""
        candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
        manifest_value = self._active_manifest()
        manifest_path = (
            resolve_gui_project_path(manifest_value, repo_root=self.repo_root)
            if manifest_value
            else None
        )
        if manifest_path is None:
            return ""
        manifest_parent = manifest_path.resolve().parent
        for _mtime_ns, candidate in candidates:
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
                source_value = str(
                    payload.get("source_collection", payload.get("source_manifest", ""))
                ).strip()
                if source_value and Path(source_value).resolve() == manifest_path.resolve():
                    return str(candidate)
                checkpoints = payload.get("checkpoints", [])
                if any(
                    Path(str(item.get("train_manifest", ""))).resolve().parent
                    == manifest_parent
                    for item in checkpoints
                    if isinstance(item, dict) and str(item.get("train_manifest", "")).strip()
                ):
                    return str(candidate)
            except (OSError, ValueError, json.JSONDecodeError):
                continue
        # Do not silently route a different dataset merely because its router is newer.
        return ""

    def _refresh_rule_rows(self) -> None:
        rows = load_rule_rows(
            self._resolve_data_root_path(),
            self._resolve_result_root_path(),
        )
        manifest = self._active_manifest()
        expected = manifest_model_kind(manifest, repo_root=self.repo_root) if manifest else "legacy_grid"
        if expected == "material_collection":
            expected = "direct_point_field"
        self._rule_rows = filter_rule_rows_for_model_kind(rows, expected)
        self._update_rule_material_options()
        self._update_rule_dimension_options()
        self._update_rule_mode_options()
        self._sync_checkpoint_from_rule()
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
        if self.rule_material.get() not in materials:
            self.rule_material.set(materials[0])

    def _update_rule_dimension_options(self) -> None:
        dimensions = available_dimensions(self._rule_rows, material=self.rule_material.get())
        if not dimensions:
            dimensions = [self.rule_dimension.get()] if self.rule_dimension.get() else []
        self.rule_dimension.combo.configure(values=dimensions)
        if dimensions and self.rule_dimension.get() not in dimensions:
            self.rule_dimension.set(dimensions[0])
        self._update_prediction_dimension_options()

    def _update_prediction_dimension_options(self) -> None:
        if not hasattr(self, "prediction_dimension"):
            return
        source_dimension = (
            "two"
            if hasattr(self, "auto_material_routing") and self.auto_material_routing.get()
            else str(self.rule_dimension.get() or "one")
        )
        allowed = ["two", "one"] if source_dimension == "two" else ["one"]
        self.prediction_dimension.combo.configure(values=allowed, state="readonly")
        if self.prediction_dimension.get() not in allowed:
            self.prediction_dimension.set(allowed[0])

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

    def _sync_checkpoint_from_rule(self) -> None:
        preferred = self._selected_checkpoint_path() or self._initial_checkpoint_path
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

    def build_form(self, parent: tk.Misc) -> None:
        """Create manifest, model routing, output, plot, and runtime controls."""
        section = Section(parent, "输入 / 输出")
        section.pack(fill="x", padx=PADX, pady=PADY)

        self.manifest = FileEntry(
            section,
            "待预测清单路径",
            default=str(
                self._cfg_value(
                    "manifest", "database/test_manifest.json"
                )
            ),
            filetypes=[("JSON 清单", "*.json"), ("所有文件", "*.*")],
        )
        self.manifest.pack(fill="x", padx=PADX, pady=PADY)
        self.auto_manifest = LabeledCheck(
            section,
            "清单选择",
            "自动使用当前输入根目录中最新的测试清单",
            default=bool(self._cfg_value("auto_manifest", True)),
        )
        self.auto_manifest.pack(fill="x", padx=PADX, pady=PADY)
        if self.auto_manifest.get():
            auto_manifest = self.resolve_latest_split_manifest("test")
            self.manifest.set(
                auto_manifest or str(self._resolve_data_root_path() / "test_manifest.json")
            )

        dims, modes = default_rule_choices()
        rule_section = Section(parent, "参数规则选择（仅可选择已登记项）")
        rule_section.pack(fill="x", padx=PADX, pady=PADY)
        self.auto_material_routing = LabeledCheck(
            rule_section,
            "多材料预测策略",
            "按样本记录/材料文件夹的 material_key 选择对应的完整 checkpoint",
            default=bool(self._cfg_value("auto_material_routing", False)),
        )
        self.auto_material_routing.pack(fill="x", padx=PADX, pady=PADY)
        self.auto_material_routing.check.configure(
            command=lambda: (
                self._sync_routed_manifest(),
                self._update_prediction_dimension_options(),
            )
        )
        self._sync_routed_manifest()
        self.material_router = FileEntry(
            rule_section,
            "材料路由文件",
            default=str(self._cfg_value("material_router", "")),
            filetypes=[("材料路由", "*__material_router.json"), ("JSON", "*.json")],
        )
        self.material_router.pack(fill="x", padx=PADX, pady=PADY)
        if not self.material_router.get():
            latest_router = self._latest_material_router()
            if latest_router:
                self.material_router.set(latest_router)
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
            "模型检查点",
            [],
            default="",
            hint="同一规则三元组下若有多次训练，可在此选择登记时间与训练任务",
            width=56,
        )
        self.checkpoint.pack(fill="x", padx=PADX, pady=PADY)
        ttk.Button(rule_section, text="刷新规则映射", command=self._refresh_rule_rows).pack(
            anchor="w", padx=PADX, pady=PADY
        )

        self.output_dir = FileEntry(
            section,
            "输出目录（留空则自动）",
            default=str(self._cfg_value("output_dir", "")),
            directory=True,
        )
        self.output_dir.pack(fill="x", padx=PADX, pady=PADY)

        self.predict_name = LabeledEntry(
            section,
            "推理名称",
            default=str(self._cfg_value("predict_name", "")),
            hint="为空时使用「检查点所在目录名_预测时间」",
        )
        self.predict_name.pack(fill="x", padx=PADX, pady=PADY)

        plot_section = Section(parent, "可视化 / 测速")
        plot_section.pack(fill="x", padx=PADX, pady=PADY)

        self.prediction_dimension = LabeledCombobox(
            plot_section,
            "预测输出维度",
            ["two", "one"],
            default=str(self._cfg_value("prediction_dimension", "two")),
            hint="二维模型可选 two/one；one 提取归一化 x=0.5 中心轴；一维模型只能选 one",
        )
        self.prediction_dimension.pack(fill="x", padx=PADX, pady=PADY)

        self.enable_plots = LabeledCheck(
            plot_section,
            "生成对比图",
            "",
            default=bool(self._cfg_value("enable_plots", False)),
        )
        self.enable_plots.pack(fill="x", padx=PADX, pady=PADY)

        self.num_field_samples = LabeledNumber(
            plot_section,
            "三联图样本数",
            int(self._cfg_value("num_field_samples", 6)),
            hint="仅勾选「生成对比图」时生效；全量出图过慢故只抽样",
        )
        self.num_field_samples.pack(fill="x", padx=PADX, pady=PADY)

        self.enable_benchmark = LabeledCheck(
            plot_section,
            "纯推理测速",
            "",
            default=bool(self._cfg_value("enable_benchmark", False)),
        )
        self.enable_benchmark.pack(fill="x", padx=PADX, pady=PADY)

        self.benchmark_warmup = LabeledNumber(
            plot_section,
            "测速预热样本数",
            int(self._cfg_value("benchmark_warmup_samples", 64)),
            hint="大于清单样本数时循环使用清单样本直至达到该数",
        )
        self.benchmark_warmup.pack(fill="x", padx=PADX, pady=PADY)

        self.benchmark_runs = LabeledNumber(
            plot_section,
            "测速计时轮数",
            int(self._cfg_value("benchmark_runs", 3)),
            hint="计时的前向 batch 次数；每轮从清单开头取一个 batch",
        )
        self.benchmark_runs.pack(fill="x", padx=PADX, pady=PADY)

        self.override_preprocess = LabeledCheck(
            parent,
            "预处理覆盖策略",
            "覆盖 checkpoint 中保存的波形预处理（不勾选则完整继承训练配置）",
            default=bool(self._cfg_value("override_preprocess", False)),
        )
        self.override_preprocess.pack(fill="x", padx=PADX * 2, pady=PADY)
        self.preprocess = self.add_preprocess_section(parent)

        runtime_section = Section(parent, "推理运行参数")
        runtime_section.pack(fill="x", padx=PADX, pady=PADY)
        self.infer_device = LabeledCombobox(
            runtime_section,
            "设备",
            ["", *DeviceChoices],
            default=str(self._cfg_value("device", "")),
            hint="选第一项（空）则使用 checkpoint 中的设备；否则覆盖 checkpoint 配置",
        )
        self.infer_device.pack(fill="x", padx=PADX, pady=PADY)

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
            lambda _e: (
                self._update_rule_mode_options(),
                self._sync_checkpoint_from_rule(),
                self._update_prediction_dimension_options(),
            ),
        )
        self.rule_mode.combo.bind("<<ComboboxSelected>>", lambda _e: self._sync_checkpoint_from_rule())
        self._refresh_rule_rows()

    def _sync_routed_manifest(self) -> None:
        if not self.auto_manifest.get():
            return
        if self.auto_material_routing.get():
            value = self.resolve_latest_material_collection()
        else:
            candidates = [
                item
                for item in (
                    self.resolve_latest_split_manifest("test"),
                    self.resolve_latest_material_collection(),
                )
                if item
            ]
            value = (
                max(candidates, key=lambda item: Path(item).stat().st_mtime_ns)
                if candidates
                else ""
            )
        if value:
            self.manifest.set(value)

    def validate_form(self) -> None:
        """Validate model routing and output-dimension compatibility."""
        active_manifest = self._active_manifest()
        if not active_manifest:
            raise ValueError("请填写待预测清单路径")
        if (
            not self.auto_material_routing.get()
            and self.rule_dimension.get() == "one"
            and self.prediction_dimension.get() != "one"
        ):
            raise ValueError("一维模型只能选择 one（一维）预测输出")
        if self.auto_manifest.get():
            self.manifest.set(active_manifest)
        if self.auto_material_routing.get():
            router = self.material_router.get() or self._latest_material_router()
            if not router:
                raise ValueError("请填写材料路由文件，或先执行按材料分开训练")
            self.material_router.set(router)
            if str(self.predict_name.get() or "").strip():
                validate_artifact_basename(self.predict_name.get(), label="推理名称")
            return
        if self._rule_context != self._rule_context_key():
            self._refresh_rule_rows()
        if not self._selected_checkpoint_path():
            raise ValueError("当前规则组合未匹配到模型检查点，请先在训练页登记该组合")
        if str(self.predict_name.get() or "").strip():
            validate_artifact_basename(self.predict_name.get(), label="推理名称")

    def compose_command(self) -> list[str]:
        """Translate the selected prediction workflow into a CLI command."""
        manifest = self._active_manifest()
        if self.auto_manifest.get():
            self.manifest.set(manifest)
        args: list[str] = [
            "predict",
            *self.shared_io_root_args(),
            "--manifest",
            manifest,
        ]
        if self.auto_material_routing.get():
            router = self.material_router.get() or self._latest_material_router()
            if router:
                self.material_router.set(router)
            args.extend(["--material-router", router])
        else:
            args.extend(
                [
                    "--checkpoint",
                    self._selected_checkpoint_path(),
                    "--rule-dimension",
                    self.rule_dimension.get(),
                    "--rule-mode",
                    self.rule_mode.get(),
                    "--rule-material",
                    self.rule_material.get(),
                ]
            )
        out_dir = self.output_dir.get()
        if out_dir:
            args.extend(["--output-dir", out_dir])
        predict_name = str(self.predict_name.get() or "").strip()
        if not predict_name:
            model_source = (
                self.material_router.get()
                if self.auto_material_routing.get()
                else self._selected_checkpoint_path()
            )
            if model_source:
                predict_name = default_predict_output_name(model_source)
        if predict_name:
            args.extend(["--predict-name", predict_name])
        if self.enable_plots.get():
            args.append("--plots")
        else:
            args.append("--no-plots")
        args.extend(
            ["--prediction-dimension", str(self.prediction_dimension.get() or "one")]
        )
        n = self.num_field_samples.get()
        if n is not None:
            args.extend(["--num-field-samples", str(n)])
        if self.enable_benchmark.get():
            args.append("--benchmark")
        else:
            args.append("--no-benchmark")
        warmup = self.benchmark_warmup.get()
        if warmup is not None:
            args.extend(["--benchmark-warmup-samples", str(warmup)])
        runs = self.benchmark_runs.get()
        if runs is not None:
            args.extend(["--benchmark-runs", str(runs)])
        device = str(self.infer_device.get() or "").strip()
        if device:
            args.extend(["--device", device])
        if self.override_preprocess.get():
            args.extend(self.preprocess_args(self.preprocess, include_empty=True))
        return self.python_module_cmd("ai_model", *args)

    def to_settings_section(self) -> dict[str, Any]:
        """Serialize the prediction form without executing inference."""
        data: dict[str, Any] = {
            "manifest": self.manifest.get(),
            "auto_manifest": bool(self.auto_manifest.get()),
            "checkpoint": self._selected_checkpoint_path(),
            "auto_material_routing": bool(self.auto_material_routing.get()),
            "material_router": self.material_router.get(),
            "rule_dimension": self.rule_dimension.get(),
            "rule_mode": self.rule_mode.get(),
            "rule_material": self.rule_material.get(),
            "output_dir": self.output_dir.get(),
            "predict_name": self.predict_name.get(),
            "prediction_dimension": self.prediction_dimension.get(),
            "enable_plots": bool(self.enable_plots.get()),
            "num_field_samples": self.num_field_samples.get(),
            "enable_benchmark": bool(self.enable_benchmark.get()),
            "benchmark_warmup_samples": self.benchmark_warmup.get(),
            "benchmark_runs": self.benchmark_runs.get(),
            "override_preprocess": bool(self.override_preprocess.get()),
        }
        data["device"] = self.infer_device.get()
        data.update(self.preprocess_to_dict(self.preprocess))
        return data
