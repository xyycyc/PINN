"""模型产物管理：按三元组匹配并一键删除权重/报告/登记信息。"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from ...model.artifact_cleanup import build_cleanup_plan, execute_cleanup
from ..rules import (
    NO_CHECKPOINT_LABEL,
    apply_checkpoint_combobox,
    available_dimensions,
    available_modes,
    default_rule_choices,
    load_rule_rows,
    unique_materials,
)
from ..settings import DEFAULT_SETTINGS, Settings
from ..widgets import FileEntry, LabeledCombobox, PADX, PADY, ScrollableFrame, Section, wrapped_label
from ..theme import COLORS

NO_RULE_PLACEHOLDER = "暂无"


def _standalone_root_default(
    common: dict[str, Any],
    local: dict[str, Any],
    key: str,
    fallback: str,
) -> str:
    """Resolve roots when this tab is instantiated without ``PathSettingsTab``.

    ``Settings.section()`` fills ``local`` with built-in defaults.  Treating that
    filled value as an explicit override would therefore hide a customized
    ``common`` root.  A genuinely customized per-tab value still wins for the
    standalone use case.
    """

    local_value = str(local.get(key, "") or "").strip()
    common_value = str(common.get(key, "") or "").strip()
    built_in = str(
        DEFAULT_SETTINGS.get("manage_artifacts", {}).get(key, fallback) or fallback
    ).strip()
    if local_value and local_value != built_in:
        return local_value
    return common_value or local_value or fallback


class ManageArtifactsTab(ttk.Frame):
    title = "项目清理"
    settings_section = "manage_artifacts"
    description = (
        "按规则三元组选择模型检查点，预览并删除相关权重、训练报告、预测结果与 CSV 登记。"
    )

    def __init__(
        self,
        master: tk.Misc,
        run_callback: Callable[[list[str]], None],
        stop_callback: Callable[[], None],
        *,
        repo_root: Path,
        settings: Settings | None = None,
        path_settings_tab: Any | None = None,
    ) -> None:
        del run_callback, stop_callback
        super().__init__(master)
        self.repo_root = Path(repo_root)
        self.settings = settings or Settings()
        self.common = self.settings.section("common")
        self.local = self.settings.section(self.settings_section)
        shared_roots = getattr(path_settings_tab, "io_roots", None)
        self._shared_io_roots = shared_roots if isinstance(shared_roots, dict) else None

        def root_default(key: str, fallback: str) -> str:
            if self._shared_io_roots is not None:
                control = self._shared_io_roots.get(key)
                if control is not None and hasattr(control, "get"):
                    value = str(control.get() or "").strip()
                    if value:
                        return value
            return _standalone_root_default(self.common, self.local, key, fallback)

        self._busy = False
        actions = ttk.Frame(self, padding=(20, 12))
        actions.pack(side="bottom", fill="x")
        self._delete_button = ttk.Button(actions, text="执行删除…", command=self._execute_cleanup, style="Danger.TButton")
        self._delete_button.pack(side="right")
        ttk.Button(actions, text="扫描关联产物", command=self._preview_cleanup, style="Primary.TButton").pack(side="right", padx=(0, 10))
        self._cleanup_note = ttk.Label(actions, text="先扫描并核对待删除的文件。", style="Hint.TLabel")
        self._cleanup_note.pack(side="left")
        ttk.Separator(self).pack(side="bottom", fill="x")
        heading = ttk.Frame(self, padding=(24, 16, 24, 6))
        heading.pack(fill="x")
        ttk.Label(heading, text=self.title, style="PageTitle.TLabel").pack(anchor="w")
        wrapped_label(heading, self.description, style="Muted.TLabel").pack(fill="x", pady=(5, 0))
        self.scroll = ScrollableFrame(self)
        self.scroll.pack(fill="both", expand=True, padx=12, pady=(6, 10))
        parent = self.scroll.inner

        root_section = Section(parent, "数据根目录")
        root_section.pack(fill="x", padx=PADX, pady=PADY)
        self.data_root = FileEntry(
            root_section,
            "输入根目录",
            default=root_default("data_root", "database"),
            directory=True,
        )
        self.data_root.pack(fill="x", padx=PADX, pady=PADY)
        self.result_root = FileEntry(
            root_section,
            "输出根目录",
            default=root_default("result_root", "result"),
            directory=True,
        )
        self.result_root.pack(fill="x", padx=PADX, pady=PADY)
        self._bind_shared_root_variable("data_root", self.data_root)
        self._bind_shared_root_variable("result_root", self.result_root)

        dims, modes = default_rule_choices()
        rule_section = Section(parent, "规则三元组")
        rule_section.pack(fill="x", padx=PADX, pady=PADY)
        self.rule_material = LabeledCombobox(
            rule_section,
            "材料种类(英文)",
            [],
            default=str(self.local.get("rule_material", "")),
            hint="先选材料，再筛选维度和稳瞬态",
        )
        self.rule_material.pack(fill="x", padx=PADX, pady=PADY)
        self.rule_dimension = LabeledCombobox(
            rule_section,
            "维度",
            list(dims),
            default=str(self.local.get("rule_dimension", "one")),
        )
        self.rule_dimension.pack(fill="x", padx=PADX, pady=PADY)
        self.rule_mode = LabeledCombobox(
            rule_section,
            "稳瞬态",
            list(modes),
            default=str(self.local.get("rule_mode", "steady")),
        )
        self.rule_mode.pack(fill="x", padx=PADX, pady=PADY)
        self._checkpoint_label_to_path: dict[str, str] = {}
        self._initial_checkpoint_path = str(self.local.get("checkpoint", "")).strip()
        self.checkpoint = LabeledCombobox(
            rule_section,
            "模型检查点",
            [],
            default="",
            width=64,
            hint="按登记时间从新到旧展示；分材料 checkpoint 会按完整路由组清理",
        )
        self.checkpoint.pack(fill="x", padx=PADX, pady=PADY)

        action_row = ttk.Frame(rule_section)
        action_row.pack(anchor="w", padx=PADX, pady=(PADY, PADY))
        ttk.Button(action_row, text="刷新规则映射", command=self._refresh_rule_rows).pack(side="left")

        preview_section = Section(parent, "删除预览")
        preview_section.pack(fill="both", expand=True, padx=PADX, pady=PADY)
        self._preview = tk.Text(
            preview_section,
            wrap="word",
            font=("Consolas", 10),
            state="disabled",
            background=COLORS["log"], foreground=COLORS["ink"], relief="flat", padx=10, pady=8,
            height=12,
        )
        preview_scroll = ttk.Scrollbar(preview_section, command=self._preview.yview)
        preview_scroll.pack(side="right", fill="y")
        self._preview.configure(yscrollcommand=preview_scroll.set)
        self._preview.pack(fill="both", expand=True, padx=PADX, pady=PADY)

        self._rule_rows: list[dict[str, str]] = []
        self._latest_plan = None
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
        self._refresh_rule_rows()
        self._root_trace_handles: list[tuple[tk.Variable, str]] = []
        self._install_root_change_guards()

    def _cfg_data_root(self) -> str:
        return self._root_value("data_root", self.data_root, "database")

    def _cfg_result_root(self) -> str:
        return self._root_value("result_root", self.result_root, "result")

    def _bind_shared_root_variable(self, key: str, local_control: FileEntry) -> None:
        """Make the cleanup form and Path Settings edit the same live value."""

        if self._shared_io_roots is None:
            return
        shared_control = self._shared_io_roots.get(key)
        shared_var = getattr(shared_control, "var", None)
        if shared_var is None:
            return
        local_control.var = shared_var
        local_control.entry.configure(textvariable=shared_var)

    def _root_value(self, key: str, local_control: FileEntry, fallback: str) -> str:
        if self._shared_io_roots is not None:
            shared_control = self._shared_io_roots.get(key)
            if shared_control is not None and hasattr(shared_control, "get"):
                value = str(shared_control.get() or "").strip()
                if value:
                    return value
        return local_control.get() or fallback

    def _install_root_change_guards(self) -> None:
        """Invalidate a selection immediately when either configured root changes."""

        seen: set[str] = set()
        for control in (self.data_root, self.result_root):
            variable = getattr(control, "var", None)
            if variable is None or str(variable) in seen:
                continue
            seen.add(str(variable))
            token = variable.trace_add("write", self._on_root_changed)
            self._root_trace_handles.append((variable, token))

    def _on_root_changed(self, *_args: object) -> None:
        # Never retain an absolute checkpoint selected under a previous root.
        # Scanning is left to the explicit Refresh button so typing a path does
        # not repeatedly touch partially entered directories.
        self._latest_plan = None
        self._rule_rows = []
        self._checkpoint_label_to_path.clear()
        self.checkpoint.combo.configure(values=[NO_CHECKPOINT_LABEL], state="disabled")
        self.checkpoint.set(NO_CHECKPOINT_LABEL)
        self._set_preview_text("根目录已变化；旧选择已清空，请点击“刷新规则映射”。")

    def destroy(self) -> None:
        for variable, token in getattr(self, "_root_trace_handles", []):
            try:
                variable.trace_remove("write", token)
            except tk.TclError:
                pass
        self._root_trace_handles = []
        super().destroy()

    def _selected_checkpoint_path(self) -> str:
        label = self.checkpoint.get()
        if label == NO_CHECKPOINT_LABEL:
            return ""
        return self._checkpoint_label_to_path.get(label, "")

    def _refresh_rule_rows(self) -> None:
        self._rule_rows = load_rule_rows(
            self._cfg_data_root(),
            self._cfg_result_root(),
        )
        self._update_rule_material_options()
        self._update_rule_dimension_options()
        self._update_rule_mode_options()
        self._sync_checkpoint_from_rule()

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

    def _set_preview_text(self, text: str) -> None:
        self._preview.configure(state="normal")
        self._preview.delete("1.0", "end")
        self._preview.insert("1.0", text)
        self._preview.configure(state="disabled")

    def _preview_cleanup(self) -> None:
        checkpoint = self._selected_checkpoint_path()
        if not checkpoint:
            messagebox.showwarning("未选中模型", "当前规则组合未匹配到模型检查点。")
            return
        try:
            plan = build_cleanup_plan(
                data_root=self._cfg_data_root(),
                result_root=self._cfg_result_root(),
                checkpoint_path=checkpoint,
            )
        except Exception as exc:
            messagebox.showerror("扫描失败", str(exc))
            return
        self._latest_plan = plan
        lines: list[str] = [
            f"目标检查点: {plan.target_checkpoint}",
            f"目标类型: {'增量模型' if plan.target_is_incremental else '全量模型'}",
            f"训练任务目录: {plan.target_train_name}",
            f"CSV 登记条数: {plan.csv_rows_for_target}",
        ]
        if plan.material_router_paths:
            lines.extend(
                [
                    "分材料路由组: 是（为避免路由失效，将整组清理）",
                    "材料路由文件:",
                    *[f"  - {path}" for path in plan.material_router_paths],
                    "组内 checkpoint:",
                    *[f"  - {path}" for path in plan.material_group_checkpoints],
                ]
            )
        lines.extend(["", "将删除(目标)训练报告:"])
        if plan.target_report_paths:
            lines.extend([f"  - {p}" for p in plan.target_report_paths])
        else:
            lines.append("  - (未找到)")
        lines.append("")
        lines.append(f"发现依赖增量模型: {len(plan.dependent_incrementals)}")
        for dep in plan.dependent_incrementals:
            lines.append(f"  - {dep}")
            for report in plan.dependent_report_paths.get(str(dep), []):
                lines.append(f"      report: {report}")
        lines.append("")
        lines.append("发现可关联的预测目录:")
        has_predict = False
        for path_list in plan.predict_dirs.values():
            for run_dir in path_list:
                has_predict = True
                lines.append(f"  - {run_dir}")
        if not has_predict:
            lines.append("  - (未找到)")
        self._set_preview_text("\n".join(lines))

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._delete_button.configure(state="disabled" if busy else "normal")
        self._cleanup_note.configure(text="任务运行中，结束后可清理产物。" if busy else "先扫描并核对待删除的文件。")
        # A task may produce new files; rebuild the plan before the next deletion.
        self._latest_plan = None

    def _execute_cleanup(self) -> None:
        if self._busy:
            return
        checkpoint = self._selected_checkpoint_path()
        if not checkpoint:
            messagebox.showwarning("未选中模型", "当前规则组合未匹配到模型检查点。")
            return
        if self._latest_plan is None or str(self._latest_plan.target_checkpoint) != str(Path(checkpoint).resolve()):
            self._preview_cleanup()
            if self._latest_plan is None:
                return

        include_dependents = False
        promote_dependents = False
        dependent_count = len(self._latest_plan.dependent_incrementals)
        if dependent_count > 0:
            choice = messagebox.askyesnocancel(
                "检测到增量依赖",
                (
                    f"发现 {dependent_count} 个基于该参数的增量模型。\n\n"
                    "选择「是」：一并删除这些增量模型。\n"
                    "选择「否」：保留增量模型，并将其从 Incremental 提升为独立目录。\n"
                    "选择「取消」：终止本次操作。"
                ),
            )
            if choice is None:
                return
            include_dependents = bool(choice)
            promote_dependents = not bool(choice)

        group_note = ""
        if self._latest_plan.material_router_paths:
            group_note = (
                f"\n\n该检查点属于分材料路由组，将同时删除 "
                f"{len(self._latest_plan.material_group_checkpoints)} 个材料 checkpoint 和路由文件。"
            )
        if not messagebox.askyesno(
            "确认删除",
            f"将执行不可恢复的文件删除与 CSV 更新，确认继续吗？{group_note}",
        ):
            return

        try:
            result = execute_cleanup(
                data_root=self._cfg_data_root(),
                result_root=self._cfg_result_root(),
                checkpoint_path=checkpoint,
                include_dependents=include_dependents,
                promote_dependents=promote_dependents,
            )
        except Exception as exc:
            messagebox.showerror("删除失败", str(exc))
            return

        lines = [
            "删除完成。",
            f"删除文件: {len(result.removed_files)}",
            f"删除目录: {len(result.removed_dirs)}",
            f"删除 CSV 记录: {result.removed_csv_rows}",
            f"CSV 当前记录总数: {result.updated_csv_rows}",
            f"提升为独立增量模型: {len(result.promoted_incrementals)}",
        ]
        if result.promoted_incrementals:
            lines.append("")
            lines.append("提升详情:")
            for old_path, new_path in result.promoted_incrementals:
                lines.append(f"  - {old_path} -> {new_path}")
        self._set_preview_text("\n".join(lines))
        self._refresh_rule_rows()
        messagebox.showinfo("完成", "项目清理已完成。")

    def to_settings_section(self) -> dict[str, Any]:
        return {
            "data_root": self.data_root.get(),
            "result_root": self.result_root.get(),
            "rule_dimension": self.rule_dimension.get(),
            "rule_mode": self.rule_mode.get(),
            "rule_material": self.rule_material.get(),
            "checkpoint": self._selected_checkpoint_path(),
        }
