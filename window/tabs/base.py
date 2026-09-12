"""Tab 基类：统一参数收集 → 命令构造 → 执行 / 停止 流程。"""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

from ...data_process import latest_material_collection, latest_split_manifest
from ...paths import resolve_source_path
from ..settings import Settings
from ..rules import resolve_gui_project_path
from ..widgets import (
    PADX,
    PADY,
    FileEntry,
    LabeledCheck,
    LabeledCombobox,
    LabeledEntry,
    LabeledNumber,
    ScrollableFrame,
    Section,
    FormValidationError,
    wrapped_label,
)


PreprocessPresets: list[tuple[str, str]] = [
    ("自定义 / 不预处理", ""),
    ("clip,smooth", "clip,smooth"),
    ("clip,smooth,detrend", "clip,smooth,detrend"),
    ("smooth,robust_norm", "smooth,robust_norm"),
    ("clip,smooth,detrend,robust_norm", "clip,smooth,detrend,robust_norm"),
]
DeviceChoices = ["cuda", "cpu", "auto", "gpu"]
TrainingModes = ["normal", "residual_pinn"]


class BaseCommandTab(ttk.Frame):
    """所有命令型 Tab 的父类。

    子类通常需要：
    1. 在 ``build_form()`` 中添加表单控件（用 ``self.scroll.inner`` 作 master）；
    2. 实现 ``compose_command() -> list[str]`` 返回最终要执行的命令；
    3. 可选实现 ``validate_form()`` 抛出 ``ValueError`` 阻止运行；
    4. 可选实现 ``to_settings_section()`` 返回当前表单状态的 dict，
       供 “保存当前表单为默认值” 写回 ``settings.json``。

    settings 中的 ``common`` section 是跨 Tab 复用的默认值（设备、
    训练模式、fixed 权重等）；每个 Tab 又有自己的 section 名（由
    ``settings_section`` 类属性决定）。
    """

    title: str = ""
    description: str = ""
    primary_button_text: str = "执行"
    settings_section: str = ""

    def __init__(
        self,
        master: tk.Misc,
        run_callback: Callable[[list[str]], None],
        stop_callback: Callable[[], None],
        *,
        repo_root,
        settings: Settings | None = None,
        path_settings_tab: BaseCommandTab | None = None,
    ) -> None:
        super().__init__(master)
        self._run_cb, self._stop_cb = run_callback, stop_callback
        self.repo_root = repo_root
        self.settings = settings or Settings()
        self._path_settings = path_settings_tab
        self.common = self.settings.section("common")
        self.local = (
            self.settings.section(self.settings_section)
            if self.settings_section
            else {}
        )
        self._busy = False

        # Pack the actions first so that they remain reachable on short screens.
        self._action_bar = ttk.Frame(self, padding=(20, 12))
        self._action_bar.pack(side="bottom", fill="x")
        self._feedback = tk.StringVar(value="")
        self._feedback_label = wrapped_label(
            self._action_bar, "", textvariable=self._feedback, style="Error.TLabel"
        )
        self._button_row = buttons = ttk.Frame(self._action_bar)
        buttons.pack(fill="x")
        action_text = {
            "build_db": "构建数据库",
            "train": "开始训练",
            "predict": "开始预测",
            "validate": "开始校验",
            "demo": "运行演示",
        }.get(self.settings_section, self.primary_button_text)
        self._run_button = ttk.Button(
            buttons, text=action_text, command=self._on_run, style="Primary.TButton"
        )
        self._run_button.pack(side="right")
        self._stop_button = None
        if self.settings_section != "common":
            self._stop_button = ttk.Button(
                buttons,
                text="停止任务",
                command=self._on_stop,
                style="Danger.TButton",
                state="disabled",
            )
            self._stop_button.pack(side="right", padx=(0, 10))
            self._preview_button = ttk.Button(
                buttons,
                text="预览命令",
                command=self._on_preview,
                style="Quiet.TButton",
            )
            self._preview_button.pack(side="left")
        ttk.Separator(self).pack(side="bottom", fill="x")

        heading = ttk.Frame(self, padding=(24, 16, 24, 6))
        heading.pack(fill="x")
        ttk.Label(heading, text=self.title, style="PageTitle.TLabel").pack(anchor="w")
        if self.description:
            wrapped_label(heading, self.description, style="Muted.TLabel").pack(
                fill="x", pady=(5, 0)
            )
        self.scroll = ScrollableFrame(self)
        self.scroll.pack(fill="both", expand=True, padx=12, pady=(6, 10))
        self.build_form(self.scroll.inner)

    # ---- 子类需要重写 ------------------------------------------------------
    def build_form(self, parent: tk.Misc) -> None:  # pragma: no cover - 抽象
        raise NotImplementedError

    def compose_command(self) -> list[str]:  # pragma: no cover - 抽象
        raise NotImplementedError

    def validate_form(self) -> None:
        return None

    def to_settings_section(self) -> dict[str, Any] | None:
        """子类可重写：返回当前表单状态。返回 None 表示不参与持久化。"""

        return None

    # ---- 复用工具 ----------------------------------------------------------
    def add_runtime_section(
        self,
        parent: tk.Misc,
        *,
        include_epochs: bool = True,
        include_fixed_weights: bool = True,
        epochs_default: int | None = None,
        epochs_label: str = "训练轮数",
        runtime_hint: str = "",
    ) -> dict[str, object]:
        """创建训练相关的统一参数组（设备 / 模式 / 残差权重 / fixed 权重 / epochs）。

        默认值优先从当前 Tab 的 ``settings_section`` 取，缺失时回落
        到 ``common`` section，最后才用硬编码兜底。
        """

        section = Section(parent, "训练运行参数")
        section.pack(fill="x", padx=PADX, pady=PADY)
        if runtime_hint:
            ttk.Label(section, text=runtime_hint, wraplength=560, justify="left").pack(
                anchor="w", padx=PADX, pady=(0, PADY)
            )

        device = LabeledCombobox(
            section,
            "设备",
            DeviceChoices,
            default=self._cfg_value("device", self.common.get("device", "cuda")),
            hint="优先使用图形处理器，不可用时退回中央处理器",
        )
        device.pack(fill="x", padx=PADX, pady=PADY)

        training_mode = LabeledCombobox(
            section,
            "训练模式",
            TrainingModes,
            default=self._cfg_value(
                "training_mode", self.common.get("training_mode", "normal")
            ),
            hint="普通模式为纯监督；残差物理模式附加物理残差项",
        )
        training_mode.pack(fill="x", padx=PADX, pady=PADY)

        residual_weight = LabeledNumber(
            section,
            "物理残差权重",
            self._cfg_value(
                "physics_residual_weight",
                self.common.get("physics_residual_weight", 0.1),
            ),
            is_float=True,
            hint="仅在残差物理模式下生效，须 ≥ 0",
        )
        residual_weight.pack(fill="x", padx=PADX, pady=PADY)

        learnable = LabeledCheck(
            section,
            "分支权重",
            "启用可学习分支权重",
            default=bool(
                self._cfg_value(
                    "learnable_branch_weights",
                    self.common.get("learnable_branch_weights", False),
                )
            ),
        )
        learnable.pack(fill="x", padx=PADX, pady=PADY)

        result: dict[str, object] = {
            "device": device,
            "training_mode": training_mode,
            "residual_weight": residual_weight,
            "learnable": learnable,
        }

        if include_fixed_weights:
            fixed_section = Section(
                parent, "CNN/LSTM 分支权重（可学习模式下作为初始值）"
            )
            fixed_section.pack(fill="x", padx=PADX, pady=PADY)

            fw_cnn = LabeledNumber(
                fixed_section,
                "卷积网络分支权重",
                self._cfg_value(
                    "fixed_weight_cnn", self.common.get("fixed_weight_cnn", 0.75)
                ),
                is_float=True,
            )
            fw_cnn.pack(fill="x", padx=PADX, pady=PADY)
            fw_lstm = LabeledNumber(
                fixed_section,
                "长短期记忆网络分支权重",
                self._cfg_value(
                    "fixed_weight_lstm", self.common.get("fixed_weight_lstm", 0.75)
                ),
                is_float=True,
            )
            fw_lstm.pack(fill="x", padx=PADX, pady=PADY)

            result.update(
                {
                    "fixed_weight_cnn": fw_cnn,
                    "fixed_weight_lstm": fw_lstm,
                }
            )

        if include_epochs:
            default_epochs = (
                epochs_default
                if epochs_default is not None
                else self._cfg_value("epochs", 20)
            )
            epochs = LabeledNumber(
                section,
                epochs_label,
                int(default_epochs),
                is_float=False,
            )
            epochs.pack(fill="x", padx=PADX, pady=PADY)
            result["epochs"] = epochs
        return result

    def add_io_root_section(
        self, parent: tk.Misc, title: str = "输入/输出根目录"
    ) -> dict[str, object]:
        section = Section(parent, title)
        section.pack(fill="x", padx=PADX, pady=PADY)
        data_root = FileEntry(
            section,
            "输入根目录",
            default=str(self._cfg_value("data_root", "database")),
            directory=True,
        )
        data_root.pack(fill="x", padx=PADX, pady=PADY)
        result_root = FileEntry(
            section,
            "输出根目录",
            default=str(self._cfg_value("result_root", "result")),
            directory=True,
        )
        result_root.pack(fill="x", padx=PADX, pady=PADY)
        return {"data_root": data_root, "result_root": result_root}

    def io_root_args(self, controls: dict[str, object]) -> list[str]:
        args: list[str] = []
        if val := controls["data_root"].get():  # type: ignore[union-attr]
            args.extend(["--data-root", str(val)])
        if val := controls["result_root"].get():  # type: ignore[union-attr]
            args.extend(["--result-root", str(val)])
        return args

    def io_root_to_dict(self, controls: dict[str, object]) -> dict[str, Any]:
        return {
            "data_root": controls["data_root"].get(),  # type: ignore[union-attr]
            "result_root": controls["result_root"].get(),  # type: ignore[union-attr]
        }

    def add_launch_environment_section(self, parent: tk.Misc) -> dict[str, object]:
        """子进程工作目录（``python -m ai_model`` 的 cwd）；解释器固定为启动 GUI 的 Python。"""

        section = Section(parent, "子进程工作目录")
        section.pack(fill="x", padx=PADX, pady=PADY)
        wrapped_label(
            section,
            "通常留空即可。仅在需要指定运行位置时修改；程序使用启动窗口时的 Python 环境。",
            style="Hint.TLabel",
        ).pack(fill="x", padx=PADX, pady=(0, PADY))

        launch_root = FileEntry(
            section,
            "子进程工作目录",
            default=str(
                self._cfg_value("launch_root", self.common.get("launch_root", ""))
            ),
            directory=True,
        )
        launch_root.pack(fill="x", padx=PADX, pady=PADY)

        return {"launch_root": launch_root}

    def launch_env_to_dict(self, controls: dict[str, object]) -> dict[str, Any]:
        return {
            "launch_root": controls["launch_root"].get(),  # type: ignore[union-attr]
        }

    def python_module_cmd(self, module: str, *args: str) -> list[str]:
        """使用启动 GUI 的 Python 构造 ``python -m ...`` 命令。"""

        from ..runner import python_module_command

        return python_module_command(module, *args)

    def add_preprocess_section(self, parent: tk.Misc) -> dict[str, object]:
        """实验波形可选预处理（训练/推理前）；z-score 在模型输入前自动执行。"""

        section = Section(parent, "实验波形预处理（训练/推理）")
        section.pack(fill="x", padx=PADX, pady=PADY)

        default_preset = self._cfg_value(
            "preprocess_preset",
            self.common.get("preprocess_preset", PreprocessPresets[0][0]),
        )
        preset = LabeledCombobox(
            section,
            "预处理预设",
            [name for name, _ in PreprocessPresets],
            default=str(default_preset),
            hint="选择后会同步填充自定义框；可继续手改顺序",
        )
        preset.pack(fill="x", padx=PADX, pady=PADY)

        custom = LabeledEntry(
            section,
            "自定义流水线",
            default=str(
                self._cfg_value(
                    "preprocess_custom", self.common.get("preprocess_custom", "")
                )
            ),
            hint="逗号分隔: clip,smooth,detrend,robust_norm（不含 zscore）",
        )
        custom.pack(fill="x", padx=PADX, pady=PADY)

        clip_q = LabeledNumber(
            section,
            "裁峰分位数",
            self._cfg_value("clip_quantile", self.common.get("clip_quantile", 1.0)),
            is_float=True,
        )
        clip_q.pack(fill="x", padx=PADX, pady=PADY)

        smooth_w = LabeledNumber(
            section,
            "平滑窗口宽度",
            int(self._cfg_value("smooth_window", self.common.get("smooth_window", 11))),
            is_float=False,
        )
        smooth_w.pack(fill="x", padx=PADX, pady=PADY)

        # 联动：选择预设后自动填充自定义框
        def _on_preset_change(_event=None) -> None:
            label = preset.get()
            for display, value in PreprocessPresets:
                if display == label:
                    custom.set(value)
                    return

        preset.combo.bind("<<ComboboxSelected>>", _on_preset_change)

        return {
            "preset": preset,
            "custom": custom,
            "clip_quantile": clip_q,
            "smooth_window": smooth_w,
        }

    def apply_runtime_to_controls(
        self, controls: dict[str, object], runtime: dict[str, Any]
    ) -> None:
        """用解析得到的训练运行参数填充 runtime 控件。"""

        controls["device"].set(str(runtime.get("device", "cuda")))  # type: ignore[union-attr]
        controls["training_mode"].set(str(runtime.get("training_mode", "normal")))  # type: ignore[union-attr]
        controls["residual_weight"].set(
            float(runtime.get("physics_residual_weight", 0.1))
        )  # type: ignore[union-attr]
        controls["learnable"].set(bool(runtime.get("learnable_branch_weights", False)))  # type: ignore[union-attr]
        for key in ("fixed_weight_cnn", "fixed_weight_lstm"):
            ctrl = controls.get(key)
            if ctrl is not None and key in runtime:
                ctrl.set(float(runtime[key]))  # type: ignore[union-attr]
        if "epochs" in controls:
            controls["epochs"].set(int(runtime.get("online_epochs", 5)))  # type: ignore[union-attr]

    def runtime_args(
        self,
        controls: dict[str, object],
        include_epochs: bool = True,
        *,
        explicit_online: bool = False,
    ) -> list[str]:
        """把 ``add_runtime_section`` 收集的控件转成 CLI 参数。"""

        args: list[str] = []
        device = str(controls["device"].get())  # type: ignore[union-attr]
        training_mode = str(controls["training_mode"].get())  # type: ignore[union-attr]
        residual = controls["residual_weight"].get()  # type: ignore[union-attr]
        learnable = bool(controls["learnable"].get())  # type: ignore[union-attr]

        if explicit_online:
            if device:
                args.extend(["--device", device])
            if training_mode:
                args.extend(["--training-mode", training_mode])
            if residual is not None:
                args.extend(["--physics-residual-weight", str(residual)])
            args.append(
                "--learnable-branch-weights"
                if learnable
                else "--no-learnable-branch-weights"
            )
            for cli_flag, key in (
                ("--fixed-weight-cnn", "fixed_weight_cnn"),
                ("--fixed-weight-lstm", "fixed_weight_lstm"),
            ):
                ctrl = controls.get(key)
                if ctrl is None:
                    continue
                value = ctrl.get()  # type: ignore[union-attr]
                if value is not None:
                    args.extend([cli_flag, str(value)])
            if include_epochs and "epochs" in controls:
                epochs = controls["epochs"].get()  # type: ignore[union-attr]
                if epochs is not None:
                    args.extend(["--epochs", str(epochs)])
            return args

        if device:
            args.extend(["--device", device])
        if training_mode:
            args.extend(["--training-mode", training_mode])
        if residual is not None:
            args.extend(["--physics-residual-weight", str(residual)])
        if learnable:
            args.append("--learnable-branch-weights")

        for cli_flag, key in (
            ("--fixed-weight-cnn", "fixed_weight_cnn"),
            ("--fixed-weight-lstm", "fixed_weight_lstm"),
        ):
            ctrl = controls.get(key)
            if ctrl is None:
                continue
            value = ctrl.get()  # type: ignore[union-attr]
            if value is not None:
                args.extend([cli_flag, str(value)])

        if include_epochs and "epochs" in controls:
            epochs = controls["epochs"].get()  # type: ignore[union-attr]
            if epochs is not None:
                args.extend(["--epochs", str(epochs)])
        return args

    def runtime_to_dict(self, controls: dict[str, object]) -> dict[str, Any]:
        """把 runtime 控件状态导出成字典（用于 to_settings_section）。"""

        out: dict[str, Any] = {
            "device": str(controls["device"].get()),  # type: ignore[union-attr]
            "training_mode": str(controls["training_mode"].get()),  # type: ignore[union-attr]
            "physics_residual_weight": controls["residual_weight"].get(),  # type: ignore[union-attr]
            "learnable_branch_weights": bool(controls["learnable"].get()),  # type: ignore[union-attr]
        }
        for key in ("fixed_weight_cnn", "fixed_weight_lstm"):
            ctrl = controls.get(key)
            if ctrl is not None:
                out[key] = ctrl.get()  # type: ignore[union-attr]
        if "epochs" in controls:
            out["epochs"] = controls["epochs"].get()  # type: ignore[union-attr]
        return out

    def preprocess_args(
        self,
        controls: dict[str, object],
        *,
        include_empty: bool = False,
    ) -> list[str]:
        """把 ``add_preprocess_section`` 收集的控件转成 CLI 参数。"""

        args: list[str] = []
        custom = str(controls["custom"].get()).strip()  # type: ignore[union-attr]
        if custom or include_empty:
            args.extend(["--preprocess", custom])
        clip_q = controls["clip_quantile"].get()  # type: ignore[union-attr]
        if clip_q is not None:
            args.extend(["--clip-quantile", str(clip_q)])
        smooth_w = controls["smooth_window"].get()  # type: ignore[union-attr]
        if smooth_w is not None:
            args.extend(["--smooth-window", str(smooth_w)])
        return args

    def preprocess_to_dict(self, controls: dict[str, object]) -> dict[str, Any]:
        return {
            "preprocess_preset": str(controls["preset"].get()),  # type: ignore[union-attr]
            "preprocess_custom": str(controls["custom"].get()),  # type: ignore[union-attr]
            "clip_quantile": controls["clip_quantile"].get(),  # type: ignore[union-attr]
            "smooth_window": controls["smooth_window"].get(),  # type: ignore[union-attr]
        }

    # ---- 默认值取值优先级 --------------------------------------------------
    def _cfg_value(self, key: str, fallback: Any) -> Any:
        """优先用本 Tab 自己的 section，其次 common，再次 fallback。"""

        if key in self.local:
            return self.local[key]
        if key in self.common:
            return self.common[key]
        return fallback

    # ---- 主流程「路径设置」Tab 共用控件 ----------------------------------
    def _path_io_roots(self) -> dict[str, object]:
        ps = self._path_settings
        if ps is not None and getattr(ps, "io_roots", None) is not None:
            return ps.io_roots  # type: ignore[return-value]
        roots = getattr(self, "io_roots", None)
        if roots is not None:
            return roots
        raise ValueError("缺少输入/输出根目录：请在「主流程 → 路径设置」中填写。")

    def shared_io_root_args(self) -> list[str]:
        return self.io_root_args(self._path_io_roots())

    def _resolve_data_root_path(self) -> Path:
        roots = self._path_io_roots()
        raw = str(roots["data_root"].get() or "").strip()  # type: ignore[union-attr]
        return resolve_gui_project_path(
            raw or "database",
            repo_root=self.repo_root,
        )

    def _resolve_result_root_path(self) -> Path:
        roots = self._path_io_roots()
        raw = str(roots["result_root"].get() or "").strip()  # type: ignore[union-attr]
        return resolve_gui_project_path(
            raw or "result",
            repo_root=self.repo_root,
        )

    def resolve_latest_split_manifest(self, kind: str) -> str:
        if kind not in {"train", "test", "combined"}:
            return ""
        try:
            path = latest_split_manifest(self._resolve_data_root_path(), kind)
            return str(path) if path is not None else ""
        except Exception:
            return ""

    def resolve_latest_material_collection(self) -> str:
        try:
            path = latest_material_collection(self._resolve_data_root_path())
            return str(path) if path is not None else ""
        except Exception:
            return ""

    # ---- 内部按钮事件 ------------------------------------------------------
    def show_validation_error(self, error: ValueError) -> None:
        self._feedback.set(str(error))
        self._feedback_label.pack(fill="x", before=self._button_row, pady=(0, 8))
        if isinstance(error, FormValidationError):
            self.scroll.reveal(error.widget)
            error.widget.focus_set()

    def clear_validation_error(self) -> None:
        self._feedback.set("")
        self._feedback_label.pack_forget()

    def _prepare_command(self) -> list[str] | None:
        self.clear_validation_error()
        try:
            self.validate_form()
            return self.compose_command()
        except ValueError as exc:
            self.show_validation_error(exc)
        except Exception as exc:  # pragma: no cover
            messagebox.showerror("内部错误", str(exc), parent=self.winfo_toplevel())
        return None

    def _on_run(self) -> None:
        if not self._busy:
            command = self._prepare_command()
            if command is not None:
                self._run_cb(command)

    def _on_preview(self) -> None:
        from ..dialogs import show_command_preview
        from ...paths import resolve_project_path

        command = self._prepare_command()
        if command is not None:
            paths = self._path_settings
            raw = (
                paths.launch_env["launch_root"].get()
                if paths is not None
                else self.common.get("launch_root", "")
            )
            cwd = (
                resolve_project_path(raw, repo_root=self.repo_root)
                if raw
                else Path(self.repo_root).parent
            )
            show_command_preview(self.winfo_toplevel(), self.title, command, cwd)

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        if self._stop_button is not None:
            self._run_button.configure(state="disabled" if busy else "normal")
            self._stop_button.configure(state="normal" if busy else "disabled")

    def _resolve_source_path(self, value: str) -> Path:
        return resolve_source_path(
            value, data_root=self._resolve_data_root_path(), repo_root=self.repo_root
        )

    def _on_stop(self) -> None:
        if self._busy and messagebox.askyesno(
            "停止任务", "确定要停止当前任务吗？", parent=self.winfo_toplevel()
        ):
            self._stop_cb()
