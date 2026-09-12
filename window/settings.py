"""统一的 GUI 默认值配置中心。

所有 Tab 的初始表单值、fixed 权重默认值、预处理流水线预设等，
都集中在 ``settings.json`` 里维护。GUI 启动时加载该文件作为各表单
的默认值；用户也可以在窗口里点 “保存当前表单为默认值” 写回去，
或直接用编辑器修改 ``settings.json``。

设计要点
--------
1. ``DEFAULT_SETTINGS`` 是内置默认 schema，作为兜底——即使
   ``settings.json`` 缺失或字段被改坏，GUI 仍能正常启动；
2. ``load_settings`` 做深度合并：用户文件里有的覆盖默认，
   缺的用默认补齐，避免“漏配置就崩”的情况；
3. ``Settings.save`` 把当前 dict 以 UTF-8 + 2 空格缩进写回 JSON，
   方便人手工编辑；
4. 每个 section 的 key 与对应 Tab 的控件一一对应，键名建议保持
   和 CLI 参数 ``--xxx`` 中下划线风格一致，方便对照。
"""

from __future__ import annotations

import copy
import json
import os
import tempfile
from pathlib import Path
from typing import Any


SETTINGS_FILE_NAME = "settings.json"
SETTINGS_PATH = Path(__file__).resolve().parent / SETTINGS_FILE_NAME


# ---------------------------------------------------------------------------
# 内置默认 schema：与 ai_model/cli.py、ai_model/config/、ai_model/data_process/、ai_model/batch/
# 中的默认值一一对齐；用户可以修改 settings.json 来改变这些默认。
# ---------------------------------------------------------------------------
DEFAULT_SETTINGS: dict[str, Any] = {
    "$schema_version": 1,
    "common": {
        "data_root": "database",
        "result_root": "result",
        "device": "cuda",
        "training_mode": "normal",
        "physics_residual_weight": 0.1,
        "learnable_branch_weights": False,
        "fixed_weight_cnn": 0.75,
        "fixed_weight_lstm": 0.75,
        "preprocess_preset": "自定义 / 不预处理",
        "preprocess_custom": "",
        "clip_quantile": 1.0,
        "smooth_window": 11,
        "epochs": 20,
        "launch_root": "",
    },
    "build_db": {
        "data_root": "database",
        "result_root": "result",
        "sim_per_material": 400,
        "skip_simulation": True,
        "experiment_dir": "raw/wumu",
        "experiment_material": "wumu",
        "multi_material_input": False,
        "material_splits": {},
        "experiment_limit": 2000,
        "waveform_crop_length": 1097,
        "external_sim_dir": "",
        "external_sim_material": "",
        "external_sim_limit": -1,
        "external_test_enabled": False,
        "external_test_dir": "",
        "external_test_material": "wumu",
        "external_test_dataset_name": "wumu_exp",
        "external_test_signal_column": "amplitude_filtered_residual",
        "external_test_time_column": "time_s",
        "external_test_time_min_s": 0.0,
        "external_test_limit": -1,
        "split_dataset": True,
        "split_test_ratio": 0.2,
        "split_validation_ratio": 0.1,
        "split_seed": 42,
        "split_experiment_policy": "uniform",
        "train_manifest_name": "train_manifest.json",
        "validation_manifest_name": "validation_manifest.json",
        "test_manifest_name": "test_manifest.json",
    },
    "train": {
        "data_root": "database",
        "result_root": "result",
        "preprocess_preset": "自定义 / 不预处理",
        "preprocess_custom": "",
        "clip_quantile": 1.0,
        "smooth_window": 11,
        "manifest": "database",
        "auto_manifest": False,
        "train_name": "",
        "checkpoint_name": "ai_model.pt",
        "separate_materials": False,
        "rule_dimension": "one",
        "rule_mode": "steady",
        "rule_material": "metal_matrix",
        "epochs": 20,
        "early_stopping_patience": 10,
    },
    "predict": {
        "data_root": "database",
        "result_root": "result",
        "device": "",
        "preprocess_preset": "自定义 / 不预处理",
        "preprocess_custom": "",
        "clip_quantile": 1.0,
        "smooth_window": 11,
        "manifest": "database/test_manifest.json",
        "auto_manifest": False,
        "checkpoint": "result/train/checkpoint/ai_model/ai_model.pt",
        "auto_material_routing": False,
        "material_router": "",
        "override_preprocess": False,
        "rule_dimension": "one",
        "rule_mode": "steady",
        "rule_material": "",
        "output_dir": "",
        "predict_name": "",
        "prediction_dimension": "two",
        "enable_plots": False,
        "num_field_samples": 6,
        "enable_benchmark": False,
        "benchmark_warmup_samples": 64,
        "benchmark_runs": 3,
    },
    "validate": {
        "data_root": "database",
        "result_root": "result",
        "manifest": "database/combined_manifest.json",
        "auto_manifest": False,
    },
    "online_update": {
        "data_root": "database",
        "result_root": "result",
        "preprocess_preset": "自定义 / 不预处理",
        "preprocess_custom": "",
        "clip_quantile": 1.0,
        "smooth_window": 11,
        "manifest": "database/combined_manifest.json",
        "auto_manifest": False,
        "checkpoint": "",
        "override_preprocess": False,
        "rule_dimension": "one",
        "rule_mode": "steady",
        "rule_material": "",
        "output_name": "",
        "epochs": 5,
    },
    "demo": {
        "data_root": "database",
        "result_root": "result",
        "sim_per_material": 400,
        "experiment_dir": "raw/wumu",
        "multi_material_input": False,
        "material_splits": {},
        "experiment_limit": 2000,
        "experiment_material": "wumu",
        "waveform_crop_length": 1097,
        "train_name": "",
        "separate_materials": False,
        "rule_dimension": "two",
        "rule_mode": "steady",
        "rule_material": "metal_matrix",
        "epochs": 20,
        "split_dataset": True,
        "split_test_ratio": 0.2,
        "split_validation_ratio": 0.1,
        "split_seed": 42,
        "split_experiment_policy": "uniform",
        "train_manifest_name": "train_manifest.json",
        "validation_manifest_name": "validation_manifest.json",
        "test_manifest_name": "test_manifest.json",
    },
    "batch_modes": {
        "data_root": "database/raw",
        "result_root": "result",
        "test_ratio": 0.2,
        "epochs": 5000,
        "device": "cuda",
        "seed": 42,
        "physics_residual_weight": 0.1,
    },
    "batch_preprocess": {
        "data_root": "database/raw",
        "result_root": "result",
        "test_ratio": 0.2,
        "epochs": 5000,
        "device": "cuda",
        "seed": 42,
        "physics_residual_weight": 0.1,
        "clip_quantile": 1.0,
        "smooth_window": 11,
        "pipelines": "base;clip,smooth;clip,smooth,detrend;smooth,robust_norm",
        "resume_run_dir": "",
    },
    "search_weights": {
        "data_root": "database/raw",
        "result_root": "result",
        "test_ratio": 0.2,
        "epochs": 1000,
        "device": "cuda",
        "seed": 42,
        "training_mode": "residual_pinn",
        "physics_residual_weight": 0.1,
        "network_weights": "0.75,1.0,1.25",
    },
    "plot_loss": {
        "run_dir": "",
        "output": "",
        "dpi": 150,
    },
    "rerun_predict": {
        "run_dir": "",
        "device": "cuda",
        "num_field_samples": 6,
        "overwrite": False,
    },
    "result_browser": {
        "root_dir": "result",
    },
    "manage_artifacts": {
        "data_root": "database",
        "result_root": "result",
        "rule_dimension": "one",
        "rule_mode": "steady",
        "rule_material": "",
        "checkpoint": "",
    },
}


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深度合并：``override`` 中已有的键覆盖 ``base``，否则保留 ``base``。"""

    result = copy.deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


class Settings:
    """对 ``settings.json`` 的轻量包装，提供 section/key 取值与回写。"""

    def __init__(self, data: dict[str, Any] | None = None, path: Path | None = None) -> None:
        self._path = Path(path) if path is not None else SETTINGS_PATH
        self._data = copy.deepcopy(DEFAULT_SETTINGS if data is None else data)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def data(self) -> dict[str, Any]:
        return self._data

    # ---- 取值 --------------------------------------------------------------
    def section(self, name: str) -> dict[str, Any]:
        """返回隔离的 section 副本；调用方修改表单数据不会更改默认值。"""

        section = self._data.get(name, {})
        if not isinstance(section, dict):
            section = {}
        # 用 DEFAULT_SETTINGS 兜底，缺失字段也能拿到默认值。
        default = DEFAULT_SETTINGS.get(name, {})
        merged = _deep_merge(default if isinstance(default, dict) else {}, section)
        return dict(merged)

    def get(self, section: str, key: str, default: Any = None) -> Any:
        sec = self.section(section)
        return sec.get(key, default)

    # ---- 写值 --------------------------------------------------------------
    def update_section(self, name: str, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            raise TypeError("payload 必须是 dict")
        existing = self._data.get(name, {})
        if not isinstance(existing, dict):
            existing = {}
        existing.update(copy.deepcopy(payload))
        self._data[name] = existing

    # ---- IO ----------------------------------------------------------------
    def save(self, path: Path | None = None) -> Path:
        target = Path(path) if path is not None else self._path
        _write_settings(target, self._data)
        return target

    def save_sections(self, sections: dict[str, dict[str, Any]]) -> Path:
        """Commit collected forms only after the entire file is safely saved."""
        candidate = Settings(self._data, path=self._path)
        for name, payload in sections.items():
            candidate.update_section(name, payload)
        target = candidate.save()
        self._data = candidate.data
        return target

    def reload(self) -> None:
        loaded = load_settings(self._path)
        self._data = loaded.data


def load_settings(path: Path | None = None) -> Settings:
    """加载配置文件并与默认 schema 深度合并。文件缺失时直接返回内置默认。"""

    target = Path(path) if path is not None else SETTINGS_PATH
    if not target.exists():
        return Settings(copy.deepcopy(DEFAULT_SETTINGS), path=target)
    try:
        loaded = json.loads(target.read_text(encoding="utf-8-sig"))
        if not isinstance(loaded, dict):
            raise ValueError("settings.json 顶层必须是 JSON 对象")
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        # 解析失败时回退到默认，避免 GUI 因为坏配置启动失败。
        print(f"[settings] 解析失败，使用内置默认值: {exc}")
        return Settings(copy.deepcopy(DEFAULT_SETTINGS), path=target)
    merged = _deep_merge(copy.deepcopy(DEFAULT_SETTINGS), loaded)
    return Settings(merged, path=target)


def write_default_settings_file(path: Path | None = None, force: bool = False) -> Path:
    """Write the built-in settings, preserving an existing file by default."""
    target = Path(path) if path is not None else SETTINGS_PATH
    if target.exists() and not force:
        return target
    _write_settings(target, DEFAULT_SETTINGS)
    return target


def _write_settings(target: Path, data: dict[str, Any]) -> None:
    """Replace a complete UTF-8 file; a failed save leaves the previous file intact."""
    content = json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=target.parent,
            prefix=f".{target.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
