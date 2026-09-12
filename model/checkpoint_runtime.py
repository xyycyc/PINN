from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ..config import AIModelConfig
from .rule_registry import find_rule_row_for_checkpoint

RUNTIME_CONFIG_KEYS: tuple[str, ...] = (
    "training_mode",
    "physics_residual_weight",
    "learnable_branch_weights",
    "fixed_weight_cnn",
    "fixed_weight_lstm",
    "fixed_weight_material",
    "fixed_weight_dimension",
    "fixed_weight_mode",
    "online_epochs",
    "device",
)

MODEL_CONFIG_KEYS: tuple[str, ...] = (
    "waveform_length",
    "hidden_dim",
    "latent_dim",
    "field_grid_1d",
    "field_grid_2d",
    "batch_size",
)

PREPROCESS_CONFIG_KEYS: tuple[str, ...] = (
    "preprocess_steps",
    "clip_quantile",
    "smooth_window",
)

INFERENCE_CONFIG_KEYS: tuple[str, ...] = RUNTIME_CONFIG_KEYS + MODEL_CONFIG_KEYS + PREPROCESS_CONFIG_KEYS


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"cannot parse bool: {value!r}")


def default_training_runtime() -> dict[str, Any]:
    cfg = AIModelConfig()
    return {
        "training_mode": cfg.training_mode,
        "physics_residual_weight": float(cfg.physics_residual_weight),
        "learnable_branch_weights": bool(cfg.learnable_branch_weights),
        "fixed_weight_cnn": float(cfg.fixed_weight_cnn),
        "fixed_weight_lstm": float(cfg.fixed_weight_lstm),
        "fixed_weight_material": float(cfg.fixed_weight_material),
        "fixed_weight_dimension": float(cfg.fixed_weight_dimension),
        "fixed_weight_mode": float(cfg.fixed_weight_mode),
        "online_epochs": int(cfg.online_epochs),
        "device": str(cfg.device),
    }


def resolve_checkpoint_config_dict(
    checkpoint_path: Path | str,
    *,
    rule_csv_path: Path | str | None = None,
) -> dict[str, Any]:
    """从 checkpoint（及增量训练的 base_checkpoint 链）读取完整 ``config`` 字典。"""

    start = Path(checkpoint_path).resolve()
    if not start.exists():
        raise FileNotFoundError(f"checkpoint not found: {start}")

    visited: set[Path] = set()
    current = start
    while current not in visited:
        visited.add(current)
        bundle = torch.load(current, map_location="cpu", weights_only=False)
        config_dict = bundle.get("config") if isinstance(bundle, dict) else None
        if isinstance(config_dict, dict):
            return dict(config_dict)
        base = bundle.get("base_checkpoint") if isinstance(bundle, dict) else None
        if base:
            base_path = Path(str(base))
            current = (base_path if base_path.is_absolute() else current.parent / base_path).resolve()
            if current.exists():
                continue
        break

    if rule_csv_path is not None:
        row = find_rule_row_for_checkpoint(rule_csv_path, start)
        if row is not None:
            merged = default_training_runtime()
            merged.update(runtime_from_rule_row(row))
            return merged
    return {}


def runtime_from_config_dict(config_dict: dict[str, Any]) -> dict[str, Any]:
    runtime = default_training_runtime()
    for key in RUNTIME_CONFIG_KEYS:
        if key not in config_dict:
            continue
        value = config_dict[key]
        if value is None:
            continue
        if key == "learnable_branch_weights":
            runtime[key] = _parse_bool(value)
        elif key == "online_epochs":
            runtime[key] = int(value)
        elif key == "device":
            runtime[key] = str(value)
        elif key == "training_mode":
            runtime[key] = str(value)
        elif key == "physics_residual_weight":
            runtime[key] = float(value)
        elif key.startswith("fixed_weight_"):
            runtime[key] = float(value)
    return runtime


def runtime_from_rule_row(row: dict[str, str]) -> dict[str, Any]:
    runtime = default_training_runtime()
    if mode := str(row.get("training_mode", "")).strip():
        runtime["training_mode"] = mode
    if raw := str(row.get("physics_residual_weight", "")).strip():
        runtime["physics_residual_weight"] = float(raw)
    if raw := str(row.get("learnable_branch_weights", "")).strip():
        runtime["learnable_branch_weights"] = _parse_bool(raw)
    for key in (
        "fixed_weight_cnn",
        "fixed_weight_lstm",
        "fixed_weight_material",
        "fixed_weight_dimension",
        "fixed_weight_mode",
    ):
        if raw := str(row.get(key, "")).strip():
            runtime[key] = float(raw)
    return runtime


def resolve_training_runtime(
    checkpoint_path: Path | str,
    *,
    rule_csv_path: Path | str | None = None,
) -> dict[str, Any]:
    """Resolve training runtime settings from checkpoint metadata, with rule-CSV fallback."""

    config_dict = resolve_checkpoint_config_dict(checkpoint_path, rule_csv_path=rule_csv_path)
    if config_dict:
        return runtime_from_config_dict(config_dict)
    return default_training_runtime()


def apply_training_runtime(config: AIModelConfig, runtime: dict[str, Any]) -> None:
    config.training_mode = str(runtime["training_mode"])
    config.physics_residual_weight = float(runtime["physics_residual_weight"])
    config.learnable_branch_weights = bool(runtime["learnable_branch_weights"])
    config.fixed_weight_cnn = float(runtime["fixed_weight_cnn"])
    config.fixed_weight_lstm = float(runtime["fixed_weight_lstm"])
    config.fixed_weight_material = float(runtime["fixed_weight_material"])
    config.fixed_weight_dimension = float(runtime["fixed_weight_dimension"])
    config.fixed_weight_mode = float(runtime["fixed_weight_mode"])
    config.online_epochs = int(runtime["online_epochs"])
    config.device = str(runtime["device"])


def _apply_preprocess_config(config: AIModelConfig, config_dict: dict[str, Any]) -> None:
    if "preprocess_steps" in config_dict and config_dict["preprocess_steps"] is not None:
        config.preprocess_steps = str(config_dict["preprocess_steps"])
    if "clip_quantile" in config_dict and config_dict["clip_quantile"] is not None:
        config.clip_quantile = float(config_dict["clip_quantile"])
    if "smooth_window" in config_dict and config_dict["smooth_window"] is not None:
        config.smooth_window = int(config_dict["smooth_window"])


def _apply_model_config(config: AIModelConfig, config_dict: dict[str, Any]) -> None:
    for key in ("waveform_length", "hidden_dim", "latent_dim", "field_grid_1d", "batch_size"):
        if key not in config_dict or config_dict[key] is None:
            continue
        setattr(config, key, int(config_dict[key]))
    if "field_grid_2d" in config_dict and config_dict["field_grid_2d"] is not None:
        grid = config_dict["field_grid_2d"]
        if isinstance(grid, (list, tuple)) and len(grid) >= 2:
            config.field_grid_2d = (int(grid[0]), int(grid[1]))


def apply_inference_config(
    config: AIModelConfig,
    config_dict: dict[str, Any],
) -> None:
    """将 checkpoint 中记录的训练配置同步到推理用 ``AIModelConfig``。"""
    if not config_dict:
        return
    apply_training_runtime(config, runtime_from_config_dict(config_dict))
    _apply_model_config(config, config_dict)
    _apply_preprocess_config(config, config_dict)


def sync_config_for_inference(
    config: AIModelConfig,
    checkpoint_path: Path | str,
    *,
    rule_csv_path: Path | str | None = None,
) -> dict[str, Any]:
    """从 checkpoint 恢复推理所需全部配置，返回读到的 config 字典。"""
    config_dict = resolve_checkpoint_config_dict(checkpoint_path, rule_csv_path=rule_csv_path)
    if not config_dict:
        raise ValueError(
            f"checkpoint 中缺少 config 元数据，无法保证与训练一致: {Path(checkpoint_path).resolve()}"
        )
    apply_inference_config(config, config_dict)
    return config_dict


def load_model_state_strict(
    model: torch.nn.Module,
    checkpoint_path: Path | str,
    *,
    map_location: torch.device | str | None = None,
) -> dict[str, Any]:
    """加载权重；默认 strict=True，结构不一致时立即报错。"""
    checkpoint_path = Path(checkpoint_path)
    bundle = torch.load(checkpoint_path, map_location=map_location or "cpu", weights_only=False)
    if not isinstance(bundle, dict) or "model_state" not in bundle:
        raise ValueError(f"无效的 checkpoint 格式: {checkpoint_path}")
    model.load_state_dict(bundle["model_state"], strict=True)
    return bundle
