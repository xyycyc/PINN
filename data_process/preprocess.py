from __future__ import annotations

import numpy as np

SUPPORTED_PREPROCESS_STEPS = {
    "clip",
    "smooth",
    "detrend",
    "robust_norm",
}

# z-score 在模型输入前强制应用，不作为可选流水线步骤。
FORCED_MODEL_ZSCORE_STEP = "zscore"


def parse_preprocess_steps(raw: str | None) -> list[str]:
    if raw is None:
        return []
    steps = [item.strip().lower() for item in str(raw).split(",") if item.strip()]
    steps = [item for item in steps if item != FORCED_MODEL_ZSCORE_STEP]
    invalid = [item for item in steps if item not in SUPPORTED_PREPROCESS_STEPS]
    if invalid:
        raise ValueError(
            f"未知预处理步骤: {invalid}，支持: {sorted(SUPPORTED_PREPROCESS_STEPS)}；"
            f"z-score 在模型输入前自动执行，请勿写入 --preprocess"
        )
    return steps


def clip_outliers(values: np.ndarray, q_low: float = 1.0, q_high: float = 99.0) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    lo = float(np.percentile(x, q_low))
    hi = float(np.percentile(x, q_high))
    return np.clip(x, lo, hi).astype(np.float32)


def smooth_signal(values: np.ndarray, window: int = 11) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if window < 3:
        return x
    w = int(window)
    if w % 2 == 0:
        w += 1
    kernel = np.ones(w, dtype=np.float32) / float(w)
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def detrend_baseline(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.size < 2:
        return x
    axis = np.linspace(0.0, 1.0, x.size, dtype=np.float32)
    trend = np.polyval(np.polyfit(axis, x, 1), axis).astype(np.float32)
    return (x - trend).astype(np.float32)


def robust_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    med = float(np.median(x))
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    if iqr < eps:
        iqr = eps
    return ((x - med) / iqr).astype(np.float32)


def zscore_normalize(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    std = float(x.std())
    if std > eps:
        return ((x - float(x.mean())) / std).astype(np.float32)
    return x


def zscore_stats(values: np.ndarray, eps: float = 1e-8) -> tuple[float, float]:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    mean = float(x.mean())
    std = float(x.std())
    if std < eps:
        std = 1.0
    return mean, std


def apply_zscore_with_stats(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    return ((x - float(mean)) / float(std)).astype(np.float32)


def inverse_zscore(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    return (x * float(std) + float(mean)).astype(np.float32)


def apply_optional_preprocessing(
    values: np.ndarray,
    preprocess_steps: list[str] | tuple[str, ...] | None = None,
    clip_quantile: float = 1.0,
    smooth_window: int = 11,
) -> np.ndarray:
    """可选预处理（不含 z-score），用于建库后、进入模型前的波形处理。"""
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    steps = [step.lower() for step in (preprocess_steps or []) if step.lower() != FORCED_MODEL_ZSCORE_STEP]
    for step in steps:
        if step == "clip":
            x = clip_outliers(x, q_low=clip_quantile, q_high=100.0 - clip_quantile)
        elif step == "smooth":
            x = smooth_signal(x, window=smooth_window)
        elif step == "detrend":
            x = detrend_baseline(x)
        elif step == "robust_norm":
            x = robust_normalize(x)
    return x.astype(np.float32)


def apply_preprocessing_pipeline(
    values: np.ndarray,
    preprocess_steps: list[str] | tuple[str, ...] | None = None,
    clip_quantile: float = 1.0,
    smooth_window: int = 11,
) -> np.ndarray:
    """向后兼容：仅执行可选步骤，不再隐式附加 z-score。"""
    return apply_optional_preprocessing(
        values,
        preprocess_steps=preprocess_steps,
        clip_quantile=clip_quantile,
        smooth_window=smooth_window,
    )
