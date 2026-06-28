from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from .preprocess import apply_optional_preprocessing

_CACHE_VERSION = 1


def preprocess_pipeline_label(
    preprocess_steps: list[str] | tuple[str, ...] | None,
    clip_quantile: float,
    smooth_window: int,
) -> str:
    """生成可读的流水线目录名（用于 cache/preprocess/<label>/）。"""
    steps = [str(item).strip().lower() for item in (preprocess_steps or []) if str(item).strip()]
    if not steps:
        base = "base"
    else:
        base = "_".join(steps)
    clip = float(clip_quantile)
    smooth = int(smooth_window)
    return f"{base}_cq{clip:g}_sw{smooth}"


def _sanitize_label(label: str) -> str:
    text = re.sub(r"[^\w.\-]+", "_", str(label).strip())
    return text.strip("._") or "pipeline"


def preprocess_cache_dir(cache_root: Path, pipeline_label: str) -> Path:
    return cache_root / "preprocess" / _sanitize_label(pipeline_label)


def _meta_path(cache_file: Path) -> Path:
    return cache_file.with_suffix(".meta.json")


def _read_meta(meta_path: Path) -> dict[str, Any] | None:
    if not meta_path.is_file():
        return None
    try:
        payload = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _raw_file_signature(raw_path: Path) -> dict[str, Any]:
    stat = raw_path.stat()
    return {
        "mtime_ns": int(stat.st_mtime_ns),
        "size": int(stat.st_size),
    }


def _meta_matches_raw(meta: dict[str, Any], raw_path: Path, pipeline_label: str) -> bool:
    if int(meta.get("cache_version", 0)) != _CACHE_VERSION:
        return False
    if str(meta.get("pipeline_label", "")) != pipeline_label:
        return False
    if str(meta.get("raw_path", "")) != str(raw_path.resolve()):
        return False
    try:
        current = _raw_file_signature(raw_path)
    except OSError:
        return False
    return (
        int(meta.get("mtime_ns", -1)) == current["mtime_ns"]
        and int(meta.get("size", -1)) == current["size"]
    )


def _cache_file_path(cache_dir: Path, sample_id: str) -> Path:
    safe_id = _sanitize_label(sample_id) or "sample"
    return cache_dir / f"{safe_id}.npy"


def load_cached_preprocessed_waveform(
    cache_dir: Path,
    *,
    sample_id: str,
    raw_waveform_path: Path,
    pipeline_label: str,
) -> np.ndarray | None:
    """若缓存存在且与原始 npy 一致，则返回缓存波形。"""
    cache_file = _cache_file_path(cache_dir, sample_id)
    meta_path = _meta_path(cache_file)
    meta = _read_meta(meta_path)
    if meta is None or not cache_file.is_file():
        return None
    if not _meta_matches_raw(meta, raw_waveform_path, pipeline_label):
        return None
    try:
        return np.load(cache_file).astype(np.float32).reshape(-1)
    except OSError:
        return None


def save_preprocessed_waveform_cache(
    cache_dir: Path,
    *,
    sample_id: str,
    raw_waveform_path: Path,
    pipeline_label: str,
    waveform: np.ndarray,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = _cache_file_path(cache_dir, sample_id)
    values = np.asarray(waveform, dtype=np.float32).reshape(-1)
    np.save(cache_file, values)
    signature = _raw_file_signature(raw_waveform_path)
    meta = {
        "cache_version": _CACHE_VERSION,
        "pipeline_label": pipeline_label,
        "sample_id": sample_id,
        "raw_path": str(raw_waveform_path.resolve()),
        "mtime_ns": signature["mtime_ns"],
        "size": signature["size"],
    }
    _meta_path(cache_file).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return cache_file


def load_or_apply_preprocessed_waveform(
    waveform_raw: np.ndarray,
    *,
    raw_waveform_path: Path,
    sample_id: str,
    cache_root: Path,
    preprocess_steps: list[str] | tuple[str, ...] | None,
    clip_quantile: float,
    smooth_window: int,
    use_cache: bool = True,
) -> np.ndarray:
    """可选预处理 + ``database/cache/preprocess`` 磁盘缓存（不含 z-score）。"""
    pipeline_label = preprocess_pipeline_label(preprocess_steps, clip_quantile, smooth_window)
    steps = [str(item).strip().lower() for item in (preprocess_steps or []) if str(item).strip()]
    cache_enabled = bool(use_cache and steps)

    cache_dir = preprocess_cache_dir(cache_root, pipeline_label)

    if cache_enabled:
        cached = load_cached_preprocessed_waveform(
            cache_dir,
            sample_id=sample_id,
            raw_waveform_path=raw_waveform_path,
            pipeline_label=pipeline_label,
        )
        if cached is not None:
            return cached

    processed = apply_optional_preprocessing(
        waveform_raw,
        preprocess_steps=preprocess_steps,
        clip_quantile=clip_quantile,
        smooth_window=smooth_window,
    )

    if cache_enabled:
        save_preprocessed_waveform_cache(
            cache_dir,
            sample_id=sample_id,
            raw_waveform_path=raw_waveform_path,
            pipeline_label=pipeline_label,
            waveform=processed,
        )
    return processed
