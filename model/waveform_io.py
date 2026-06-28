from __future__ import annotations

import torch


def zscore_waveform_batch(waveform: torch.Tensor, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """对波形 [B, C, L] 按样本做 z-score，返回 (normalized, mean, std)。"""
    if waveform.dim() != 3:
        raise ValueError(f"waveform must be [B, C, L], got shape {tuple(waveform.shape)}")
    reduce_dims = tuple(range(1, waveform.dim()))
    mean = waveform.mean(dim=reduce_dims, keepdim=True)
    std = waveform.std(dim=reduce_dims, keepdim=True).clamp_min(eps)
    normalized = (waveform - mean) / std
    return normalized, mean, std


def inverse_zscore_waveform_batch(
    waveform: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """与 ``zscore_waveform_batch`` 配对的反标准化。"""
    return waveform * std + mean


def prepare_model_waveform_input(waveform: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """模型输入前强制 z-score。"""
    return zscore_waveform_batch(waveform)


def postprocess_model_outputs(
    outputs: dict[str, torch.Tensor],
    waveform_mean: torch.Tensor,
    waveform_std: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """模型输出后：对波形重建类输出做与输入对应的 z-score 反标准化。"""
    if "waveform" not in outputs:
        return outputs
    result = dict(outputs)
    result["waveform"] = inverse_zscore_waveform_batch(
        result["waveform"],
        waveform_mean,
        waveform_std,
    )
    return result
