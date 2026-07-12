"""Direct fixed-node temperature predictor and Kelvin-domain evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .network import ConvEncoder, LSTMEncoder

POINT_FIELD_MODEL_VERSION = 1


class DirectPointFieldModel(nn.Module):
    """Backward-independent model for P fixed nodes, with chunked output heads."""
    def __init__(self, point_count: int, hidden_dim: int = 128, latent_dim: int = 128,
                 chunk_size: int = 1000):
        super().__init__()
        if not 1 <= point_count <= 10000:
            raise ValueError("point_count 必须在 [1, 10000]")
        self.point_count, self.chunk_size = point_count, chunk_size
        self.encoder, self.lstm_encoder = ConvEncoder(hidden_dim=hidden_dim), LSTMEncoder(hidden_dim=hidden_dim)
        self.backbone = nn.Sequential(nn.Linear(hidden_dim * 2, latent_dim), nn.GELU(),
                                      nn.Linear(latent_dim, latent_dim), nn.GELU())
        self.heads = nn.ModuleList(nn.Linear(latent_dim, min(chunk_size, point_count-i))
                                   for i in range(0, point_count, chunk_size))

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        latent = self.backbone(torch.cat((self.encoder(waveform), self.lstm_encoder(waveform)), dim=-1))
        return torch.cat([head(latent) for head in self.heads], dim=-1)


def weighted_temperature_loss(prediction: torch.Tensor, target: torch.Tensor,
                              sample_weights: torch.Tensor) -> torch.Tensor:
    weights = sample_weights / sample_weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return (((prediction - target) ** 2) * weights).sum(dim=-1).mean()


def kelvin_metrics(prediction_k: np.ndarray, target_k: np.ndarray, *,
                   masks: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    error = np.asarray(prediction_k, float) - np.asarray(target_k, float)
    def one(values: np.ndarray) -> dict[str, float]:
        return {"mae_k": float(np.mean(np.abs(values))), "rmse_k": float(np.sqrt(np.mean(values**2))),
                "max_absolute_error_k": float(np.max(np.abs(values)))}
    result: dict[str, Any] = {"full_field": one(error)}
    for name, mask in (masks or {}).items():
        selected = error[..., np.asarray(mask, bool)]
        result[name] = one(selected) if selected.size else None
    return result


@dataclass(frozen=True)
class TemperatureNormalizer:
    mean_k: float
    std_k: float

    def normalize(self, value: np.ndarray) -> np.ndarray:
        return (value - self.mean_k) / max(self.std_k, 1e-12)

    def denormalize(self, value: np.ndarray) -> np.ndarray:
        return value * self.std_k + self.mean_k
