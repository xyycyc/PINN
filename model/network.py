"""CNN/LSTM encoders and the legacy multi-head reconstruction network."""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvEncoder(nn.Module):
    """Encode local waveform patterns with a compact one-dimensional CNN."""

    def __init__(self, in_channels: int = 1, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, hidden_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return one pooled CNN feature vector per waveform."""
        x = self.net(waveform)
        return x.squeeze(-1)


class LSTMEncoder(nn.Module):
    """Encode ordered waveform evolution with the final LSTM hidden state."""

    def __init__(self, input_dim: int = 1, hidden_dim: int = 128, num_layers: int = 1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Return the last-layer hidden state for each input waveform."""
        # Conv1d uses [B, C, L], while LSTM expects [B, L, C].
        sequence = waveform.transpose(1, 2)
        _, (hidden, _) = self.lstm(sequence)
        return hidden[-1]


class AIReconstructionModel(nn.Module):
    """
    Unified model:
    - waveform encoder (cnn + lstm)
    - temperature field decoder
    - acoustic parameter head
    """

    def __init__(
        self,
        waveform_length: int,
        hidden_dim: int = 128,
        latent_dim: int = 128,
        field_height: int = 24,
        field_width: int = 24,
        learnable_branch_weights: bool = False,
        fixed_weight_cnn: float = 1.0,
        fixed_weight_lstm: float = 1.0,
    ):
        super().__init__()
        self.field_height = field_height
        self.field_width = field_width
        self.learnable_branch_weights = learnable_branch_weights
        self.encoder = ConvEncoder(hidden_dim=hidden_dim)
        self.lstm_encoder = LSTMEncoder(hidden_dim=hidden_dim)

        # 分支权重支持两种模式：
        # - 可学习：nn.Parameter，会被优化器更新
        # - 固定值：buffer，保持为 1 仅作显式门控
        self._init_branch_weight("weight_cnn", fixed_weight_cnn)
        self._init_branch_weight("weight_lstm", fixed_weight_lstm)

        fusion_dim = hidden_dim + hidden_dim
        self.backbone = nn.Sequential(
            nn.Linear(fusion_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
        )
        self.field_head = nn.Linear(latent_dim, field_height * field_width)
        self.acoustic_head = nn.Linear(latent_dim, 3)
        self.temperature_head = nn.Linear(latent_dim, 1)

    def _init_branch_weight(self, name: str, initial_value: float) -> None:
        value = torch.tensor([float(initial_value)], dtype=torch.float32)
        if self.learnable_branch_weights:
            setattr(self, name, nn.Parameter(value))
        else:
            self.register_buffer(name, value)

    def forward(
        self,
        waveform: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Predict the legacy grid, acoustic features, and scalar temperature."""
        encoded = self.weight_cnn * self.encoder(waveform)
        lstm_encoded = self.weight_lstm * self.lstm_encoder(waveform)

        cond = torch.cat(
            [
                encoded,
                lstm_encoded,
            ],
            dim=-1,
        )
        features = self.backbone(cond)
        field = self.field_head(features).view(-1, self.field_height, self.field_width)
        acoustic = self.acoustic_head(features)
        temperature = self.temperature_head(features)
        return {
            "field": torch.sigmoid(field),
            "acoustic": acoustic,
            "temperature": temperature,
        }
