"""Weighted POD fitted strictly on training cases."""

from __future__ import annotations

import numpy as np

from .point_field import kelvin_metrics


class WeightedPOD:
    def __init__(self, max_components: int = 512):
        self.max_components = max_components

    def fit(self, train_temperature_k: np.ndarray, sample_weights: np.ndarray) -> "WeightedPOD":
        data = np.asarray(train_temperature_k, np.float64)
        weights = np.asarray(sample_weights, np.float64)
        if data.ndim != 2 or weights.shape != (data.shape[1],):
            raise ValueError("POD 输入必须为 cases×P，权重必须为 [P]")
        self.mean_k_ = data.mean(axis=0)
        self.weights_ = weights / weights.mean()
        weighted = (data - self.mean_k_) * np.sqrt(self.weights_)[None, :]
        _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
        count = min(self.max_components, len(singular))
        self.basis_ = vt[:count] / np.sqrt(self.weights_)[None, :]
        energy = singular ** 2
        self.explained_variance_ratio_ = energy[:count] / max(energy.sum(), 1e-30)
        return self

    def reconstruct(self, temperature_k: np.ndarray, components: int) -> np.ndarray:
        if components < 1 or components > len(self.basis_):
            raise ValueError("POD components 超出已拟合范围")
        centered = np.asarray(temperature_k) - self.mean_k_
        basis = self.basis_[:components]
        coeff = (centered * self.weights_) @ basis.T
        return self.mean_k_ + coeff @ basis

    def evaluate(self, temperature_k: np.ndarray, components: int,
                 masks: dict[str, np.ndarray] | None = None) -> dict:
        reconstructed = self.reconstruct(temperature_k, components)
        return {"components": components,
                "cumulative_explained_variance": float(self.explained_variance_ratio_[:components].sum()),
                **kelvin_metrics(reconstructed, temperature_k, masks=masks)}
