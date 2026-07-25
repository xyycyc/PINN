"""Training and incremental-update workflows for legacy and fixed-node models."""

from __future__ import annotations

import json
import re
import copy
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import csv
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..artifact_paths import normalize_pt_filename, validate_artifact_basename
from ..config import AIModelConfig
from ..data_process import (
    AITemperatureDataset,
    load_material_collection,
    resolve_collection_manifest,
)
from .network import AIReconstructionModel
from .point_field import (
    POINT_FIELD_CHECKPOINT_VERSION,
    SUPPORTED_POINT_FIELD_CHECKPOINT_VERSIONS,
    DirectPointFieldModel,
    load_compatible_point_field_state,
    weighted_temperature_loss,
)
from .point_physics import POINT_SMOOTHNESS_COEFFICIENT, PointPhysicsOperator
from .rule_registry import default_training_time_stamp
from .waveform_io import prepare_model_waveform_input

_LEGACY_INCREMENTAL_CHECKPOINT_NAMES = frozenset({"ai_model_online.pt"})
MATERIAL_ROUTER_KIND = "sample_material_checkpoints"
MATERIAL_ROUTER_VERSION = 2

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _default_device(config: AIModelConfig) -> torch.device:
    normalized = str(config.device).strip().lower()
    if normalized in {"gpu", "cuda", "cuda:0"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
        # 用户明确希望走 GPU，但当前环境无 CUDA 时自动回退。
        return torch.device("cpu")
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(normalized)


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    while mask.dim() < pred.dim():
        mask = mask.unsqueeze(-1)
    weighted = (pred - target) ** 2 * mask
    denom = mask.sum().clamp_min(1.0)
    return weighted.sum() / denom


def _finite_difference_second_x(field: torch.Tensor) -> torch.Tensor:
    if field.shape[-1] < 3:
        return torch.zeros_like(field[:, :, :1])
    return field[:, :, 2:] - 2.0 * field[:, :, 1:-1] + field[:, :, :-2]


def _finite_difference_second_y(field: torch.Tensor) -> torch.Tensor:
    if field.shape[-2] < 3:
        return torch.zeros_like(field[:, :1, :])
    return field[:, 2:, :] - 2.0 * field[:, 1:-1, :] + field[:, :-2, :]


def _point_field_objective(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weights: torch.Tensor,
    *,
    training_mode: str,
    physics_residual_weight: float,
    point_physics: PointPhysicsOperator | None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the fixed-node objective without touching legacy-grid losses."""

    field_loss = weighted_temperature_loss(prediction, target, sample_weights)
    if training_mode == "normal":
        zero = prediction.new_zeros(())
        return field_loss, {
            "field_loss": field_loss,
            "smoothness_loss": zero,
            "physics_residual_loss": zero,
            "total_loss": field_loss,
        }
    if training_mode != "residual_pinn":
        raise ValueError("training_mode must be `normal` or `residual_pinn`")
    if point_physics is None:
        raise ValueError("residual_pinn fixed-node training requires point physics")
    smoothness_loss = point_physics.smoothness_loss(prediction)
    physics_residual_loss = point_physics.laplacian_loss(prediction)
    total_loss = (
        field_loss
        + POINT_SMOOTHNESS_COEFFICIENT * smoothness_loss
        + float(physics_residual_weight) * physics_residual_loss
    )
    return total_loss, {
        "field_loss": field_loss,
        "smoothness_loss": smoothness_loss,
        "physics_residual_loss": physics_residual_loss,
        "total_loss": total_loss,
    }


class ReconstructionTrainer:
    """Train one shared model or a router-managed set of material models."""

    def __init__(self, config: AIModelConfig | None = None):
        self.config = config or AIModelConfig()
        if int(self.config.epochs) <= 0:
            raise ValueError("epochs must be > 0")
        if int(self.config.early_stopping_patience) <= 0:
            raise ValueError("early_stopping_patience must be > 0")
        if self.config.training_mode not in {"normal", "residual_pinn"}:
            raise ValueError("training_mode must be `normal` or `residual_pinn`")
        if self.config.physics_residual_weight < 0:
            raise ValueError("physics_residual_weight must be >= 0")
        self.config.ensure_dirs()
        self.device = _default_device(self.config)

    def build_model(self) -> AIReconstructionModel:
        """Construct the legacy grid reconstruction model from runtime config."""
        return AIReconstructionModel(
            waveform_length=self.config.waveform_length,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            field_height=self.config.field_grid_2d[0],
            field_width=self.config.field_grid_2d[1],
            learnable_branch_weights=self.config.learnable_branch_weights,
            fixed_weight_cnn=self.config.fixed_weight_cnn,
            fixed_weight_lstm=self.config.fixed_weight_lstm,
        ).to(self.device)

    @staticmethod
    def _validate_dataset_pair(
        train_dataset: AITemperatureDataset,
        validation_dataset: AITemperatureDataset,
    ) -> None:
        if len(validation_dataset) == 0:
            raise ValueError("validation manifest 没有记录")
        if train_dataset.is_point_field != validation_dataset.is_point_field:
            raise ValueError("train/validation manifest 的模型类型不一致")
        if train_dataset.waveform_length != validation_dataset.waveform_length:
            raise ValueError(
                "train/validation 波形长度不一致: "
                f"{train_dataset.waveform_length} != {validation_dataset.waveform_length}"
            )
        if not train_dataset.is_point_field:
            return
        if train_dataset.point_count != validation_dataset.point_count:
            raise ValueError(
                "train/validation 温度场点数不一致: "
                f"{train_dataset.point_count} != {validation_dataset.point_count}"
            )
        for key in ("mean_k", "std_k"):
            if not np.isclose(
                float(train_dataset.normalization.get(key, np.nan)),
                float(validation_dataset.normalization.get(key, np.nan)),
            ):
                raise ValueError(f"train/validation 温度标准化参数不一致: {key}")
        for key in ("sampling_version", "source_mesh_fingerprint"):
            if train_dataset.sampling_metadata.get(key) != validation_dataset.sampling_metadata.get(key):
                raise ValueError(f"train/validation 固定节点采样定义不一致: {key}")

    def _loss(
        self,
        outputs: dict[str, torch.Tensor],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float | str]]:
        # 基础监督损失：温度场/声学参数/标量温度 + 平滑先验
        field_loss = _masked_mse(outputs["field"], batch["field"], batch["field_mask"])
        acoustic_loss = _masked_mse(outputs["acoustic"], batch["acoustic"], batch["acoustic_mask"])
        temperature_loss = F.mse_loss(outputs["temperature"], batch["temperature"])
        smoothness_loss = torch.mean(torch.abs(outputs["field"][:, :, 1:] - outputs["field"][:, :, :-1]))

        # residual_pinn 模式才附加物理残差，normal 模式保持现有训练逻辑。
        physics_residual_loss = torch.zeros((), device=outputs["field"].device)
        if self.config.training_mode == "residual_pinn":
            physics_residual_loss = self._physics_residual_loss(outputs["field"])

        loss = field_loss + 0.2 * acoustic_loss + 0.1 * temperature_loss + 0.05 * smoothness_loss
        if self.config.training_mode == "residual_pinn":
            loss = loss + self.config.physics_residual_weight * physics_residual_loss
        stats = {
            "field_loss": float(field_loss.detach().cpu()),
            "acoustic_loss": float(acoustic_loss.detach().cpu()),
            "temperature_loss": float(temperature_loss.detach().cpu()),
            "smoothness_loss": float(smoothness_loss.detach().cpu()),
            "physics_residual_loss": float(physics_residual_loss.detach().cpu()),
            "training_mode": self.config.training_mode,
            "total_loss": float(loss.detach().cpu()),
        }
        return loss, stats

    def _physics_residual_loss(
        self,
        field: torch.Tensor,
    ) -> torch.Tensor:
        # 标签不参与损失分支选择：统一计算网格拉普拉斯残差。
        d2x = _finite_difference_second_x(field)
        if field.shape[-2] < 3 or field.shape[-1] < 3:
            return torch.mean(d2x ** 2)
        d2x_center = d2x[:, 1:-1, :]
        d2y_center = _finite_difference_second_y(field)[:, :, 1:-1]
        return torch.mean((d2x_center + d2y_center) ** 2)

    def _save_history_artifacts(
        self,
        history: list[dict[str, float | str]],
        *,
        artifact_name: str,
        report_dir: Path,
    ) -> None:
        history_path = report_dir / f"{artifact_name}_history.json"
        history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

        if not history:
            return

        # CSV: 便于后续做表格分析/二次处理。
        csv_path = report_dir / f"{artifact_name}_history.csv"
        fieldnames = list(history[0].keys())
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(history)

        if plt is None:
            return

        epochs = [float(item.get("epoch", idx + 1)) for idx, item in enumerate(history)]
        curve_keys = [
            ("mean_epoch_loss", "Mean Epoch Loss"),
            ("field_loss", "Field Loss"),
            ("acoustic_loss", "Acoustic Loss"),
            ("temperature_loss", "Temperature Loss"),
            ("smoothness_loss", "Smoothness Loss"),
            ("physics_residual_loss", "Physics Residual Loss"),
        ]

        fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
        axes_flat = axes.ravel()
        for ax, (key, title) in zip(axes_flat, curve_keys):
            values = [float(item.get(key, 0.0)) for item in history]
            ax.plot(epochs, values, linewidth=1.2)
            if key == "mean_epoch_loss" and any("validation_loss" in item for item in history):
                validation_values = [float(item.get("validation_loss", float("nan"))) for item in history]
                ax.plot(epochs, validation_values, linewidth=1.2, label="Validation Loss")
                ax.legend()
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(key)
            ax.grid(True, linestyle="--", alpha=0.3)
        fig.tight_layout()
        plot_path = report_dir / f"{artifact_name}_loss_curve.png"
        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    def train(
        self,
        manifest_path: str | Path,
        validation_manifest_path: str | Path | None = None,
        checkpoint_name: str = "ai_model.pt",
        train_name: str | None = None,
    ) -> Path:
        """Train the model selected by the manifest schema and save its artifacts."""
        checkpoint_name = normalize_pt_filename(
            checkpoint_name,
            label="checkpoint 文件名",
        )
        if train_name is not None:
            train_name = validate_artifact_basename(
                train_name,
                label="训练任务名称",
                allow_empty=True,
            ) or None
        dataset = AITemperatureDataset(manifest_path, config=self.config)
        if len(dataset) == 0:
            raise ValueError(f"训练 manifest 没有记录: {manifest_path}")
        validation_dataset = (
            AITemperatureDataset(validation_manifest_path, config=self.config)
            if validation_manifest_path is not None
            else None
        )
        if validation_dataset is not None:
            self._validate_dataset_pair(dataset, validation_dataset)
        if dataset.is_point_field:
            return self._train_point_field(
                dataset,
                validation_dataset=validation_dataset,
                train_manifest_path=manifest_path,
                validation_manifest_path=validation_manifest_path,
                checkpoint_name=checkpoint_name,
                train_name=train_name,
            )
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        validation_loader = (
            DataLoader(validation_dataset, batch_size=self.config.batch_size, shuffle=False)
            if validation_dataset is not None
            else None
        )
        model = self.build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        history: list[dict[str, float | str]] = []
        best_validation_loss = float("inf")
        best_epoch: int | None = None
        best_model_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        stopped_early = False
        stopped_epoch: int | None = None

        epoch_bar = tqdm(
            range(self.config.epochs),
            desc=f"train[{self.config.training_mode}]",
            unit="epoch",
        )
        for epoch in epoch_bar:
            model.train()
            epoch_loss = 0.0
            epoch_stats_acc: dict[str, float] = defaultdict(float)
            batch_count = 0
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                optimizer.zero_grad()
                waveform_norm, _, _ = prepare_model_waveform_input(batch["waveform"])
                outputs = model(waveform=waveform_norm)
                loss, stats = self._loss(outputs, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                batch_count += 1
                for key in ("field_loss", "acoustic_loss", "temperature_loss", "smoothness_loss", "physics_residual_loss"):
                    epoch_stats_acc[key] += float(stats[key])

            mean_stats: dict[str, float | str] = {
                key: epoch_stats_acc[key] / max(batch_count, 1)
                for key in ("field_loss", "acoustic_loss", "temperature_loss", "smoothness_loss", "physics_residual_loss")
            }
            mean_stats["epoch"] = float(epoch + 1)
            mean_stats["training_mode"] = self.config.training_mode
            mean_stats["mean_epoch_loss"] = epoch_loss / max(batch_count, 1)
            # 与 mean_epoch_loss 语义一致，保留 total 命名便于外部统一读取。
            mean_stats["total_loss"] = mean_stats["mean_epoch_loss"]
            if validation_loader is not None:
                model.eval()
                validation_total = 0.0
                validation_samples = 0
                with torch.no_grad():
                    for validation_batch in validation_loader:
                        validation_batch = {
                            key: value.to(self.device) for key, value in validation_batch.items()
                        }
                        waveform_norm, _, _ = prepare_model_waveform_input(validation_batch["waveform"])
                        validation_outputs = model(waveform=waveform_norm)
                        validation_loss, _ = self._loss(validation_outputs, validation_batch)
                        sample_count = int(validation_batch["waveform"].shape[0])
                        validation_total += float(validation_loss.detach().cpu()) * sample_count
                        validation_samples += sample_count
                validation_mean = validation_total / max(validation_samples, 1)
                mean_stats["validation_loss"] = validation_mean
                if validation_mean < best_validation_loss:
                    best_validation_loss = validation_mean
                    best_epoch = epoch + 1
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= int(self.config.early_stopping_patience):
                        stopped_early = True
                        stopped_epoch = epoch + 1
            history.append(mean_stats)
            epoch_bar.set_postfix(
                loss=f"{mean_stats['mean_epoch_loss']:.4f}",
                val=(f"{mean_stats['validation_loss']:.4f}" if "validation_loss" in mean_stats else "-"),
                field=f"{mean_stats['field_loss']:.4f}",
                acoustic=f"{mean_stats['acoustic_loss']:.4f}",
                temp=f"{mean_stats['temperature_loss']:.4f}",
                phy=f"{mean_stats['physics_residual_loss']:.4f}",
            )
            if stopped_early:
                epoch_bar.write(
                    "early stopping: validation loss did not improve for "
                    f"{self.config.early_stopping_patience} consecutive epochs "
                    f"(stopped at epoch {stopped_epoch}, best epoch {best_epoch})"
                )
                break

        if best_model_state is not None:
            model.load_state_dict(best_model_state, strict=True)

        run_name = str(train_name or Path(checkpoint_name).stem).strip()
        if not run_name:
            run_name = Path(checkpoint_name).stem
        checkpoint_dir = self.config.train_checkpoint_root / run_name
        report_dir = self.config.train_report_root / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / checkpoint_name
        config_dict = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in asdict(self.config).items()
        }
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config_dict,
                "history": history,
                "train_manifest": str(Path(manifest_path).resolve()),
                "validation_manifest": (
                    str(Path(validation_manifest_path).resolve())
                    if validation_manifest_path is not None
                    else None
                ),
                "best_epoch": best_epoch,
                "best_validation_loss": (
                    float(best_validation_loss) if best_epoch is not None else None
                ),
                "early_stopping_patience": int(self.config.early_stopping_patience),
                "stopped_early": stopped_early,
                "stopped_epoch": stopped_epoch,
                "completed_epochs": len(history),
                "requested_epochs": int(self.config.epochs),
            },
            checkpoint_path,
        )
        self._save_history_artifacts(
            history,
            artifact_name=Path(checkpoint_name).stem,
            report_dir=report_dir,
        )
        return checkpoint_path

    def _train_point_field(
        self,
        dataset: AITemperatureDataset,
        *,
        validation_dataset: AITemperatureDataset | None = None,
        train_manifest_path: str | Path | None = None,
        validation_manifest_path: str | Path | None = None,
        checkpoint_name: str,
        train_name: str | None,
        sample_material: dict[str, object] | None = None,
    ) -> Path:
        """Train the versioned direct P-node model without changing legacy behavior."""
        checkpoint_name = normalize_pt_filename(
            checkpoint_name,
            label="checkpoint 文件名",
        )
        if train_name is not None:
            train_name = validate_artifact_basename(
                train_name,
                label="训练任务名称",
                allow_empty=True,
            ) or None
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        validation_loader = (
            DataLoader(validation_dataset, batch_size=self.config.batch_size, shuffle=False)
            if validation_dataset is not None
            else None
        )
        model = DirectPointFieldModel(
            point_count=int(dataset.point_count),
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            learnable_branch_weights=self.config.learnable_branch_weights,
            fixed_weight_cnn=self.config.fixed_weight_cnn,
            fixed_weight_lstm=self.config.fixed_weight_lstm,
        ).to(self.device)
        point_physics = (
            PointPhysicsOperator(
                dataset.point_coordinates_m,
                dataset.point_node_ids,
                dataset.constituent_material_ids,
                dataset.point_interface_side,
            ).to(self.device)
            if self.config.training_mode == "residual_pinn"
            else None
        )

        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        history: list[dict[str, float | str]] = []
        best_validation_loss = float("inf")
        best_validation_mae_k = float("inf")
        best_epoch: int | None = None
        best_model_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        stopped_early = False
        stopped_epoch: int | None = None
        training_started = time.perf_counter()
        material_key = str((sample_material or {}).get("material_key", "")).strip()
        material_label = str(
            (sample_material or {}).get("material_name", material_key)
        ).strip()
        progress_label = (
            f"train[sample-material:{material_key}]"
            if sample_material is not None
            else "train[direct_point_field]"
        )
        epoch_bar = tqdm(range(self.config.epochs), desc=progress_label, unit="epoch")
        for epoch in epoch_bar:
            model.train()
            total_loss = 0.0
            total_mae_k = 0.0
            component_totals = {
                "field_loss": 0.0,
                "smoothness_loss": 0.0,
                "physics_residual_loss": 0.0,
            }
            batches = 0
            for batch in loader:
                waveform = batch["waveform"].to(self.device)
                target = batch["field"].to(self.device)
                weights = batch["sample_weights"].to(self.device)
                optimizer.zero_grad()
                waveform_norm, _, _ = prepare_model_waveform_input(waveform)
                prediction = model(waveform_norm)
                loss, loss_parts = _point_field_objective(
                    prediction,
                    target,
                    weights,
                    training_mode=self.config.training_mode,
                    physics_residual_weight=self.config.physics_residual_weight,
                    point_physics=point_physics,
                )
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu())
                for key in component_totals:
                    component_totals[key] += float(loss_parts[key].detach().cpu())
                total_mae_k += float((prediction.detach() - target).abs().mean().cpu()) * dataset.temperature_std_k
                batches += 1
            mean_loss = total_loss / max(batches, 1)
            epoch_stats: dict[str, float | str] = {
                "epoch": float(epoch + 1), "training_mode": self.config.training_mode,
                "mean_epoch_loss": mean_loss, "total_loss": mean_loss,
                "field_loss": component_totals["field_loss"] / max(batches, 1),
                "temperature_mae_k": total_mae_k / max(batches, 1),
                "acoustic_loss": 0.0, "temperature_loss": 0.0,
                "smoothness_loss": component_totals["smoothness_loss"] / max(batches, 1),
                "physics_residual_loss": (
                    component_totals["physics_residual_loss"] / max(batches, 1)
                ),
            }
            if validation_loader is not None and validation_dataset is not None:
                model.eval()
                validation_loss_total = 0.0
                validation_component_totals = {
                    "field_loss": 0.0,
                    "smoothness_loss": 0.0,
                    "physics_residual_loss": 0.0,
                }
                validation_abs_error_total = 0.0
                validation_samples = 0
                validation_values = 0
                with torch.no_grad():
                    for validation_batch in validation_loader:
                        waveform = validation_batch["waveform"].to(self.device)
                        target = validation_batch["field"].to(self.device)
                        weights = validation_batch["sample_weights"].to(self.device)
                        waveform_norm, _, _ = prepare_model_waveform_input(waveform)
                        prediction = model(waveform_norm)
                        validation_loss, validation_parts = _point_field_objective(
                            prediction,
                            target,
                            weights,
                            training_mode=self.config.training_mode,
                            physics_residual_weight=self.config.physics_residual_weight,
                            point_physics=point_physics,
                        )
                        sample_count = int(waveform.shape[0])
                        validation_loss_total += float(validation_loss.detach().cpu()) * sample_count
                        for key in validation_component_totals:
                            validation_component_totals[key] += (
                                float(validation_parts[key].detach().cpu()) * sample_count
                            )
                        validation_abs_error_total += float((prediction - target).abs().sum().detach().cpu())
                        validation_samples += sample_count
                        validation_values += int(target.numel())
                validation_mean = validation_loss_total / max(validation_samples, 1)
                validation_mae_k = (
                    validation_abs_error_total / max(validation_values, 1)
                    * validation_dataset.temperature_std_k
                )
                epoch_stats["validation_loss"] = validation_mean
                epoch_stats["validation_temperature_mae_k"] = validation_mae_k
                epoch_stats["validation_field_loss"] = (
                    validation_component_totals["field_loss"]
                    / max(validation_samples, 1)
                )
                epoch_stats["validation_smoothness_loss"] = (
                    validation_component_totals["smoothness_loss"]
                    / max(validation_samples, 1)
                )
                epoch_stats["validation_physics_residual_loss"] = (
                    validation_component_totals["physics_residual_loss"]
                    / max(validation_samples, 1)
                )
                if validation_mean < best_validation_loss:
                    best_validation_loss = validation_mean
                    best_validation_mae_k = validation_mae_k
                    best_epoch = epoch + 1
                    best_model_state = copy.deepcopy(model.state_dict())
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= int(self.config.early_stopping_patience):
                        stopped_early = True
                        stopped_epoch = epoch + 1
            history.append(epoch_stats)
            epoch_bar.set_postfix(
                loss=f"{mean_loss:.4f}",
                val=(f"{epoch_stats['validation_loss']:.4f}" if "validation_loss" in epoch_stats else "-"),
            )
            if stopped_early:
                epoch_bar.write(
                    "early stopping: validation loss did not improve for "
                    f"{self.config.early_stopping_patience} consecutive epochs "
                    f"(stopped at epoch {stopped_epoch}, best epoch {best_epoch})"
                )
                break
        if best_model_state is not None:
            model.load_state_dict(best_model_state, strict=True)
        run_name = str(train_name or Path(checkpoint_name).stem).strip() or Path(checkpoint_name).stem
        checkpoint_dir = self.config.train_checkpoint_root / run_name
        report_dir = self.config.train_report_root / run_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True); report_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / checkpoint_name
        config_dict = {key: (str(value) if isinstance(value, Path) else value)
                       for key, value in asdict(self.config).items()}
        config_dict["waveform_length"] = int(dataset.waveform_length)
        training_summary = {
            "training_seconds": float(time.perf_counter() - training_started),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "parameter_bytes": int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())),
            "device": str(self.device),
            "learnable_branch_weights": bool(model.learnable_branch_weights),
            "branch_weight_cnn": float(model.weight_cnn.detach().cpu().item()),
            "branch_weight_lstm": float(model.weight_lstm.detach().cpu().item()),
            "train_samples": int(len(dataset)),
            "validation_samples": int(len(validation_dataset)) if validation_dataset is not None else 0,
            "best_epoch": best_epoch,
            "best_validation_loss": float(best_validation_loss) if best_epoch is not None else None,
            "best_validation_temperature_mae_k": (
                float(best_validation_mae_k) if best_epoch is not None else None
            ),
            "early_stopping_patience": int(self.config.early_stopping_patience),
            "stopped_early": stopped_early,
            "stopped_epoch": stopped_epoch,
            "completed_epochs": len(history),
            "requested_epochs": int(self.config.epochs),
        }
        if sample_material is not None:
            training_summary.update(
                {
                    "sample_material_key": material_key,
                    "sample_material_name": material_label,
                    "point_count": int(dataset.point_count),
                }
            )
        torch.save({
            "checkpoint_version": POINT_FIELD_CHECKPOINT_VERSION,
            "model_kind": "direct_point_field",
            "model_state": model.state_dict(),
            "config": config_dict,
            "history": history,
            "train_manifest": (
                str(Path(train_manifest_path).resolve()) if train_manifest_path is not None else None
            ),
            "validation_manifest": (
                str(Path(validation_manifest_path).resolve())
                if validation_manifest_path is not None
                else None
            ),
            "schema_version": int(dataset.schema_version),
            "waveform_length": int(dataset.waveform_length),
            "point_count": int(dataset.point_count),
            "full_point_count": int(dataset.point_count),
            "point_indices": None,
            "sample_material_key": material_key or None,
            "sample_material_name": material_label or None,
            "chunk_size": int(model.chunk_size),
            "normalization": dict(dataset.normalization),
            "sampling_metadata": dict(dataset.sampling_metadata),
            "point_physics": (
                point_physics.metadata() if point_physics is not None else None
            ),
            "training_summary": training_summary,
        }, checkpoint_path)
        (report_dir / "training_summary.json").write_text(
            json.dumps(training_summary, ensure_ascii=False, indent=2), encoding="utf-8")
        self._save_history_artifacts(history, artifact_name=Path(checkpoint_name).stem, report_dir=report_dir)
        return checkpoint_path

    def train_material_checkpoints(
        self,
        collection_path: str | Path,
        checkpoint_name: str = "ai_model.pt",
        train_name: str | None = None,
    ) -> Path:
        """Train one complete field checkpoint per sample-level material folder."""
        collection_path = Path(collection_path).resolve()
        collection = load_material_collection(collection_path)
        checkpoint_name = normalize_pt_filename(
            checkpoint_name or "ai_model.pt",
            label="checkpoint 文件名",
        )
        base_checkpoint = Path(checkpoint_name)
        suffix = base_checkpoint.suffix or ".pt"
        stem = base_checkpoint.stem or "ai_model"
        run_name = validate_artifact_basename(
            str(train_name or stem),
            label="训练任务名称",
        )
        checkpoints: list[dict[str, object]] = []
        for raw_entry in collection["materials"]:
            entry = dict(raw_entry)
            material_key = str(entry["material_key"]).strip()
            material_name = str(entry.get("material_name", material_key)).strip()
            safe_key = re.sub(r"[^A-Za-z0-9_-]+", "_", material_key).strip("_")
            if not safe_key:
                raise ValueError(f"材料路由字段不能生成 checkpoint 文件名: {material_key!r}")
            train_manifest = resolve_collection_manifest(
                collection_path,
                entry,
                "train",
            )
            validation_manifest = resolve_collection_manifest(
                collection_path,
                entry,
                "validation",
            )
            dataset = AITemperatureDataset(train_manifest, config=self.config)
            validation_dataset = AITemperatureDataset(validation_manifest, config=self.config)
            self._validate_dataset_pair(dataset, validation_dataset)
            record_materials = {
                str(record.get("material_key", "")).strip()
                for record in dataset.records
            }
            if record_materials != {material_key}:
                raise ValueError(
                    f"材料 {material_key} 的训练清单包含其他材料: {sorted(record_materials)}"
                )
            if not dataset.is_point_field:
                raise ValueError(
                    f"材料 {material_key} 不是固定节点数据集，无法训练完整温度场 checkpoint"
                )
            material_checkpoint_name = f"{stem}__{safe_key}{suffix}"
            checkpoint_path = self._train_point_field(
                dataset,
                validation_dataset=validation_dataset,
                train_manifest_path=train_manifest,
                validation_manifest_path=validation_manifest,
                checkpoint_name=material_checkpoint_name,
                train_name=run_name,
                sample_material={
                    "material_key": material_key,
                    "material_name": material_name,
                },
            )
            checkpoints.append(
                {
                    "material_key": material_key,
                    "material_name": material_name,
                    "rule_material": material_key,
                    "point_count": int(dataset.point_count),
                    "waveform_length": int(dataset.waveform_length),
                    "train_manifest": str(train_manifest),
                    "checkpoint": checkpoint_path.name,
                }
            )

        checkpoint_dir = self.config.train_checkpoint_root / run_name
        router_path = checkpoint_dir / f"{stem}__material_router.json"
        router_payload = {
            "router_version": MATERIAL_ROUTER_VERSION,
            "router_kind": MATERIAL_ROUTER_KIND,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "source_collection": str(collection_path),
            "dataset_label": str(collection.get("dataset_label", "")),
            "routing_field": "records[].material_key",
            "checkpoints": checkpoints,
        }
        router_path.write_text(
            json.dumps(router_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return router_path

def resolve_incremental_artifacts(
    config: AIModelConfig,
    base_checkpoint_path: str | Path,
    *,
    output_name: str | None = None,
    run_stamp: str | None = None,
) -> tuple[Path, Path, str]:
    """解析增量训练的 checkpoint 与报告目录。

    - 权重写入 ``result/train/checkpoint/<基础任务名>/<时间戳>.pt``
    - 报告写入 ``result/train/report/<基础任务名>/Incremental/<时间戳>/``
    """

    base_ckpt = Path(base_checkpoint_path).resolve()
    base_run_name = base_ckpt.parent.name
    stamp = validate_artifact_basename(
        str(run_stamp or default_training_time_stamp()),
        label="增量训练时间戳",
    )
    name = str(output_name or "").strip()
    if not name or name in _LEGACY_INCREMENTAL_CHECKPOINT_NAMES:
        checkpoint_name = f"{stamp}.pt"
    else:
        checkpoint_name = normalize_pt_filename(name, label="增量 checkpoint 文件名")

    checkpoint_dir = config.train_checkpoint_root / base_run_name
    checkpoint_path = checkpoint_dir / checkpoint_name
    report_dir = config.train_report_root / base_run_name / "Incremental" / stamp
    return checkpoint_path, report_dir, stamp


class OnlineUpdater:
    """在已有 checkpoint 上做增量微调（权重与报告写入基础任务对应目录）。"""

    def __init__(self, config: AIModelConfig | None = None):
        self.config = config or AIModelConfig()
        if int(self.config.online_epochs) <= 0:
            raise ValueError("online_epochs must be > 0")
        if self.config.training_mode not in {"normal", "residual_pinn"}:
            raise ValueError("training_mode must be `normal` or `residual_pinn`")
        if self.config.physics_residual_weight < 0:
            raise ValueError("physics_residual_weight must be >= 0")
        self.device = _default_device(self.config)

    def update(
        self,
        manifest_path: str | Path,
        checkpoint_path: str | Path,
        output_name: str | None = None,
        train_name: str | None = None,
    ) -> Path:
        """Fine-tune a compatible checkpoint and persist a new immutable artifact."""
        del train_name  # 增量训练目录固定为基础 checkpoint 父目录名
        base_checkpoint = Path(checkpoint_path).resolve()

        output_path, report_dir, stamp = resolve_incremental_artifacts(
            self.config,
            base_checkpoint,
            output_name=output_name,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        dataset = AITemperatureDataset(manifest_path, config=self.config)
        if len(dataset) == 0:
            raise ValueError(f"增量训练 manifest 没有记录: {manifest_path}")
        state = torch.load(base_checkpoint, map_location=self.device)
        if dataset.is_point_field:
            return self._update_point_field(dataset, state, base_checkpoint, output_path, report_dir, stamp)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        model = AIReconstructionModel(
            waveform_length=self.config.waveform_length,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            field_height=self.config.field_grid_2d[0],
            field_width=self.config.field_grid_2d[1],
            learnable_branch_weights=self.config.learnable_branch_weights,
            fixed_weight_cnn=self.config.fixed_weight_cnn,
            fixed_weight_lstm=self.config.fixed_weight_lstm,
        ).to(self.device)
        model.load_state_dict(state["model_state"], strict=False)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate * 0.2)
        loss_helper = ReconstructionTrainer(self.config)
        history: list[dict[str, float | str]] = []

        epoch_bar = tqdm(
            range(self.config.online_epochs),
            desc=f"incremental[{self.config.training_mode}]",
            unit="epoch",
        )
        for epoch in epoch_bar:
            epoch_loss = 0.0
            epoch_stats_acc: dict[str, float] = defaultdict(float)
            batch_count = 0
            for batch in loader:
                batch = {key: value.to(self.device) for key, value in batch.items()}
                optimizer.zero_grad()
                waveform_norm, _, _ = prepare_model_waveform_input(batch["waveform"])
                outputs = model(waveform=waveform_norm)
                loss, stats = loss_helper._loss(outputs, batch)
                loss.backward()
                optimizer.step()
                epoch_loss += float(loss.detach().cpu())
                batch_count += 1
                for key in (
                    "field_loss",
                    "acoustic_loss",
                    "temperature_loss",
                    "smoothness_loss",
                    "physics_residual_loss",
                ):
                    epoch_stats_acc[key] += float(stats[key])

            mean_stats: dict[str, float | str] = {
                key: epoch_stats_acc[key] / max(batch_count, 1)
                for key in (
                    "field_loss",
                    "acoustic_loss",
                    "temperature_loss",
                    "smoothness_loss",
                    "physics_residual_loss",
                )
            }
            mean_stats["epoch"] = float(epoch + 1)
            mean_stats["training_mode"] = self.config.training_mode
            mean_stats["mean_epoch_loss"] = epoch_loss / max(batch_count, 1)
            mean_stats["total_loss"] = mean_stats["mean_epoch_loss"]
            history.append(mean_stats)
            epoch_bar.set_postfix(loss=f"{mean_stats['mean_epoch_loss']:.4f}")

        config_dict = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in asdict(self.config).items()
        }
        torch.save(
            {
                "model_state": model.state_dict(),
                "config": config_dict,
                "history": history,
                "base_checkpoint": str(base_checkpoint),
                "incremental_stamp": stamp,
            },
            output_path,
        )
        loss_helper._save_history_artifacts(
            history,
            artifact_name=stamp,
            report_dir=report_dir,
        )
        return output_path

    def _update_point_field(
        self,
        dataset: AITemperatureDataset,
        state: dict,
        base_checkpoint: Path,
        output_path: Path,
        report_dir: Path,
        stamp: str,
    ) -> Path:
        checkpoint_version = int(state.get("checkpoint_version", 0))
        if (
            state.get("model_kind") != "direct_point_field"
            or checkpoint_version not in SUPPORTED_POINT_FIELD_CHECKPOINT_VERSIONS
        ):
            raise ValueError(
                "固定节点增量训练需要 direct_point_field checkpoint，"
                f"支持版本 {SUPPORTED_POINT_FIELD_CHECKPOINT_VERSIONS}"
            )
        if int(state.get("point_count", 0)) != dataset.point_count:
            raise ValueError("固定节点增量训练的 checkpoint 与 manifest 点数不一致")
        checkpoint_waveform_length = int(
            state.get("waveform_length", state.get("config", {}).get("waveform_length", 0))
        )
        if checkpoint_waveform_length and checkpoint_waveform_length != dataset.waveform_length:
            raise ValueError(
                "固定节点增量训练的 checkpoint 与 manifest 波形长度不一致: "
                f"{checkpoint_waveform_length} != {dataset.waveform_length}"
            )
        for key in ("mean_k", "std_k"):
            if not np.isclose(float(state.get("normalization", {}).get(key, np.nan)),
                              float(dataset.normalization.get(key, np.nan))):
                raise ValueError(f"固定节点增量训练标准化参数不一致: {key}")
        checkpoint_sampling = state.get("sampling_metadata", {})
        for key in ("sampling_version", "source_mesh_fingerprint"):
            if checkpoint_sampling.get(key) != dataset.sampling_metadata.get(key):
                raise ValueError(f"固定节点增量训练采样定义不一致: {key}")
        model = DirectPointFieldModel(
            dataset.point_count,
            hidden_dim=self.config.hidden_dim,
            latent_dim=self.config.latent_dim,
            chunk_size=int(state.get("chunk_size", 1000)),
            learnable_branch_weights=self.config.learnable_branch_weights,
            fixed_weight_cnn=self.config.fixed_weight_cnn,
            fixed_weight_lstm=self.config.fixed_weight_lstm,
        ).to(self.device)
        legacy_missing = load_compatible_point_field_state(model, state["model_state"])
        if not legacy_missing and not model.learnable_branch_weights:
            with torch.no_grad():
                model.weight_cnn.fill_(float(self.config.fixed_weight_cnn))
                model.weight_lstm.fill_(float(self.config.fixed_weight_lstm))
        point_physics = (
            PointPhysicsOperator(
                dataset.point_coordinates_m,
                dataset.point_node_ids,
                dataset.constituent_material_ids,
                dataset.point_interface_side,
            ).to(self.device)
            if self.config.training_mode == "residual_pinn"
            else None
        )
        model.train()
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate * 0.2)
        history: list[dict[str, float | str]] = []
        for epoch in tqdm(
            range(self.config.online_epochs),
            desc=f"incremental[direct_point_field:{self.config.training_mode}]",
            unit="epoch",
        ):
            total = 0.0
            total_mae_k = 0.0
            component_totals = {
                "field_loss": 0.0,
                "smoothness_loss": 0.0,
                "physics_residual_loss": 0.0,
            }
            batches = 0
            for batch in loader:
                waveform = batch["waveform"].to(self.device)
                target = batch["field"].to(self.device)
                weights = batch["sample_weights"].to(self.device)
                optimizer.zero_grad()
                waveform_norm, _, _ = prepare_model_waveform_input(waveform)
                prediction = model(waveform_norm)
                loss, loss_parts = _point_field_objective(
                    prediction,
                    target,
                    weights,
                    training_mode=self.config.training_mode,
                    physics_residual_weight=self.config.physics_residual_weight,
                    point_physics=point_physics,
                )
                loss.backward()
                optimizer.step()
                total += float(loss.detach().cpu())
                for key in component_totals:
                    component_totals[key] += float(loss_parts[key].detach().cpu())
                total_mae_k += (
                    float((prediction.detach() - target).abs().mean().cpu())
                    * dataset.temperature_std_k
                )
                batches += 1
            value = total / max(batches, 1)
            history.append({
                "epoch": float(epoch + 1),
                "training_mode": self.config.training_mode,
                "mean_epoch_loss": value,
                "total_loss": value,
                "field_loss": component_totals["field_loss"] / max(batches, 1),
                "temperature_mae_k": total_mae_k / max(batches, 1),
                "acoustic_loss": 0.0,
                "temperature_loss": 0.0,
                "smoothness_loss": (
                    component_totals["smoothness_loss"] / max(batches, 1)
                ),
                "physics_residual_loss": (
                    component_totals["physics_residual_loss"] / max(batches, 1)
                ),
            })
        config_dict = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in asdict(self.config).items()
        }
        config_dict["waveform_length"] = int(dataset.waveform_length)
        if legacy_missing and not model.learnable_branch_weights:
            config_dict["fixed_weight_cnn"] = 1.0
            config_dict["fixed_weight_lstm"] = 1.0
        updated = dict(state)
        updated.update({
            "checkpoint_version": POINT_FIELD_CHECKPOINT_VERSION,
            "model_state": model.state_dict(),
            "config": config_dict,
            "history": history,
            "base_checkpoint": str(base_checkpoint),
            "incremental_stamp": stamp,
            "point_physics": (
                point_physics.metadata() if point_physics is not None else None
            ),
        })
        torch.save(updated, output_path)
        ReconstructionTrainer(self.config)._save_history_artifacts(history, artifact_name=stamp, report_dir=report_dir)
        return output_path
