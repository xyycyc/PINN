from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import csv

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..config import AIModelConfig
from ..data_process import AITemperatureDataset
from .network import AIReconstructionModel
from .rule_registry import default_training_time_stamp
from .waveform_io import prepare_model_waveform_input

_LEGACY_INCREMENTAL_CHECKPOINT_NAMES = frozenset({"ai_model_online.pt"})

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


class ReconstructionTrainer:
    def __init__(self, config: AIModelConfig | None = None):
        self.config = config or AIModelConfig()
        self.config.ensure_dirs()
        if self.config.training_mode not in {"normal", "residual_pinn"}:
            raise ValueError("training_mode must be `normal` or `residual_pinn`")
        if self.config.physics_residual_weight < 0:
            raise ValueError("physics_residual_weight must be >= 0")
        self.device = _default_device(self.config)

    def build_model(self) -> AIReconstructionModel:
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
        checkpoint_name: str = "ai_model.pt",
        train_name: str | None = None,
    ) -> Path:
        dataset = AITemperatureDataset(manifest_path, config=self.config)
        loader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
        model = self.build_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        history: list[dict[str, float | str]] = []

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
            history.append(mean_stats)
            epoch_bar.set_postfix(
                loss=f"{mean_stats['mean_epoch_loss']:.4f}",
                field=f"{mean_stats['field_loss']:.4f}",
                acoustic=f"{mean_stats['acoustic_loss']:.4f}",
                temp=f"{mean_stats['temperature_loss']:.4f}",
                phy=f"{mean_stats['physics_residual_loss']:.4f}",
            )

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
            },
            checkpoint_path,
        )
        self._save_history_artifacts(
            history,
            artifact_name=Path(checkpoint_name).stem,
            report_dir=report_dir,
        )
        return checkpoint_path


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
    stamp = str(run_stamp or default_training_time_stamp()).strip() or default_training_time_stamp()
    name = str(output_name or "").strip()
    if not name or name in _LEGACY_INCREMENTAL_CHECKPOINT_NAMES:
        checkpoint_name = f"{stamp}.pt"
    elif name.endswith(".pt"):
        checkpoint_name = name
    else:
        checkpoint_name = f"{name}.pt"

    checkpoint_dir = config.train_checkpoint_root / base_run_name
    checkpoint_path = checkpoint_dir / checkpoint_name
    report_dir = config.train_report_root / base_run_name / "Incremental" / stamp
    return checkpoint_path, report_dir, stamp


class OnlineUpdater:
    """在已有 checkpoint 上做增量微调（权重与报告写入基础任务对应目录）。"""

    def __init__(self, config: AIModelConfig | None = None):
        self.config = config or AIModelConfig()
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
        state = torch.load(base_checkpoint, map_location=self.device)
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
