from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import AIModelConfig
from ..data_process import AITemperatureDataset
from ..fortran import backend_name, compute_prediction_metrics
from .network import AIReconstructionModel
from .checkpoint_runtime import load_model_state_strict, sync_config_for_inference
from .trainer import ReconstructionTrainer, _default_device
from .waveform_io import postprocess_model_outputs, prepare_model_waveform_input

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def default_predict_output_name(
    checkpoint_path: str | Path,
    now: datetime | None = None,
) -> str:
    """默认推理目录名：<检查点父目录名>_<预测时间>。"""
    path = Path(checkpoint_path).resolve()
    folder_name = path.parent.name.strip() or path.stem
    dt = now or datetime.now()
    stamp = f"{dt.year}_{dt.month}_{dt.day}_{dt.strftime('%H%M%S')}"
    return f"{folder_name}_{stamp}"


def _build_model_from_cfg(cfg: AIModelConfig) -> AIReconstructionModel:
    return AIReconstructionModel(
        waveform_length=cfg.waveform_length,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        field_height=cfg.field_grid_2d[0],
        field_width=cfg.field_grid_2d[1],
        learnable_branch_weights=cfg.learnable_branch_weights,
        fixed_weight_cnn=cfg.fixed_weight_cnn,
        fixed_weight_lstm=cfg.fixed_weight_lstm,
    )


def _plot_scatter(
    temp_true: np.ndarray,
    temp_pred: np.ndarray,
    output_path: Path,
) -> None:
    if plt is None or temp_true.size == 0:
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(temp_true, temp_pred, s=16, alpha=0.7, label="samples")
    lo = float(min(temp_true.min(), temp_pred.min()))
    hi = float(max(temp_true.max(), temp_pred.max()))
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1.0, label="y = x")
    plt.xlabel("Ground truth temperature (K)")
    plt.ylabel("Predicted temperature (K)")
    plt.title("Temperature: prediction vs ground truth")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_field_triptych(
    sample_id: str,
    true_field: np.ndarray,
    pred_field: np.ndarray,
    output_path: Path,
) -> None:
    if plt is None:
        return
    err_field = pred_field - true_field
    vmin = float(min(true_field.min(), pred_field.min()))
    vmax = float(max(true_field.max(), pred_field.max()))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(true_field, vmin=vmin, vmax=vmax, cmap="inferno", aspect="auto")
    axes[0].set_title(f"true ({sample_id})")
    plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = axes[1].imshow(pred_field, vmin=vmin, vmax=vmax, cmap="inferno", aspect="auto")
    axes[1].set_title("pred")
    plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    im2 = axes[2].imshow(err_field, cmap="seismic", aspect="auto")
    axes[2].set_title("error (pred - true)")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _synchronize_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)


DEFAULT_BENCHMARK_RUNS = 3


def _benchmark_inference_only(
    model: AIReconstructionModel,
    loader: DataLoader,
    device: torch.device,
    *,
    warmup_requested: int,
    benchmark_runs: int,
) -> dict[str, float]:
    warmup_steps = max(0, int(warmup_requested))
    benchmark_runs = max(0, int(benchmark_runs))
    if benchmark_runs <= 0:
        return {}

    def _forward_once(batch: dict[str, torch.Tensor]) -> None:
        waveform_norm, wf_mean, wf_std = prepare_model_waveform_input(batch["waveform"])
        outputs = model(waveform=waveform_norm)
        _ = postprocess_model_outputs(outputs, wf_mean, wf_std)

    if warmup_steps > 0:
        warmup_done = 0
        with torch.no_grad():
            while warmup_done < warmup_steps:
                for batch in loader:
                    batch = {key: value.to(device) for key, value in batch.items()}
                    _forward_once(batch)
                    warmup_done += int(batch["waveform"].shape[0])
        _synchronize_if_cuda(device)

    elapsed_seconds = 0.0
    seen_samples = 0
    repeats_done = 0
    with torch.no_grad():
        while repeats_done < benchmark_runs:
            for batch in loader:
                if repeats_done >= benchmark_runs:
                    break
                batch = {key: value.to(device) for key, value in batch.items()}
                _synchronize_if_cuda(device)
                t0 = time.perf_counter()
                _forward_once(batch)
                _synchronize_if_cuda(device)
                elapsed_seconds += time.perf_counter() - t0
                seen_samples += int(batch["waveform"].shape[0])
                repeats_done += 1

    if seen_samples <= 0 or elapsed_seconds <= 0:
        return {}
    return {
        "benchmark_enabled": 1.0,
        "benchmark_warmup_samples_requested": float(warmup_requested),
        "benchmark_warmup_samples_effective": float(warmup_steps),
        "benchmark_runs_requested": float(benchmark_runs),
        "benchmark_runs": float(repeats_done),
        "benchmark_samples": float(seen_samples),
        "benchmark_total_seconds": float(elapsed_seconds),
        "benchmark_latency_ms_per_sample": float(elapsed_seconds * 1000.0 / seen_samples),
        "benchmark_throughput_samples_per_sec": float(seen_samples / elapsed_seconds),
    }


def predict_and_compare(
    cfg: AIModelConfig,
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    rule_csv_path: str | Path | None = None,
    sync_config_from_checkpoint: bool = True,
    compute_loss: bool = False,
    enable_plots: bool = False,
    num_field_samples: int = 6,
    enable_benchmark: bool = False,
    benchmark_warmup_samples: int = 64,
    benchmark_runs: int = DEFAULT_BENCHMARK_RUNS,
) -> dict[str, Any]:
    """Run inference on a manifest and produce prediction-vs-ground-truth artifacts.

    默认仅前向推理 + 温度 MAE/RMSE，不计算训练用 loss。
    ``enable_plots`` 为真时写出散点图与温度场三联图；``enable_benchmark`` 为真时额外纯前向测速。
    预热样本数大于清单样本数时，循环遍历 DataLoader 直至达到请求预热数。

    产物（写入 ``output_dir``）：
      - ``predictions.npz`` / ``predictions.csv``: 全样本预测与真值
      - ``scatter_temperature.png``、``field_compare/``（仅 enable_plots）
      - ``metrics.json``: 聚合指标
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = Path(manifest_path)
    checkpoint_path = Path(checkpoint_path)

    if sync_config_from_checkpoint:
        sync_config_for_inference(
            cfg,
            checkpoint_path,
            rule_csv_path=rule_csv_path,
        )

    dataset = AITemperatureDataset(manifest_path, config=cfg)
    if len(dataset) == 0:
        metrics = {"samples": 0, "artifacts_dir": str(output_dir)}
        (output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return metrics

    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    device = _default_device(cfg)
    model = _build_model_from_cfg(cfg).to(device)
    load_model_state_strict(model, checkpoint_path, map_location=device)
    model.eval()

    loss_helper = ReconstructionTrainer(cfg) if compute_loss else None

    sample_ids = [str(r.get("sample_id", f"idx_{i}")) for i, r in enumerate(dataset.records)]
    material_keys = [str(r.get("material_key", "")) for r in dataset.records]
    sources = [str(r.get("source", "")) for r in dataset.records]

    waveform_mean_chunks: list[np.ndarray] = []
    waveform_std_chunks: list[np.ndarray] = []
    temp_pred_chunks: list[np.ndarray] = []
    temp_true_chunks: list[np.ndarray] = []
    field_pred_chunks: list[np.ndarray] = []
    field_true_chunks: list[np.ndarray] = []
    field_mask_chunks: list[np.ndarray] = []
    acoustic_pred_chunks: list[np.ndarray] = []
    acoustic_true_chunks: list[np.ndarray] = []

    acc = {
        "total_loss": 0.0,
        "field_loss": 0.0,
        "acoustic_loss": 0.0,
        "temperature_loss": 0.0,
        "smoothness_loss": 0.0,
        "physics_residual_loss": 0.0,
    }
    n_samples = 0

    _synchronize_if_cuda(device)
    eval_t0 = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            waveform_norm, wf_mean, wf_std = prepare_model_waveform_input(batch["waveform"])
            outputs = model(waveform=waveform_norm)
            outputs = postprocess_model_outputs(outputs, wf_mean, wf_std)
            batch_size = int(batch["waveform"].shape[0])
            waveform_mean_chunks.append(wf_mean.detach().cpu().numpy())
            waveform_std_chunks.append(wf_std.detach().cpu().numpy())
            n_samples += batch_size
            if compute_loss and loss_helper is not None:
                loss, stats = loss_helper._loss(outputs, batch)
                acc["total_loss"] += float(loss.detach().cpu()) * batch_size
                for key in (
                    "field_loss",
                    "acoustic_loss",
                    "temperature_loss",
                    "smoothness_loss",
                    "physics_residual_loss",
                ):
                    acc[key] += float(stats[key]) * batch_size
            temp_pred_chunks.append(outputs["temperature"].detach().cpu().numpy())
            temp_true_chunks.append(batch["temperature"].detach().cpu().numpy())
            field_pred_chunks.append(outputs["field"].detach().cpu().numpy())
            field_true_chunks.append(batch["field"].detach().cpu().numpy())
            field_mask_chunks.append(batch["field_mask"].detach().cpu().numpy())
            acoustic_pred_chunks.append(outputs["acoustic"].detach().cpu().numpy())
            acoustic_true_chunks.append(batch["acoustic"].detach().cpu().numpy())
    _synchronize_if_cuda(device)
    eval_seconds = time.perf_counter() - eval_t0

    temp_pred = np.concatenate(temp_pred_chunks, axis=0).reshape(-1)
    temp_true = np.concatenate(temp_true_chunks, axis=0).reshape(-1)
    field_pred = np.concatenate(field_pred_chunks, axis=0)
    field_true = np.concatenate(field_true_chunks, axis=0)
    field_mask = np.concatenate(field_mask_chunks, axis=0).reshape(-1)
    acoustic_pred = np.concatenate(acoustic_pred_chunks, axis=0)
    acoustic_true = np.concatenate(acoustic_true_chunks, axis=0)
    waveform_mean = np.concatenate(waveform_mean_chunks, axis=0) if waveform_mean_chunks else np.array([])
    waveform_std = np.concatenate(waveform_std_chunks, axis=0) if waveform_std_chunks else np.array([])

    np.savez(
        output_dir / "predictions.npz",
        sample_ids=np.array(sample_ids[: len(temp_pred)], dtype=object),
        sources=np.array(sources[: len(temp_pred)], dtype=object),
        material_keys=np.array(material_keys[: len(temp_pred)], dtype=object),
        temperature_pred=temp_pred,
        temperature_true=temp_true,
        field_pred=field_pred,
        field_true=field_true,
        field_mask=field_mask,
        acoustic_pred=acoustic_pred,
        acoustic_true=acoustic_true,
        waveform_zscore_mean=waveform_mean,
        waveform_zscore_std=waveform_std,
    )

    csv_path = output_dir / "predictions.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "sample_id",
                "source",
                "material_key",
                "temperature_true_K",
                "temperature_pred_K",
                "abs_error_K",
                "has_field_label",
            ]
        )
        for i in range(len(temp_pred)):
            sid = sample_ids[i] if i < len(sample_ids) else f"idx_{i}"
            src = sources[i] if i < len(sources) else ""
            mat = material_keys[i] if i < len(material_keys) else ""
            writer.writerow(
                [
                    sid,
                    src,
                    mat,
                    f"{float(temp_true[i]):.6f}",
                    f"{float(temp_pred[i]):.6f}",
                    f"{abs(float(temp_pred[i]) - float(temp_true[i])):.6f}",
                    int(field_mask[i] > 0.5),
                ]
            )

    if enable_plots:
        _plot_scatter(temp_true, temp_pred, output_dir / "scatter_temperature.png")

    if enable_plots and plt is not None and num_field_samples > 0:
        # 优先挑有温度场标签的样本（field_mask==1，通常是仿真样本）
        valid_idx = np.where(field_mask > 0.5)[0]
        if valid_idx.size == 0:
            valid_idx = np.arange(len(temp_pred))
        pick_count = int(min(num_field_samples, valid_idx.size))
        pick = valid_idx[:pick_count]
        fields_dir = output_dir / "field_compare"
        fields_dir.mkdir(parents=True, exist_ok=True)
        for k, i in enumerate(pick):
            sid = sample_ids[i] if i < len(sample_ids) else f"idx_{i}"
            _plot_field_triptych(
                sample_id=sid,
                true_field=field_true[i],
                pred_field=field_pred[i],
                output_path=fields_dir / f"{k:02d}_{sid}.png",
            )

    temperature_mae, temperature_rmse, temperature_max_error = compute_prediction_metrics(
        temp_true,
        temp_pred,
    )
    denom = max(n_samples, 1)
    metrics: dict[str, Any] = {
        "samples": n_samples,
        "compute_loss": bool(compute_loss),
        "enable_plots": bool(enable_plots),
        "enable_benchmark": bool(enable_benchmark),
        "temperature_mae": temperature_mae,
        "temperature_rmse": temperature_rmse,
        "temperature_max_error": temperature_max_error,
        "metrics_backend": backend_name(),
        "eval_total_seconds": float(eval_seconds),
        "eval_latency_ms_per_sample": float((eval_seconds * 1000.0 / denom) if denom > 0 else float("nan")),
        "eval_throughput_samples_per_sec": float((n_samples / eval_seconds) if eval_seconds > 0 else float("nan")),
        "artifacts_dir": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
        "inference_config": {
            "hidden_dim": int(cfg.hidden_dim),
            "latent_dim": int(cfg.latent_dim),
            "field_grid_2d": [int(cfg.field_grid_2d[0]), int(cfg.field_grid_2d[1])],
            "learnable_branch_weights": bool(cfg.learnable_branch_weights),
            "fixed_weight_cnn": float(cfg.fixed_weight_cnn),
            "fixed_weight_lstm": float(cfg.fixed_weight_lstm),
            "preprocess_steps": str(cfg.preprocess_steps),
            "clip_quantile": float(cfg.clip_quantile),
            "smooth_window": int(cfg.smooth_window),
            "device": str(cfg.device),
        },
    }
    if compute_loss:
        metrics.update(
            {
                "total_loss": acc["total_loss"] / denom,
                "field_loss": acc["field_loss"] / denom,
                "acoustic_loss": acc["acoustic_loss"] / denom,
                "temperature_loss": acc["temperature_loss"] / denom,
                "smoothness_loss": acc["smoothness_loss"] / denom,
                "physics_residual_loss": acc["physics_residual_loss"] / denom,
            }
        )
    if enable_benchmark:
        metrics.update(
            _benchmark_inference_only(
                model=model,
                loader=loader,
                device=device,
                warmup_requested=int(benchmark_warmup_samples),
                benchmark_runs=int(benchmark_runs),
            )
        )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics
