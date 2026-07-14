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
from ..data_process import (
    AITemperatureDataset,
    MATERIAL_COLLECTION_KIND,
    load_material_collection,
    resolve_collection_manifest,
)
from ..fortran import backend_name, compute_prediction_metrics
from .network import AIReconstructionModel
from .point_field import (
    SUPPORTED_POINT_FIELD_CHECKPOINT_VERSIONS,
    DirectPointFieldModel,
    kelvin_metrics,
    load_compatible_point_field_state,
)
from .checkpoint_runtime import load_model_state_strict, sync_config_for_inference
from .trainer import (
    MATERIAL_ROUTER_KIND,
    MATERIAL_ROUTER_VERSION,
    ReconstructionTrainer,
    _default_device,
)
from .waveform_io import postprocess_model_outputs, prepare_model_waveform_input

try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except Exception:  # pragma: no cover
    plt = None
    FuncFormatter = None

def _format_integer_colorbar(colorbar: Any) -> Any:
    """Display Kelvin colorbar ticks as ordinary integers without an offset."""
    if FuncFormatter is not None:
        colorbar.formatter = FuncFormatter(lambda value, _position: f"{value:.0f}")
        colorbar.update_ticks()
        colorbar.ax.yaxis.get_offset_text().set_visible(False)
    return colorbar


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
    error_field = pred_field - true_field
    vmin = float(min(true_field.min(), pred_field.min()))
    vmax = float(max(true_field.max(), pred_field.max()))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    im0 = axes[0].imshow(
        true_field,
        vmin=vmin,
        vmax=vmax,
        cmap="inferno",
        aspect="auto",
    )
    axes[0].set_title("true K")
    _format_integer_colorbar(
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    )
    im1 = axes[1].imshow(
        pred_field,
        vmin=vmin,
        vmax=vmax,
        cmap="inferno",
        aspect="auto",
    )
    axes[1].set_title("predicted K")
    _format_integer_colorbar(
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    )
    im2 = axes[2].imshow(error_field, cmap="seismic", aspect="auto")
    axes[2].set_title("error K")
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle(sample_id, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _plot_point_field_triptych(
    sample_id: str,
    coordinates_m: np.ndarray,
    true_k: np.ndarray,
    predicted_k: np.ndarray,
    output_path: Path,
) -> None:
    if plt is None:
        return
    coordinates = np.asarray(coordinates_m)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    series = (
        (np.asarray(true_k), "true K", "inferno"),
        (np.asarray(predicted_k), "predicted K", "inferno"),
        (np.asarray(predicted_k) - np.asarray(true_k), "error K", "seismic"),
    )
    for axis, (values, title, cmap) in zip(axes, series):
        image = axis.scatter(
            coordinates[:, 0],
            coordinates[:, 1],
            c=values,
            s=3,
            cmap=cmap,
        )
        axis.set_title(title)
        colorbar = fig.colorbar(image, ax=axis)
        if title != "error K":
            _format_integer_colorbar(colorbar)
    fig.suptitle(sample_id, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
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


def _benchmark_point_models(
    loader: DataLoader,
    device: torch.device,
    forward_waveform: Any,
    *,
    warmup_requested: int,
    benchmark_runs: int,
) -> dict[str, float]:
    """Benchmark fixed-node forward passes without artifact or metric overhead."""

    warmup_steps = max(0, int(warmup_requested))
    benchmark_runs = max(0, int(benchmark_runs))
    if benchmark_runs <= 0:
        return {}

    def run_batch(batch: dict[str, torch.Tensor]) -> int:
        waveform = batch["waveform"].to(device)
        waveform_norm, _, _ = prepare_model_waveform_input(waveform)
        forward_waveform(waveform_norm)
        return int(waveform.shape[0])

    if warmup_steps > 0:
        warmup_done = 0
        with torch.no_grad():
            while warmup_done < warmup_steps:
                for batch in loader:
                    warmup_done += run_batch(batch)
                    if warmup_done >= warmup_steps:
                        break
        _synchronize_if_cuda(device)

    elapsed_seconds = 0.0
    seen_samples = 0
    repeats_done = 0
    with torch.no_grad():
        while repeats_done < benchmark_runs:
            for batch in loader:
                if repeats_done >= benchmark_runs:
                    break
                _synchronize_if_cuda(device)
                t0 = time.perf_counter()
                sample_count = run_batch(batch)
                _synchronize_if_cuda(device)
                elapsed_seconds += time.perf_counter() - t0
                seen_samples += sample_count
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

    if dataset.is_point_field:
        return _predict_point_field(
            cfg, dataset, checkpoint_path, manifest_path, output_dir,
            enable_plots=enable_plots,
            num_field_samples=num_field_samples,
            enable_benchmark=enable_benchmark,
            benchmark_warmup_samples=benchmark_warmup_samples,
            benchmark_runs=benchmark_runs,
        )

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


def _finalize_point_field_prediction(
    dataset: AITemperatureDataset,
    manifest_path: Path,
    output_dir: Path,
    prediction_k: np.ndarray,
    target_k: np.ndarray,
    *,
    eval_seconds: float,
    model_kind: str,
    checkpoint_version: int,
    checkpoint_label: str,
    parameter_count: int,
    parameter_bytes: int,
    peak_cuda_memory_bytes: int,
    enable_plots: bool,
    num_field_samples: int,
    enable_benchmark: bool,
    benchmark_warmup_samples: int,
    benchmark_runs: int,
    benchmark_metrics: dict[str, float] | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write the common artifacts for one complete full-field model."""
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    with np.load(dataset.root / str(manifest_payload["sampling_index"]), allow_pickle=False) as sampling:
        node_ids = np.asarray(sampling["node_ids"], np.int64)
        coordinates = np.asarray(sampling["coordinates_m"], np.float32)
        material_ids = np.asarray(sampling["material_ids"], np.int32)
        interface_side = np.asarray(sampling["interface_side"], np.int32)
        sample_weights = np.asarray(sampling["sample_weights"], np.float32)
    lo, hi = coordinates.min(axis=0), coordinates.max(axis=0)
    boundary_tolerance = np.maximum((hi-lo)*1e-7, 1e-9)
    boundary = np.any((np.abs(coordinates-lo) <= boundary_tolerance) |
                      (np.abs(coordinates-hi) <= boundary_tolerance), axis=1)
    masks: dict[str, np.ndarray] = {"boundary": boundary, "interface": interface_side > 0}
    for material in np.unique(material_ids):
        masks[f"material_{int(material)}"] = material_ids == material
    high_threshold = float(np.quantile(target_k, 0.9))
    metrics = kelvin_metrics(prediction_k, target_k, masks=masks)
    metrics.update({
        "samples": len(dataset), "point_count": int(dataset.point_count),
        "waveform_length": int(dataset.waveform_length),
        "model_kind": model_kind, "checkpoint_version": int(checkpoint_version),
        "high_temperature_threshold_k": high_threshold,
        "high_temperature": kelvin_metrics(prediction_k[target_k >= high_threshold], target_k[target_k >= high_threshold])["full_field"],
        "eval_total_seconds": float(eval_seconds),
        "eval_latency_ms_per_sample": float(eval_seconds * 1000.0 / max(len(dataset), 1)),
        "parameter_count": int(parameter_count),
        "parameter_bytes": int(parameter_bytes),
        "peak_cuda_memory_bytes": int(peak_cuda_memory_bytes),
        "enable_plots": bool(enable_plots),
        "enable_benchmark": bool(enable_benchmark),
        "checkpoint": checkpoint_label, "manifest": str(manifest_path),
    })
    if metadata_extra:
        metrics.update(metadata_extra)
    np.savez_compressed(
        output_dir / "predictions.npz", prediction_temperature_k=prediction_k,
        target_temperature_k=target_k, node_ids=node_ids, coordinates_m=coordinates,
        material_ids=material_ids, interface_side=interface_side, sample_weights=sample_weights,
        sample_ids=np.array([str(r.get("sample_id", "")) for r in dataset.records]),
    )
    with (output_dir / "predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "node_id", "x_m", "y_m", "temperature_k",
                         "target_temperature_k", "constituent_material_id", "interface_side"])
        for case_index, record in enumerate(dataset.records):
            sample_id = str(record.get("sample_id", f"idx_{case_index}"))
            for point in range(dataset.point_count):
                writer.writerow([sample_id, int(node_ids[point]), f"{coordinates[point,0]:.9g}",
                                 f"{coordinates[point,1]:.9g}", f"{prediction_k[case_index,point]:.7g}",
                                 f"{target_k[case_index,point]:.7g}", int(material_ids[point]), int(interface_side[point])])
    metadata = {
        "checkpoint": checkpoint_label, "checkpoint_version": int(checkpoint_version),
        "model_kind": model_kind, "schema_version": int(dataset.schema_version),
        "waveform_length": int(dataset.waveform_length),
        "normalization": dict(dataset.normalization), "sampling": dict(dataset.sampling_metadata),
        "temperature_unit": "K", "coordinate_unit": "m",
        "raw_prediction_table": "predictions.csv",
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    plotted_samples = 0
    if enable_plots and plt is not None and int(num_field_samples) > 0:
        compare_dir = output_dir / "field_compare"
        compare_dir.mkdir(parents=True, exist_ok=True)
        plot_count = min(int(num_field_samples), len(dataset))
        for sample_index in range(plot_count):
            sample_id = str(dataset.records[sample_index].get("sample_id", f"idx_{sample_index}"))
            safe_id = "".join(
                char if char.isalnum() or char in {"-", "_"} else "_"
                for char in sample_id
            ).strip("_") or f"idx_{sample_index}"
            compare_path = compare_dir / f"{sample_index:04d}_{safe_id}.png"
            _plot_point_field_triptych(
                sample_id=sample_id,
                coordinates_m=coordinates,
                true_k=target_k[sample_index],
                predicted_k=prediction_k[sample_index],
                output_path=compare_path,
            )
            if sample_index == 0:
                # Keep the original single-image artifact for downstream users.
                (output_dir / "point_field_compare.png").write_bytes(
                    compare_path.read_bytes()
                )
            plotted_samples += 1
    metrics["field_plot_samples"] = int(plotted_samples)
    if enable_benchmark and benchmark_metrics:
        metrics.update(benchmark_metrics)
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def _predict_point_field(
    cfg: AIModelConfig,
    dataset: AITemperatureDataset,
    checkpoint_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    enable_plots: bool,
    num_field_samples: int,
    enable_benchmark: bool,
    benchmark_warmup_samples: int,
    benchmark_runs: int,
) -> dict[str, Any]:
    """Predict fixed physical nodes and preserve the raw node table."""
    device = _default_device(cfg)
    bundle = torch.load(checkpoint_path, map_location=device)
    if not isinstance(bundle, dict) or bundle.get("model_kind") != "direct_point_field":
        raise ValueError("固定节点 manifest 需要 model_kind='direct_point_field' 的版本化 checkpoint")
    checkpoint_version = int(bundle.get("checkpoint_version", 0))
    if checkpoint_version not in SUPPORTED_POINT_FIELD_CHECKPOINT_VERSIONS:
        raise ValueError(f"不支持的固定节点 checkpoint_version: {bundle.get('checkpoint_version')}")
    checkpoint_material = str(bundle.get("sample_material_key", "") or "").strip()
    manifest_materials = {
        str(record.get("material_key", "")).strip()
        for record in dataset.records
    }
    if checkpoint_material and manifest_materials != {checkpoint_material}:
        raise ValueError(
            "checkpoint/manifest 的样本级材料不一致: "
            f"{checkpoint_material!r} != {sorted(manifest_materials)}"
        )
    point_count = int(bundle.get("point_count", 0))
    if point_count != dataset.point_count:
        raise ValueError(f"checkpoint/manifest 点数不一致: {point_count} != {dataset.point_count}")
    checkpoint_waveform_length = int(
        bundle.get("waveform_length", bundle.get("config", {}).get("waveform_length", 0))
    )
    if checkpoint_waveform_length and checkpoint_waveform_length != dataset.waveform_length:
        raise ValueError(
            "checkpoint/manifest 波形长度不一致: "
            f"{checkpoint_waveform_length} != {dataset.waveform_length}"
        )
    checkpoint_norm = bundle.get("normalization", {})
    for key in ("mean_k", "std_k"):
        if not np.isclose(float(checkpoint_norm.get(key, np.nan)), float(dataset.normalization.get(key, np.nan))):
            raise ValueError(f"checkpoint/manifest 温度标准化参数不一致: {key}")
    checkpoint_sampling = bundle.get("sampling_metadata", {})
    for key in ("sampling_version", "source_mesh_fingerprint"):
        if checkpoint_sampling.get(key) != dataset.sampling_metadata.get(key):
            raise ValueError(f"checkpoint/manifest 采样定义不一致: {key}")
    model = DirectPointFieldModel(
        point_count=point_count,
        hidden_dim=cfg.hidden_dim,
        latent_dim=cfg.latent_dim,
        chunk_size=int(bundle.get("chunk_size", 1000)),
        learnable_branch_weights=cfg.learnable_branch_weights,
        fixed_weight_cnn=cfg.fixed_weight_cnn,
        fixed_weight_lstm=cfg.fixed_weight_lstm,
    ).to(device)
    legacy_missing = load_compatible_point_field_state(model, bundle["model_state"])
    if not legacy_missing and not model.learnable_branch_weights:
        with torch.no_grad():
            model.weight_cnn.fill_(float(cfg.fixed_weight_cnn))
            model.weight_lstm.fill_(float(cfg.fixed_weight_lstm))
    model.eval()
    loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False)
    pred_chunks: list[np.ndarray] = []
    true_chunks: list[np.ndarray] = []
    eval_start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            waveform = batch["waveform"].to(device)
            waveform_norm, _, _ = prepare_model_waveform_input(waveform)
            normalized = model(waveform_norm)
            pred_chunks.append((normalized * dataset.temperature_std_k + dataset.temperature_mean_k).cpu().numpy())
            true_chunks.append(batch["field_k"].numpy())
    _synchronize_if_cuda(device)
    eval_seconds = time.perf_counter() - eval_start
    prediction_k = np.concatenate(pred_chunks, axis=0)
    target_k = np.concatenate(true_chunks, axis=0)
    benchmark_metrics = (
        _benchmark_point_models(
            loader,
            device,
            model,
            warmup_requested=benchmark_warmup_samples,
            benchmark_runs=benchmark_runs,
        )
        if enable_benchmark
        else {}
    )
    return _finalize_point_field_prediction(
        dataset, manifest_path, output_dir, prediction_k, target_k,
        eval_seconds=eval_seconds,
        model_kind="direct_point_field",
        checkpoint_version=checkpoint_version,
        checkpoint_label=str(checkpoint_path.resolve()),
        parameter_count=int(sum(parameter.numel() for parameter in model.parameters())),
        parameter_bytes=int(sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())),
        peak_cuda_memory_bytes=int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0,
        enable_plots=enable_plots,
        num_field_samples=num_field_samples,
        enable_benchmark=enable_benchmark,
        benchmark_warmup_samples=benchmark_warmup_samples,
        benchmark_runs=benchmark_runs,
        benchmark_metrics=benchmark_metrics,
    )


def predict_with_material_router(
    cfg: AIModelConfig,
    router_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    *,
    rule_csv_path: str | Path | None = None,
    sync_config_from_checkpoint: bool = True,
    enable_plots: bool = False,
    num_field_samples: int = 6,
    enable_benchmark: bool = False,
    benchmark_warmup_samples: int = 64,
    benchmark_runs: int = DEFAULT_BENCHMARK_RUNS,
) -> dict[str, Any]:
    """Route each complete sample to the checkpoint for its material folder."""

    router_path = Path(router_path).resolve()
    manifest_path = Path(manifest_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    router = json.loads(router_path.read_text(encoding="utf-8"))
    if router.get("router_kind") != MATERIAL_ROUTER_KIND:
        raise ValueError(
            f"不支持的样本级材料路由类型: {router.get('router_kind')}"
        )
    if int(router.get("router_version", 0)) != MATERIAL_ROUTER_VERSION:
        raise ValueError(
            f"不支持的样本级材料路由版本: {router.get('router_version')}"
        )
    raw_specs = router.get("checkpoints", [])
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError("样本级材料路由没有 checkpoints")
    specs: dict[str, dict[str, Any]] = {}
    for raw_spec in raw_specs:
        if not isinstance(raw_spec, dict):
            raise ValueError("样本级材料路由 checkpoint 条目必须是对象")
        spec = dict(raw_spec)
        material_key = str(spec.get("material_key", "")).strip()
        if not material_key or material_key in specs:
            raise ValueError("样本级材料路由的 material_key 为空或重复")
        specs[material_key] = spec

    def checkpoint_for(material_key: str) -> Path:
        if material_key not in specs:
            raise ValueError(
                f"材料路由没有 {material_key!r} 对应的完整温度场 checkpoint"
            )
        raw = Path(str(specs[material_key].get("checkpoint", "")))
        path = raw if raw.is_absolute() else router_path.parent / raw
        if not path.is_file():
            raise FileNotFoundError(f"材料 checkpoint 不存在: {path}")
        return path.resolve()

    def predict_one(
        material_key: str,
        one_manifest: Path,
        one_output: Path,
    ) -> dict[str, Any]:
        checkpoint = checkpoint_for(material_key)
        metrics = predict_and_compare(
            cfg=cfg,
            checkpoint_path=checkpoint,
            manifest_path=one_manifest,
            output_dir=one_output,
            rule_csv_path=rule_csv_path,
            sync_config_from_checkpoint=sync_config_from_checkpoint,
            enable_plots=enable_plots,
            num_field_samples=num_field_samples,
            enable_benchmark=enable_benchmark,
            benchmark_warmup_samples=benchmark_warmup_samples,
            benchmark_runs=benchmark_runs,
        )
        metrics.update(
            {
                "routing_strategy": "records[].material_key",
                "sample_material_key": material_key,
                "material_router": str(router_path),
                "checkpoint": str(checkpoint),
            }
        )
        (one_output / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return metrics

    input_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if input_payload.get("collection_kind") == MATERIAL_COLLECTION_KIND:
        collection = load_material_collection(manifest_path)
        material_results: dict[str, dict[str, Any]] = {}
        total_samples = 0
        for raw_entry in collection["materials"]:
            entry = dict(raw_entry)
            material_key = str(entry["material_key"]).strip()
            test_manifest = resolve_collection_manifest(
                manifest_path,
                entry,
                "test",
            )
            safe_key = "".join(
                ch if ch.isalnum() or ch in {"_", "-"} else "_"
                for ch in material_key
            ).strip("_") or "material"
            metrics = predict_one(
                material_key,
                test_manifest,
                output_dir / safe_key,
            )
            material_results[material_key] = metrics
            total_samples += int(metrics.get("samples", 0))
        aggregate = {
            "model_kind": MATERIAL_ROUTER_KIND,
            "routing_strategy": "records[].material_key",
            "material_router": str(router_path),
            "source_collection": str(manifest_path),
            "samples": total_samples,
            "materials": material_results,
            "artifacts_dir": str(output_dir),
        }
        (output_dir / "metrics.json").write_text(
            json.dumps(aggregate, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return aggregate

    records = input_payload.get("records", [])
    if not isinstance(records, list) or not records:
        raise ValueError("待预测 manifest 没有记录")
    material_keys = {
        str(record.get("material_key", "")).strip()
        for record in records
        if isinstance(record, dict)
    }
    if "" in material_keys or len(material_keys) != 1:
        raise ValueError(
            "单个预测 manifest 必须且只能包含一种样本级 material_key；"
            f"实际为 {sorted(material_keys)}"
        )
    return predict_one(next(iter(material_keys)), manifest_path, output_dir)


def predict_collection_with_checkpoint(
    cfg: AIModelConfig,
    checkpoint_path: str | Path,
    collection_path: str | Path,
    output_dir: str | Path,
    *,
    rule_csv_path: str | Path | None = None,
    sync_config_from_checkpoint: bool = True,
    enable_plots: bool = False,
    num_field_samples: int = 6,
    enable_benchmark: bool = False,
    benchmark_warmup_samples: int = 64,
    benchmark_runs: int = DEFAULT_BENCHMARK_RUNS,
) -> dict[str, Any]:
    """Run one mixed checkpoint on every material test split in a collection."""
    checkpoint_path = Path(checkpoint_path).resolve()
    collection_path = Path(collection_path).resolve()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    collection = load_material_collection(collection_path)
    bundle = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(bundle, dict) or bundle.get("model_kind") != "direct_point_field":
        raise ValueError("多材料集合混合预测需要 direct_point_field checkpoint")
    checkpoint_normalization = dict(bundle.get("normalization", {}))
    material_results: dict[str, dict[str, Any]] = {}
    total_samples = 0
    for raw_entry in collection["materials"]:
        entry = dict(raw_entry)
        material_key = str(entry["material_key"]).strip()
        safe_key = "".join(
            ch if ch.isalnum() or ch in {"_", "-"} else "_"
            for ch in material_key
        ).strip("_") or "material"
        source_manifest = resolve_collection_manifest(
            collection_path,
            entry,
            "test",
        )
        source_payload = json.loads(source_manifest.read_text(encoding="utf-8"))
        sampling_path = Path(str(source_payload["sampling_index"]))
        if not sampling_path.is_absolute():
            sampling_path = source_manifest.parent / sampling_path
        derived_records: list[dict[str, Any]] = []
        for raw_record in source_payload.get("records", []):
            record = dict(raw_record)
            for path_key in ("waveform_path", "field_path"):
                raw_path = Path(str(record[path_key]))
                if not raw_path.is_absolute():
                    raw_path = source_manifest.parent / raw_path
                record[path_key] = str(raw_path.resolve())
            derived_records.append(record)
        material_output = output_dir / safe_key
        material_output.mkdir(parents=True, exist_ok=True)
        derived_manifest = material_output / "mixed_prediction_manifest.json"
        source_payload["source_manifest"] = str(source_manifest)
        source_payload["sampling_index"] = str(sampling_path.resolve())
        source_payload["normalization"] = checkpoint_normalization
        source_payload["records"] = derived_records
        derived_manifest.write_text(
            json.dumps(source_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        metrics = predict_and_compare(
            cfg=cfg,
            checkpoint_path=checkpoint_path,
            manifest_path=derived_manifest,
            output_dir=material_output,
            rule_csv_path=rule_csv_path,
            sync_config_from_checkpoint=sync_config_from_checkpoint,
            enable_plots=enable_plots,
            num_field_samples=num_field_samples,
            enable_benchmark=enable_benchmark,
            benchmark_warmup_samples=benchmark_warmup_samples,
            benchmark_runs=benchmark_runs,
        )
        metrics["sample_material_key"] = material_key
        material_results[material_key] = metrics
        total_samples += int(metrics.get("samples", 0))
    aggregate = {
        "model_kind": "mixed_sample_material_checkpoint",
        "routing_strategy": "one_shared_checkpoint",
        "checkpoint": str(checkpoint_path),
        "source_collection": str(collection_path),
        "samples": total_samples,
        "materials": material_results,
        "artifacts_dir": str(output_dir),
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return aggregate
