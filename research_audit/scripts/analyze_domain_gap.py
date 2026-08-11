"""Run leakage-aware, lightweight wumu simulation-vs-experiment diagnostics."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from audit_utils import FIGURE_ROOT, STATS_ROOT, auc_score, ensure_output_dirs, write_json


RAW_FEATURES = [
    "log_rms", "log_peak_abs", "log_energy", "crest_factor", "skewness", "kurtosis",
    "zero_crossing_rate", "peak_time_fraction", "dominant_frequency_fraction",
    "spectral_centroid_fraction", "spectral_bandwidth_fraction",
]
NORMALIZED_FEATURES = [
    "crest_factor", "skewness", "kurtosis", "zero_crossing_rate", "peak_time_fraction",
    "dominant_frequency_fraction", "spectral_centroid_fraction", "spectral_bandwidth_fraction",
]


def load_records() -> list[dict[str, Any]]:
    with (STATS_ROOT / "waveform_statistics.csv").open("r", encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for row in raw:
        if row["dataset"] not in {"wumu", "wumu_post0_waveforms"}:
            continue
        converted: dict[str, Any] = dict(row)
        for key in (
            "rms", "peak_abs", "energy_mean_square", "crest_factor", "skewness", "kurtosis",
            "zero_crossing_rate", "peak_time_fraction", "dominant_frequency_fraction",
            "spectral_centroid_fraction", "spectral_bandwidth_fraction", "temperature_value",
        ):
            converted[key] = float(row[key])
        converted["log_rms"] = math.log10(max(converted["rms"], 1e-30))
        converted["log_peak_abs"] = math.log10(max(converted["peak_abs"], 1e-30))
        converted["log_energy"] = math.log10(max(converted["energy_mean_square"], 1e-30))
        converted["label"] = 1 if row["domain"] == "experiment" else 0
        rows.append(converted)
    sim = [row for row in rows if row["label"] == 0]
    exp = [row for row in rows if row["label"] == 1]
    sim = sorted(sim, key=lambda row: row["temperature_value"])
    indices = np.linspace(0, len(sim) - 1, len(exp), dtype=int)
    return [sim[int(index)] for index in indices] + exp


def fold_for_group(domain: int, group: str, folds: int = 5) -> int:
    digest = hashlib.sha256(f"{domain}:{group}:42".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % folds


def sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def train_logistic(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    design = np.column_stack((np.ones(len(x)), x))
    weights = np.zeros(design.shape[1], dtype=float)
    for step in range(1800):
        prob = sigmoid(design @ weights)
        gradient = design.T @ (prob - y) / len(y)
        gradient[1:] += 0.002 * weights[1:]
        rate = 0.12 / (1.0 + step / 900.0)
        weights -= rate * gradient
    return weights


def metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    pred = (scores >= 0.5).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    recall_pos = tp / max(tp + fn, 1)
    recall_neg = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    return {
        "accuracy": (tp + tn) / len(y),
        "balanced_accuracy": (recall_pos + recall_neg) / 2,
        "auc": auc_score(y, scores),
        "f1": 2 * precision * recall_pos / max(precision + recall_pos, 1e-30),
    }


def cross_validate(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[dict[str, Any], np.ndarray]:
    x = np.asarray([[row[name] for name in feature_names] for row in rows], dtype=float)
    y = np.asarray([row["label"] for row in rows], dtype=int)
    folds = np.asarray([fold_for_group(int(row["label"]), str(row["physical_group"])) for row in rows])
    scores = np.full(len(rows), np.nan, dtype=float)
    fold_metrics: list[dict[str, Any]] = []
    for fold in range(5):
        train = folds != fold
        test = folds == fold
        if len(np.unique(y[test])) < 2:
            continue
        mean = np.mean(x[train], axis=0)
        std = np.std(x[train], axis=0)
        std[std < 1e-12] = 1.0
        weights = train_logistic((x[train] - mean) / std, y[train])
        test_design = np.column_stack((np.ones(np.sum(test)), (x[test] - mean) / std))
        scores[test] = sigmoid(test_design @ weights)
        fold_metrics.append({"fold": fold, "test_samples": int(np.sum(test)), **metrics(y[test], scores[test])})
    valid = np.isfinite(scores)
    aggregate = metrics(y[valid], scores[valid])
    return {
        **aggregate,
        "sample_count": int(np.sum(valid)),
        "simulation_count": int(np.sum(y[valid] == 0)),
        "experiment_count": int(np.sum(y[valid] == 1)),
        "features": feature_names,
        "split": "5-fold deterministic group split; experiment repeats grouped by filename temperature",
        "folds": fold_metrics,
    }, scores


def standardized_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    x = np.asarray([[row[name] for name in feature_names] for row in rows], dtype=float)
    std = np.std(x, axis=0)
    std[std < 1e-12] = 1.0
    return (x - np.mean(x, axis=0)) / std


def mmd_rbf(x: np.ndarray, y: np.ndarray) -> float:
    combined = np.vstack((x, y))
    diff = combined[:, None, :] - combined[None, :, :]
    distance = np.sum(diff * diff, axis=-1)
    positive = distance[distance > 0]
    bandwidth = float(np.median(positive)) if len(positive) else 1.0
    kernel = np.exp(-distance / max(2.0 * bandwidth, 1e-12))
    n = len(x)
    return float(np.mean(kernel[:n, :n]) + np.mean(kernel[n:, n:]) - 2.0 * np.mean(kernel[:n, n:]))


def wasserstein_quantile_mean(x: np.ndarray, y: np.ndarray) -> float:
    quantiles = np.linspace(0.01, 0.99, 99)
    distances = [np.mean(np.abs(np.quantile(x[:, i], quantiles) - np.quantile(y[:, i], quantiles))) for i in range(x.shape[1])]
    return float(np.mean(distances))


def canvas(title: str, x_label: str, y_label: str) -> tuple[Image.Image, ImageDraw.ImageDraw, tuple[int, int, int, int]]:
    image = Image.new("RGB", (1000, 700), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((30, 20), title, fill="black", font=font)
    box = (90, 70, 960, 620)
    draw.rectangle(box, outline="#444444", width=2)
    draw.text((450, 650), x_label, fill="black", font=font)
    draw.text((10, 340), y_label, fill="black", font=font)
    return image, draw, box


def scale(values: np.ndarray, low: int, high: int, invert: bool = False) -> np.ndarray:
    minimum, maximum = float(np.min(values)), float(np.max(values))
    result = np.full(len(values), (low + high) / 2.0) if maximum <= minimum else low + (values - minimum) * (high - low) / (maximum - minimum)
    return high - (result - low) if invert else result


def plot_pca(rows: list[dict[str, Any]], x: np.ndarray) -> tuple[np.ndarray, list[float]]:
    _, singular, vt = np.linalg.svd(x, full_matrices=False)
    coords = x @ vt[:2].T
    variance = singular * singular
    explained = (variance / np.sum(variance))[:2]
    image, draw, box = canvas("Contract-normalized waveform feature PCA", f"PC1 ({explained[0]*100:.1f}%)", f"PC2 ({explained[1]*100:.1f}%)")
    px = scale(coords[:, 0], box[0] + 10, box[2] - 10)
    py = scale(coords[:, 1], box[1] + 10, box[3] - 10, invert=True)
    for index, row in enumerate(rows):
        color = "#d62728" if row["label"] else "#1f77b4"
        draw.ellipse((px[index]-3, py[index]-3, px[index]+3, py[index]+3), fill=color)
    draw.text((110, 85), "blue: simulation   red: experiment", fill="black")
    image.save(FIGURE_ROOT / "wumu_sim_real_pca.png")
    return coords, [float(value) for value in explained]


def plot_feature_gap(rows: list[dict[str, Any]]) -> None:
    image, draw, box = canvas("Wumu RMS raw-scale separation", "log10(RMS)", "density/count")
    sim = np.asarray([row["log_rms"] for row in rows if row["label"] == 0])
    exp = np.asarray([row["log_rms"] for row in rows if row["label"] == 1])
    low, high = min(np.min(sim), np.min(exp)), max(np.max(sim), np.max(exp))
    bins = np.linspace(low, high, 35)
    for values, color in ((sim, "#1f77b4"), (exp, "#d62728")):
        hist, _ = np.histogram(values, bins=bins)
        hist = hist / max(np.max(hist), 1)
        points = []
        for i, value in enumerate(hist):
            x = box[0] + (i + 0.5) / len(hist) * (box[2] - box[0])
            y = box[3] - value * (box[3] - box[1] - 20)
            points.append((x, y))
        draw.line(points, fill=color, width=3)
    draw.text((110, 85), "blue: simulation   red: experiment", fill="black")
    image.save(FIGURE_ROOT / "wumu_rms_domain_gap.png")


def plot_temperature_trend(rows: list[dict[str, Any]]) -> None:
    image, draw, box = canvas("Temperature vs peak-time fraction", "temperature (K; experiment filename C converted)", "peak-time fraction")
    x = np.asarray([row["temperature_value"] + (273.15 if row["label"] else 0.0) for row in rows])
    y = np.asarray([row["peak_time_fraction"] for row in rows])
    px = scale(x, box[0] + 10, box[2] - 10)
    py = scale(y, box[1] + 10, box[3] - 10, invert=True)
    for index, row in enumerate(rows):
        color = "#d62728" if row["label"] else "#1f77b4"
        draw.ellipse((px[index]-3, py[index]-3, px[index]+3, py[index]+3), fill=color)
    draw.text((110, 85), "blue: simulation   red: experiment (label semantics unconfirmed)", fill="black")
    image.save(FIGURE_ROOT / "wumu_temperature_peak_time.png")


def plot_roc(rows: list[dict[str, Any]], scores: np.ndarray) -> None:
    y = np.asarray([row["label"] for row in rows], dtype=int)
    valid = np.isfinite(scores)
    y, scores = y[valid], scores[valid]
    thresholds = np.r_[np.inf, np.sort(np.unique(scores))[::-1], -np.inf]
    points = []
    for threshold in thresholds:
        pred = scores >= threshold
        tpr = np.sum(pred & (y == 1)) / max(np.sum(y == 1), 1)
        fpr = np.sum(pred & (y == 0)) / max(np.sum(y == 0), 1)
        points.append((float(fpr), float(tpr)))
    image, draw, box = canvas("Contract-normalized domain classifier ROC", "false-positive rate", "true-positive rate")
    draw.line((box[0], box[3], box[2], box[1]), fill="#999999", width=2)
    scaled = [(box[0] + fpr*(box[2]-box[0]), box[3] - tpr*(box[3]-box[1])) for fpr, tpr in points]
    draw.line(scaled, fill="#9467bd", width=4)
    image.save(FIGURE_ROOT / "wumu_domain_classifier_roc.png")


def main() -> int:
    ensure_output_dirs()
    rows = load_records()
    raw_metrics, _ = cross_validate(rows, RAW_FEATURES)
    normalized_metrics, normalized_scores = cross_validate(rows, NORMALIZED_FEATURES)
    matrix = standardized_matrix(rows, NORMALIZED_FEATURES)
    labels = np.asarray([row["label"] for row in rows], dtype=int)
    sim, exp = matrix[labels == 0], matrix[labels == 1]
    coords, explained = plot_pca(rows, matrix)
    plot_feature_gap(rows)
    plot_temperature_trend(rows)
    plot_roc(rows, normalized_scores)
    payload = {
        "diagnostic_only": True,
        "comparison": "wumu simulation vs wumu post-zero processed experiment",
        "raw_feature_classifier": raw_metrics,
        "contract_normalized_classifier": normalized_metrics,
        "accuracy": normalized_metrics["accuracy"],
        "balanced_accuracy": normalized_metrics["balanced_accuracy"],
        "auc": normalized_metrics["auc"],
        "f1": normalized_metrics["f1"],
        "mmd_rbf_standardized_features": mmd_rbf(sim, exp),
        "mean_feature_wasserstein_standardized": wasserstein_quantile_mean(sim, exp),
        "pca_explained_variance_ratio": explained,
        "interpretation": "High separability after contract-normalized shape/spectral features indicates a substantive gap beyond amplitude units alone.",
    }
    write_json(STATS_ROOT / "domain_classifier_metrics.json", payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
