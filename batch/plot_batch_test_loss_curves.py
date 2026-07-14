"""
从 batch_test_modes 产物目录中读取各 mode 的 *_history.json，绘制 loss 随 epoch 的轨迹。

用法:
    python -m ai_model.batch.plot_batch_test_loss_curves --run-dir ai_model/result/batch_test_20260325_224605
    python -m ai_model.batch.plot_batch_test_loss_curves --run-dir ai_model/result/batch_test_20260325_224605 -o my_curves.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .batch_test_modes import _resolve_project_path

try:
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    plt = None


def _discover_histories(run_dir: Path) -> dict[str, list[dict]]:
    """发现当前与历史 batch_test 目录中的训练历史。"""

    histories: dict[str, list[dict]] = {}
    patterns = (
        # Current AIModelConfig layout.
        "*/result/train/report/*/*_history.json",
        # Compatibility with batch runs produced by the former output layout.
        "*/outputs/reports/*_history.json",
    )
    paths = [path for pattern in patterns for path in sorted(run_dir.glob(pattern))]
    for path in paths:
        if not path.is_file():
            continue
        stem = path.stem
        if not stem.endswith("_history"):
            continue
        mode_name = stem[: -len("_history")]
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise ValueError(f"无法解析 JSON: {path}") from e
        if not isinstance(data, list) or not data:
            continue
        # Current-layout files are visited first and take precedence if a
        # partially migrated run contains both directory layouts.
        histories.setdefault(mode_name, data)
    return histories


def _series(history: list[dict], key: str) -> tuple[list[float], list[float]]:
    epochs: list[float] = []
    values: list[float] = []
    for row in history:
        if key not in row:
            continue
        epochs.append(float(row["epoch"]))
        values.append(float(row[key]))
    return epochs, values


def plot_loss_curves(
    run_dir: Path,
    output_path: Path,
    dpi: int = 150,
) -> Path:
    if plt is None:
        raise RuntimeError("需要安装 matplotlib 才能绘图: pip install matplotlib")

    run_dir = run_dir.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"目录不存在: {run_dir}")

    histories = _discover_histories(run_dir)
    if not histories:
        raise FileNotFoundError(
            f"在 {run_dir} 下未找到训练历史（当前或历史 report 目录），"
            "请确认路径指向 batch_test_modes 生成的 run 目录。"
        )

    metrics = [
        ("mean_epoch_loss", "Mean epoch loss (加权总损失，训练集按 batch 平均)"),
        ("temperature_loss", "Temperature MSE (分项)"),
        ("acoustic_loss", "Acoustic masked MSE (分项)"),
        ("physics_residual_loss", "Physics residual (分项)"),
    ]

    n_modes = len(histories)
    cmap = plt.get_cmap("tab10" if n_modes <= 10 else "hsv")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes_flat = axes.ravel()

    for ax_idx, (key, title) in enumerate(metrics):
        ax = axes_flat[ax_idx]
        for mi, (mode_name, hist) in enumerate(sorted(histories.items())):
            ep, vals = _series(hist, key)
            if not ep:
                continue
            color = cmap(mi / max(n_modes, 1))
            ax.plot(ep, vals, label=mode_name, color=color, linewidth=1.2, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(key)
        ax.grid(True, linestyle="--", alpha=0.3)
        if ax_idx == 0:
            ax.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"Loss trajectories — {run_dir.name}", fontsize=14, y=1.02)
    fig.tight_layout()
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="绘制 batch_test 各 mode 的 loss 轨迹（读取 *_history.json）")
    p.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="batch_test_modes 生成的目录，例如 ai_model/result/batch_test_20260325_224605",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default="",
        help="输出图片路径；默认写入 <run-dir>/loss_curves.png",
    )
    p.add_argument("--dpi", type=int, default=150)
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_dir = _resolve_project_path(args.run_dir)
    out = _resolve_project_path(args.output) if args.output else run_dir / "loss_curves.png"
    path = plot_loss_curves(run_dir=run_dir, output_path=out, dpi=args.dpi)
    print(path)


if __name__ == "__main__":
    main()
