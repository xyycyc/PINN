from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AIModelConfig:
    """Centralized config for the `ai_model` subproject."""

    # 项目根 = `ai_model` 包目录（本文件位于 ai_model/config/）
    repo_root: Path = field(default_factory=lambda: Path(__file__).resolve().parents[1])
    # 原 `data/` 目录已合并：默认磁盘根为 `database/`（manifest、waveforms、fields 等）。字段名 `data_root` 仍表示该输入根目录。
    data_root: Path | str = "database"
    result_root: Path | str = "result"
    waveform_length: int = 512
    field_grid_1d: int = 64
    field_grid_2d: tuple[int, int] = (24, 24)
    latent_dim: int = 128
    hidden_dim: int = 128
    learnable_branch_weights: bool = False
    a = 0.75
    fixed_weight_cnn: float = 1.0 * a
    fixed_weight_lstm: float = 1.0 * a
    fixed_weight_material: float = 1.0
    fixed_weight_dimension: float = 1.0
    fixed_weight_mode: float = 1.0
    training_mode: str = "normal"
    physics_residual_weight: float = 0.1
    batch_size: int = 16
    learning_rate: float = 1e-3
    epochs: int = 5000
    early_stopping_patience: int = 10
    online_epochs: int = 5
    device: str = "cuda"
    min_simulation_samples: int = 1000
    min_experiment_samples: int = 20
    min_temperature_k: float = 293.15
    max_temperature_k: float = 1500.0
    random_seed: int = 42
    # 训练/推理时在模型输入前应用的实验波形可选预处理（逗号分隔，不含 zscore）
    preprocess_steps: str = ""
    clip_quantile: float = 1.0
    smooth_window: int = 11

    def __post_init__(self) -> None:
        self.data_root = self.resolve_path(self.data_root)
        self.result_root = self.resolve_path(self.result_root)

    def resolve_path(self, path_like: str | Path) -> Path:
        path = Path(path_like)
        if path.is_absolute():
            return path
        if path.parts and path.parts[0].casefold() == self.repo_root.name.casefold():
            return (self.repo_root.parent / path).resolve()
        return self.repo_root / path

    @property
    def database_dir(self) -> Path:
        """与旧版 `data/cache/database` 等价内容，现扁平为 `data_root` 根目录本身。"""
        return self.data_root

    @property
    def train_checkpoint_root(self) -> Path:
        return self.result_root / "train" / "checkpoint"

    @property
    def train_report_root(self) -> Path:
        return self.result_root / "train" / "report"

    @property
    def predict_batch_root(self) -> Path:
        return self.result_root / "predict" / "batch"

    @property
    def predict_inference_root(self) -> Path:
        return self.result_root / "predict" / "inference"

    @property
    def checkpoint_dir(self) -> Path:
        # Backward-compatible alias: new root is result/train/checkpoint.
        return self.train_checkpoint_root

    @property
    def report_dir(self) -> Path:
        # Backward-compatible alias: new root is result/train/report.
        return self.train_report_root

    @property
    def output_root(self) -> Path:
        # Backward-compatible alias for legacy callers.
        return self.result_root

    @output_root.setter
    def output_root(self, value: Path | str) -> None:
        self.result_root = self.resolve_path(value)

    def ensure_dirs(self) -> None:
        for path in (
            self.data_root,
            self.data_root / "raw",
            self.result_root,
            self.database_dir,
            self.train_checkpoint_root,
            self.train_report_root,
            self.predict_batch_root,
            self.predict_inference_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
