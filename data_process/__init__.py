"""数据集、实验波形预处理与数据库 manifest 构建。"""

from .builder import DataRecord, DatabaseBuilder
from .case_pipeline import build_case_dataset, discover_cases
from .dataset import (
    SPLIT_EXPERIMENT_POLICIES,
    AITemperatureDataset,
    split_manifest_file,
    split_manifest_records,
    split_case_records,
    write_case_split_files,
)
from .material_registry import (
    ensure_material_csv,
    load_material_to_idx,
    material_csv_path,
    register_material,
    register_materials,
)
from .preprocess_cache import (
    load_cached_preprocessed_waveform,
    load_or_apply_preprocessed_waveform,
    preprocess_cache_dir,
    preprocess_pipeline_label,
    save_preprocessed_waveform_cache,
)
from .preprocess import (
    FORCED_MODEL_ZSCORE_STEP,
    SUPPORTED_PREPROCESS_STEPS,
    apply_optional_preprocessing,
    apply_preprocessing_pipeline,
    apply_zscore_with_stats,
    clip_outliers,
    detrend_baseline,
    inverse_zscore,
    parse_preprocess_steps,
    robust_normalize,
    smooth_signal,
    zscore_normalize,
    zscore_stats,
)

__all__ = [
    "AITemperatureDataset",
    "SPLIT_EXPERIMENT_POLICIES",
    "DataRecord",
    "DatabaseBuilder",
    "build_case_dataset",
    "discover_cases",
    "split_manifest_file",
    "split_manifest_records",
    "split_case_records",
    "write_case_split_files",
    "ensure_material_csv",
    "load_material_to_idx",
    "material_csv_path",
    "register_material",
    "register_materials",
    "FORCED_MODEL_ZSCORE_STEP",
    "SUPPORTED_PREPROCESS_STEPS",
    "load_cached_preprocessed_waveform",
    "load_or_apply_preprocessed_waveform",
    "preprocess_cache_dir",
    "preprocess_pipeline_label",
    "save_preprocessed_waveform_cache",
    "apply_optional_preprocessing",
    "apply_preprocessing_pipeline",
    "apply_zscore_with_stats",
    "clip_outliers",
    "detrend_baseline",
    "inverse_zscore",
    "parse_preprocess_steps",
    "robust_normalize",
    "smooth_signal",
    "zscore_normalize",
    "zscore_stats",
]
