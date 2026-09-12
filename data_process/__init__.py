"""Dataset construction and preprocessing; public exports load on demand."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AITemperatureDataset": (".dataset", "AITemperatureDataset"),
    "latest_split_manifest": (".dataset", "latest_split_manifest"),
    "SPLIT_EXPERIMENT_POLICIES": (".split_policy", "SPLIT_EXPERIMENT_POLICIES"),
    "DataRecord": (".builder", "DataRecord"),
    "DatabaseBuilder": (".builder", "DatabaseBuilder"),
    "build_case_dataset": (".case_pipeline", "build_case_dataset"),
    "discover_cases": (".case_pipeline", "discover_cases"),
    "split_manifest_file": (".dataset", "split_manifest_file"),
    "split_manifest_records": (".dataset", "split_manifest_records"),
    "split_case_records": (".dataset", "split_case_records"),
    "write_case_split_files": (".dataset", "write_case_split_files"),
    "ensure_material_csv": (".material_registry", "ensure_material_csv"),
    "load_material_to_idx": (".material_registry", "load_material_to_idx"),
    "material_csv_path": (".material_registry", "material_csv_path"),
    "register_material": (".material_registry", "register_material"),
    "register_materials": (".material_registry", "register_materials"),
    "DEFAULT_MATERIAL_SPLIT": (".material_collection", "DEFAULT_MATERIAL_SPLIT"),
    "MATERIAL_COLLECTION_FILE": (".material_collection", "MATERIAL_COLLECTION_FILE"),
    "MATERIAL_COLLECTION_KIND": (".material_collection", "MATERIAL_COLLECTION_KIND"),
    "attach_external_test_manifest": (
        ".material_collection",
        "attach_external_test_manifest",
    ),
    "build_material_collection": (".material_collection", "build_material_collection"),
    "build_mixed_collection_manifest": (
        ".material_collection",
        "build_mixed_collection_manifest",
    ),
    "discover_material_roots": (".material_collection", "discover_material_roots"),
    "latest_material_collection": (
        ".material_collection",
        "latest_material_collection",
    ),
    "load_material_collection": (".material_collection", "load_material_collection"),
    "resolve_collection_manifest": (
        ".material_collection",
        "resolve_collection_manifest",
    ),
    "sample_material_name": (".material_collection", "sample_material_name"),
    "validate_material_split": (".material_collection", "validate_material_split"),
    "FORCED_MODEL_ZSCORE_STEP": (".preprocess", "FORCED_MODEL_ZSCORE_STEP"),
    "SUPPORTED_PREPROCESS_STEPS": (".preprocess", "SUPPORTED_PREPROCESS_STEPS"),
    "load_cached_preprocessed_waveform": (
        ".preprocess_cache",
        "load_cached_preprocessed_waveform",
    ),
    "load_or_apply_preprocessed_waveform": (
        ".preprocess_cache",
        "load_or_apply_preprocessed_waveform",
    ),
    "preprocess_cache_dir": (".preprocess_cache", "preprocess_cache_dir"),
    "preprocess_pipeline_label": (".preprocess_cache", "preprocess_pipeline_label"),
    "save_preprocessed_waveform_cache": (
        ".preprocess_cache",
        "save_preprocessed_waveform_cache",
    ),
    "apply_optional_preprocessing": (".preprocess", "apply_optional_preprocessing"),
    "apply_preprocessing_pipeline": (".preprocess", "apply_preprocessing_pipeline"),
    "apply_zscore_with_stats": (".preprocess", "apply_zscore_with_stats"),
    "clip_outliers": (".preprocess", "clip_outliers"),
    "detrend_baseline": (".preprocess", "detrend_baseline"),
    "inverse_zscore": (".preprocess", "inverse_zscore"),
    "parse_preprocess_steps": (".preprocess", "parse_preprocess_steps"),
    "robust_normalize": (".preprocess", "robust_normalize"),
    "smooth_signal": (".preprocess", "smooth_signal"),
    "zscore_normalize": (".preprocess", "zscore_normalize"),
    "zscore_stats": (".preprocess", "zscore_stats"),
    "resolve_training_manifest_pair": (
        ".training_input",
        "resolve_training_manifest_pair",
    ),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
