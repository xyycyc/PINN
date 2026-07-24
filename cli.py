"""Command-line orchestration for database, training, prediction, and GUI tasks."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .artifact_paths import validate_artifact_basename
from .config import AIModelConfig
from .data_process import (
    DEFAULT_MATERIAL_SPLIT,
    SPLIT_EXPERIMENT_POLICIES,
    MATERIAL_COLLECTION_KIND,
    DatabaseBuilder,
    attach_external_test_manifest,
    build_case_dataset,
    build_material_collection,
    build_mixed_collection_manifest,
    discover_cases,
    discover_material_roots,
    latest_material_collection,
    latest_split_manifest,
    load_material_collection,
    parse_preprocess_steps,
    register_materials,
    resolve_collection_manifest,
    resolve_training_manifest_pair,
    split_manifest_file,
    sample_material_name,
    validate_material_split,
)
from .database.raw.waveforms_by_file.build_post0_database import (
    build_inference_database as build_post0_inference_database,
)
from .model import (
    OnlineUpdater,
    ReconstructionTrainer,
    MATERIAL_ROUTER_KIND,
    apply_training_runtime,
    build_rule_record,
    default_parameter_base_name,
    default_predict_output_name,
    ensure_rule_csv,
    predict_and_compare,
    predict_collection_with_checkpoint,
    predict_with_material_router,
    register_checkpoint_rule,
    resolve_checkpoint_from_rule,
    resolve_rule_triplet_for_checkpoint,
    resolve_training_runtime,
    rule_csv_path,
    sync_config_for_inference,
    validate_rule_triplet,
)


def _register_checkpoint_in_rule_table(
    config: AIModelConfig,
    *,
    dimension: str,
    mode: str,
    material: str,
    checkpoint_path: Path,
) -> bool:
    registry = ensure_rule_csv(config.data_root)
    record = build_rule_record(
        dimension=dimension,
        mode=mode,
        material=material,
        checkpoint_path=checkpoint_path,
        training_mode=config.training_mode,
        physics_residual_weight=float(config.physics_residual_weight),
        learnable_branch_weights=bool(config.learnable_branch_weights),
        fixed_weight_cnn=float(config.fixed_weight_cnn),
        fixed_weight_lstm=float(config.fixed_weight_lstm),
        fixed_weight_material=float(config.fixed_weight_material),
        fixed_weight_dimension=float(config.fixed_weight_dimension),
        fixed_weight_mode=float(config.fixed_weight_mode),
    )
    return register_checkpoint_rule(registry, record)


def _register_material_checkpoints_in_rule_table(
    config: AIModelConfig,
    *,
    router_path: Path,
    dimension: str,
    mode: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Register each complete sample-material checkpoint from a router."""

    router_payload = json.loads(router_path.read_text(encoding="utf-8"))
    registrations: list[dict[str, object]] = []
    for item in router_payload.get("checkpoints", []):
        material_checkpoint = router_path.parent / str(item["checkpoint"])
        rule_material = str(item["material_key"])
        registered = _register_checkpoint_in_rule_table(
            config,
            dimension=dimension,
            mode=mode,
            material=rule_material,
            checkpoint_path=material_checkpoint.resolve(),
        )
        registrations.append(
            {
                "material_key": rule_material,
                "material_name": str(item.get("material_name", rule_material)),
                "rule_material": rule_material,
                "checkpoint": str(material_checkpoint.resolve()),
                "rule_registered": registered,
            }
        )
    return router_payload, registrations


def _find_material_router_for_checkpoint(checkpoint_path: str | Path) -> Path | None:
    """Find a sibling sample-material router that declares the checkpoint."""

    checkpoint = Path(checkpoint_path).resolve()
    candidates: list[tuple[int, Path]] = []
    for router_path in checkpoint.parent.glob("*__material_router.json"):
        try:
            candidates.append((router_path.stat().st_mtime_ns, router_path))
        except OSError:
            continue
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    for _mtime_ns, router_path in candidates:
        try:
            payload = json.loads(router_path.read_text(encoding="utf-8"))
            if payload.get("router_kind") != MATERIAL_ROUTER_KIND:
                continue
            checkpoints = payload.get("checkpoints", [])
            if not isinstance(checkpoints, list):
                continue
            for item in checkpoints:
                if not isinstance(item, dict):
                    continue
                raw = Path(str(item.get("checkpoint", "")))
                declared = raw if raw.is_absolute() else router_path.parent / raw
                if declared.resolve() == checkpoint:
                    return router_path.resolve()
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return None


FIXED_WEIGHT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("--fixed-weight-cnn", "fixed_weight_cnn"),
    ("--fixed-weight-lstm", "fixed_weight_lstm"),
)

LEGACY_NOOP_WEIGHT_OPTIONS: tuple[str, ...] = (
    "fixed_weight_material",
    "fixed_weight_dimension",
    "fixed_weight_mode",
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return parsed


def _reject_legacy_noop_weights(args: argparse.Namespace) -> None:
    supplied = [name for name in LEGACY_NOOP_WEIGHT_OPTIONS if getattr(args, name, None) is not None]
    if supplied:
        flags = ", ".join(f"--{name.replace('_', '-')}" for name in supplied)
        raise ValueError(
            f"当前模型只有 CNN/LSTM 两个可加权分支；{flags} 是旧版无效参数，"
            "请移除，避免误以为它们会改变训练结果"
        )


def _add_fixed_weight_options(parser: argparse.ArgumentParser) -> None:
    """CNN/LSTM branch values plus explicit rejection of legacy no-op flags.

    The values stay fixed by default and become initial values when learnable
    branch weights are enabled. Material/dimension/mode legacy flags no longer
    map to forward branches and remain hidden only to produce a migration error.
    """
    parser.add_argument(
        "--fixed-weight-cnn",
        type=float,
        default=None,
        help="CNN 分支权重；固定模式为常量，可学习模式为初始值",
    )
    parser.add_argument(
        "--fixed-weight-lstm",
        type=float,
        default=None,
        help="LSTM 分支权重；固定模式为常量，可学习模式为初始值",
    )
    parser.add_argument(
        "--fixed-weight-material",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fixed-weight-dimension",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fixed-weight-mode",
        type=float,
        default=None,
        help=argparse.SUPPRESS,
    )


def _add_runtime_options(parser: argparse.ArgumentParser, include_epochs: bool = False) -> None:
    """给需要训练模型的子命令追加统一运行参数。"""
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--learnable-branch-weights",
        action="store_true",
        help="启用分支权重学习；默认关闭，此时使用各 --fixed-weight-* 配置值",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["normal", "residual_pinn"],
        default="normal",
        help="训练模式：normal=普通监督训练，residual_pinn=附加残差物理约束",
    )
    parser.add_argument(
        "--physics-residual-weight",
        type=float,
        default=0.1,
        help="residual_pinn 模式下物理残差项权重",
    )
    _add_fixed_weight_options(parser)
    if include_epochs:
        parser.add_argument("--epochs", type=_positive_int, default=20)


def _add_online_update_runtime_options(parser: argparse.ArgumentParser) -> None:
    """增量训练：运行参数默认从基础 checkpoint 恢复，显式传入时才覆盖。"""
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备；默认与基础 checkpoint 一致",
    )
    parser.add_argument(
        "--learnable-branch-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否启用可学习分支权重；默认与基础 checkpoint 一致",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["normal", "residual_pinn"],
        default=None,
        help="训练模式；默认与基础 checkpoint 一致",
    )
    parser.add_argument(
        "--physics-residual-weight",
        type=float,
        default=None,
        help="residual_pinn 物理残差权重；默认与基础 checkpoint 一致",
    )
    _add_fixed_weight_options(parser)
    parser.add_argument(
        "--epochs",
        type=_positive_int,
        default=None,
        help="增量训练轮数；默认与基础 checkpoint 的 online_epochs 一致",
    )


def _apply_standard_runtime_args(config: AIModelConfig, args: argparse.Namespace) -> None:
    if hasattr(args, "learnable_branch_weights") and args.learnable_branch_weights:
        config.learnable_branch_weights = True
    if hasattr(args, "training_mode"):
        config.training_mode = args.training_mode
    if hasattr(args, "physics_residual_weight"):
        config.physics_residual_weight = args.physics_residual_weight
    if hasattr(args, "device"):
        config.device = args.device
    if hasattr(args, "epochs") and args.epochs is not None:
        config.epochs = args.epochs
    if hasattr(args, "early_stopping_patience") and args.early_stopping_patience is not None:
        config.early_stopping_patience = int(args.early_stopping_patience)
    for cli_name, attr in FIXED_WEIGHT_OPTIONS:
        attr_name = cli_name.lstrip("-").replace("-", "_")
        if hasattr(args, attr_name):
            value = getattr(args, attr_name)
            if value is not None:
                setattr(config, attr, float(value))


def _apply_online_update_runtime_overrides(config: AIModelConfig, args: argparse.Namespace) -> None:
    if args.device is not None:
        config.device = args.device
    if args.training_mode is not None:
        config.training_mode = args.training_mode
    if args.physics_residual_weight is not None:
        config.physics_residual_weight = args.physics_residual_weight
    if args.learnable_branch_weights is not None:
        config.learnable_branch_weights = bool(args.learnable_branch_weights)
    if args.epochs is not None:
        config.online_epochs = int(args.epochs)
    for cli_name, attr in FIXED_WEIGHT_OPTIONS:
        attr_name = cli_name.lstrip("-").replace("-", "_")
        value = getattr(args, attr_name, None)
        if value is not None:
            setattr(config, attr, float(value))


def _add_preprocess_options(
    parser: argparse.ArgumentParser,
    *,
    inherit_from_checkpoint: bool = False,
) -> None:
    numeric_default: float | int | None = None if inherit_from_checkpoint else 1.0
    inheritance_help = (
        "；不传时继承 checkpoint" if inherit_from_checkpoint else ""
    )
    preprocess_help = "实验波形可选预处理，逗号分隔: clip,smooth,detrend,robust_norm"
    if inherit_from_checkpoint:
        preprocess_help += "；显式传空字符串可关闭 checkpoint 中继承的预处理"
    else:
        preprocess_help += "；空字符串表示不启用可选预处理步骤"
    parser.add_argument(
        "--preprocess",
        type=str,
        default=None,
        help=preprocess_help,
    )
    parser.add_argument(
        "--clip-quantile",
        type=float,
        default=numeric_default,
        help=f"clip 分位数阈值{inheritance_help}",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=None if inherit_from_checkpoint else 11,
        help=f"smooth 滑动窗口（自动转为奇数）{inheritance_help}",
    )


def _apply_preprocess_args(config: AIModelConfig, args: argparse.Namespace) -> None:
    if getattr(args, "preprocess", None) is not None:
        config.preprocess_steps = str(args.preprocess)
    if getattr(args, "clip_quantile", None) is not None:
        config.clip_quantile = float(args.clip_quantile)
    if getattr(args, "smooth_window", None) is not None:
        config.smooth_window = int(args.smooth_window)


def _add_predict_runtime_options(parser: argparse.ArgumentParser) -> None:
    """推理：运行参数默认从 checkpoint 恢复，仅显式传入时覆盖。"""
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="计算设备；默认与 checkpoint 中训练配置一致",
    )
    parser.add_argument(
        "--learnable-branch-weights",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="是否启用可学习分支权重；默认与 checkpoint 一致",
    )
    parser.add_argument(
        "--training-mode",
        type=str,
        choices=["normal", "residual_pinn"],
        default=None,
        help="训练模式（影响 compute_loss）；默认与 checkpoint 一致",
    )
    parser.add_argument(
        "--physics-residual-weight",
        type=float,
        default=None,
        help="物理残差权重；默认与 checkpoint 一致",
    )
    _add_fixed_weight_options(parser)


def _apply_predict_cli_overrides(config: AIModelConfig, args: argparse.Namespace) -> None:
    if getattr(args, "device", None) is not None:
        config.device = str(args.device)
    if getattr(args, "training_mode", None) is not None:
        config.training_mode = str(args.training_mode)
    if getattr(args, "physics_residual_weight", None) is not None:
        config.physics_residual_weight = float(args.physics_residual_weight)
    if getattr(args, "learnable_branch_weights", None) is not None:
        config.learnable_branch_weights = bool(args.learnable_branch_weights)
    _apply_preprocess_args(config, args)
    for cli_name, attr in FIXED_WEIGHT_OPTIONS:
        attr_name = cli_name.lstrip("-").replace("-", "_")
        value = getattr(args, attr_name, None)
        if value is not None:
            setattr(config, attr, float(value))


def _add_dataset_split_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在建库后自动划分 manifest（固定节点为 train/validation/test，默认启用）",
    )
    parser.add_argument(
        "--split-test-ratio",
        type=float,
        default=0.2,
        help="测试集比例 (0,1)，仅在启用 --split-dataset 时生效",
    )
    parser.add_argument(
        "--split-validation-ratio",
        type=float,
        default=0.1,
        help="固定节点 case 的验证集比例，默认 0.1；旧网格数据不使用该参数",
    )
    parser.add_argument(
        "--split-seed",
        type=int,
        default=42,
        help="数据集划分随机种子，仅在启用 --split-dataset 时生效",
    )
    parser.add_argument(
        "--split-experiment-policy",
        type=str,
        choices=list(SPLIT_EXPERIMENT_POLICIES),
        default="uniform",
        help="实验样本划分策略：uniform / all_experiment_train / all_experiment_test",
    )
    parser.add_argument(
        "--train-manifest-name",
        type=str,
        default="train_manifest.json",
        help="自动划分后训练清单文件名",
    )
    parser.add_argument(
        "--test-manifest-name",
        type=str,
        default="test_manifest.json",
        help="自动划分后测试清单文件名",
    )
    parser.add_argument(
        "--validation-manifest-name",
        type=str,
        default="validation_manifest.json",
        help="固定节点 case 自动划分后的验证清单文件名",
    )


def _add_io_root_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data-root",
        type=str,
        default="database",
        help="输入数据根目录（默认相对 ai_model 项目根为 database，含 manifest/waveforms/fields/raw 等）",
    )
    parser.add_argument(
        "--result-root",
        type=str,
        default="result",
        help="输出结果根目录（默认相对 ai_model 项目根为 result）",
    )


def _resolve_source_dir(config: AIModelConfig, path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    # 兼容历史写法：传入 database/... 时按 repo_root 解释。
    if path.parts and path.parts[0] == "database":
        return config.repo_root / path
    # 新默认：相对路径按 data_root 解释（例如 raw/10times）。
    return config.data_root / path


def _resolve_project_path(config: AIModelConfig, path_value: str | Path) -> Path:
    """Resolve CLI paths consistently against the ``ai_model`` package root.

    The Window runs subprocesses from the package parent, while its form defaults
    are written as ``database/...`` and ``result/...``.  Accept both those paths
    and the older documentation form ``ai_model/database/...`` without depending
    on the caller's current working directory.
    """

    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    if path.parts and path.parts[0].casefold() == config.repo_root.name.casefold():
        return (config.repo_root.parent / path).resolve()
    return (config.repo_root / path).resolve()


def _sanitize_material_label(material: str) -> str:
    label = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(material or "").strip())
    label = label.strip("_")
    return label or "unknown_material"


def _parse_material_split_values(values: list[str] | None) -> dict[str, tuple[float, float, float]]:
    result: dict[str, tuple[float, float, float]] = {}
    for raw in values or []:
        material_key, separator, ratios_text = str(raw).partition("=")
        material_key = material_key.strip()
        if not separator or not material_key:
            raise ValueError(
                "--material-split 格式应为 <材料文件夹>=<train>,<validation>,<test>"
            )
        if material_key in result:
            raise ValueError(f"--material-split 重复指定材料: {material_key}")
        parts = [part.strip() for part in ratios_text.split(",")]
        if len(parts) != 3:
            raise ValueError(
                f"材料 {material_key} 必须提供 train,validation,test 三个比例"
            )
        try:
            result[material_key] = validate_material_split(*(float(part) for part in parts))
        except ValueError as exc:
            raise ValueError(f"材料 {material_key} 的划分比例无效: {exc}") from exc
    return result


def _validate_material_collection(
    config: AIModelConfig,
    collection_path: str | Path,
) -> dict[str, object]:
    """Validate a collection while counting sibling folders as materials."""
    collection_path = Path(collection_path).resolve()
    collection = load_material_collection(collection_path)
    validator = DatabaseBuilder(config)
    material_reports: dict[str, dict[str, object]] = {}
    for raw_entry in collection["materials"]:
        entry = dict(raw_entry)
        material_key = str(entry["material_key"])
        material_reports[material_key] = validator.validate_requirement_33(
            resolve_collection_manifest(collection_path, entry, "combined")
        )
    total_records = sum(int(item["total_records"]) for item in material_reports.values())
    simulation_records = sum(
        int(item["simulation_records"]) for item in material_reports.values()
    )
    experiment_records = sum(
        int(item["experiment_records"]) for item in material_reports.values()
    )
    lower_bounds = [
        float(item["temperature_range_k"][0])
        for item in material_reports.values()
        if item["temperature_range_k"][0] is not None
    ]
    upper_bounds = [
        float(item["temperature_range_k"][1])
        for item in material_reports.values()
        if item["temperature_range_k"][1] is not None
    ]
    materials = sorted(material_reports)
    temperature_range = [
        min(lower_bounds) if lower_bounds else None,
        max(upper_bounds) if upper_bounds else None,
    ]
    report: dict[str, object] = {
        "material_collection": str(collection_path),
        "dataset_label": str(collection.get("dataset_label", "")),
        "total_records": total_records,
        "simulation_records": simulation_records,
        "experiment_records": experiment_records,
        "materials": materials,
        "material_count": len(materials),
        "material_source": "material_collection.materials[].material_key",
        "temperature_range_k": temperature_range,
        "material_reports": material_reports,
        "meets_3_3_min_simulation": (
            simulation_records >= config.min_simulation_samples
        ),
        "meets_3_3_min_experiment": (
            experiment_records >= config.min_experiment_samples
        ),
        "meets_3_3_material_count": len(materials) >= 3,
        "meets_3_3_temperature_span": bool(
            lower_bounds
            and upper_bounds
            and min(lower_bounds) <= config.min_temperature_k
            and max(upper_bounds) >= config.max_temperature_k
        ),
    }
    report["meets_3_3"] = all(
        bool(report[key])
        for key in (
            "meets_3_3_min_simulation",
            "meets_3_3_min_experiment",
            "meets_3_3_material_count",
            "meets_3_3_temperature_span",
        )
    )
    report_dir = config.train_report_root / "validation"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "requirement_3_3_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report


def _add_rule_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rule-dimension", type=str, choices=["one", "two"], default="")
    parser.add_argument("--rule-mode", type=str, choices=["steady", "transient"], default="")
    parser.add_argument("--rule-material", type=str, default="")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AI 温度场反演子项目")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_db = subparsers.add_parser("build-db", help="构建仿真/实验数据库并合并")
    _add_io_root_options(build_db)
    build_db.add_argument("--sim-per-material", type=int, default=400)
    build_db.add_argument(
        "--experiment-dir",
        type=str,
        default="raw/10times",
        help=(
            "实验数据目录；相对路径默认按 data_root 解析（如 raw/10times）；"
            "按配置结构和 CSV 字段自动识别旧格式或直接导出 case"
        ),
    )
    build_db.add_argument(
        "--experiment-material",
        type=str,
        default="metal_matrix",
        help="数据集命名标签；样本材料路由字段始终取材料文件夹名",
    )
    build_db.add_argument(
        "--multi-material-input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="将 --experiment-dir 解释为多材料上层目录，并扫描其直接子文件夹",
    )
    build_db.add_argument(
        "--material-split",
        action="append",
        default=[],
        metavar="MATERIAL=TRAIN,VALIDATION,TEST",
        help="多材料模式下单独指定某材料的划分比例；可重复传入",
    )
    build_db.add_argument(
        "--external-test-dir",
        type=str,
        default="",
        help="仅测试的 post0 实验波形 CSV 目录；只在多材料建库中使用",
    )
    build_db.add_argument(
        "--external-test-material",
        type=str,
        default="",
        help="外部波形测试集要挂接到的材料路由字段，例如 wumu",
    )
    build_db.add_argument(
        "--external-test-dataset-name",
        type=str,
        default="wumu_exp",
        help="外部波形测试集名称及输出目录名",
    )
    build_db.add_argument(
        "--external-test-signal-column",
        type=str,
        default="amplitude_filtered_residual",
    )
    build_db.add_argument(
        "--external-test-time-column",
        type=str,
        default="time_s",
    )
    build_db.add_argument(
        "--external-test-time-min-s",
        type=float,
        default=0.0,
    )
    build_db.add_argument(
        "--external-test-post0-length",
        type=int,
        default=None,
        help="可选；不传时使用所有外部测试波形 post0 段的最短长度",
    )
    build_db.add_argument(
        "--external-test-limit",
        type=int,
        default=-1,
        help="-1 表示导入全部外部测试 CSV",
    )
    build_db.add_argument("--experiment-limit", type=int, default=2000)
    build_db.add_argument(
        "--waveform-crop-length",
        type=int,
        default=1097,
        help="case 波形保留的原始连续前缀点数；默认 1097，不重采样",
    )
    build_db.add_argument(
        "--skip-simulation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否跳过内置仿真数据生成（默认跳过，可用 --no-skip-simulation 开启）",
    )
    build_db.add_argument(
        "--external-sim-dir",
        type=str,
        default="",
        help="外部仿真 CSV 目录（默认关闭；相对路径默认按 data_root 解析）",
    )
    build_db.add_argument(
        "--external-sim-material",
        type=str,
        default="",
        help="外部仿真材料标签；不传则复用 --experiment-material",
    )
    build_db.add_argument(
        "--external-sim-limit",
        type=int,
        default=-1,
        help="外部仿真样本上限（<0 表示不限制，0 表示不导入）",
    )
    _add_dataset_split_options(build_db)

    train = subparsers.add_parser("train", help="训练 AI 温度场重构模型")
    _add_io_root_options(train)
    train.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="训练数据集文件夹，或兼容传入 train/material_collection manifest",
    )
    train.add_argument(
        "--validation-manifest",
        type=str,
        default=None,
        help="可选的验证 manifest；选择数据集文件夹时会自动解析",
    )
    train.add_argument(
        "--checkpoint-name",
        type=str,
        default="ai_model.pt",
        help="checkpoint 文件名（保存在 result/train/checkpoint/<train-name>/ 下）",
    )
    train.add_argument(
        "--train-name",
        type=str,
        default="",
        help="训练任务名（用于 result/train/checkpoint|report 的子目录名）",
    )
    train.add_argument(
        "--separate-materials",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="对多材料集合中的每个样本级材料训练一个完整温度场 checkpoint",
    )
    train.add_argument(
        "--early-stopping-patience",
        type=_positive_int,
        default=10,
        help="validation loss 连续多少代未改善后早停，默认 10",
    )
    _add_rule_options(train)
    _add_runtime_options(train, include_epochs=True)
    _add_preprocess_options(train)

    validate = subparsers.add_parser("validate", help="验证是否满足 3.3 数据要求")
    _add_io_root_options(validate)
    validate.add_argument("--manifest", type=str, default=None)

    update = subparsers.add_parser(
        "online-update",
        help="增量训练：在基础 checkpoint 目录下写入 <时间戳>.pt，报告写入 report/<任务名>/Incremental/<时间戳>/",
    )
    _add_io_root_options(update)
    update.add_argument("--manifest", type=str, required=True)
    update.add_argument("--checkpoint", type=str, required=False, default="")
    update.add_argument(
        "--output-name",
        type=str,
        default="",
        help="输出权重文件名；默认 <训练时间>.pt，保存在基础 checkpoint 同级目录",
    )
    _add_rule_options(update)
    _add_online_update_runtime_options(update)
    _add_preprocess_options(update, inherit_from_checkpoint=True)

    predict = subparsers.add_parser(
        "predict",
        help="基于 checkpoint 对指定 manifest 做推理，输出预测与真实值对比产物",
    )
    _add_io_root_options(predict)
    predict.add_argument("--manifest", type=str, required=True, help="待预测样本的 manifest.json")
    predict.add_argument("--checkpoint", type=str, required=False, default="", help="训练得到的 *.pt 文件")
    predict.add_argument(
        "--material-router",
        type=str,
        default="",
        help="按样本级材料分开训练生成的 *__material_router.json；依据 records[].material_key 选择完整 checkpoint",
    )
    predict.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="产物输出目录；不传则写到 result/predict/<kind>/<predict-name>/",
    )
    predict.add_argument(
        "--predict-name",
        type=str,
        default="",
        help="推理任务名；不传则为 <检查点所在目录名>_<预测时间>",
    )
    predict.add_argument(
        "--predict-kind",
        type=str,
        choices=["inference", "batch"],
        default="inference",
        help=argparse.SUPPRESS,
    )
    predict.add_argument(
        "--plots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="生成散点图与温度场三联图（默认关闭）",
    )
    predict.add_argument(
        "--num-field-samples",
        type=int,
        default=6,
        help="启用 --plots 时，抽样绘制温度场三联图的样本数",
    )
    predict.add_argument(
        "--prediction-dimension",
        type=str,
        choices=["auto", "one", "two"],
        default="auto",
        help=(
            "预测结果展示维度：二维固定节点模型可选 one/two；"
            "一维模型只能选 one；auto 按模型类型选择"
        ),
    )
    predict.add_argument(
        "--benchmark",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="启用纯前向测速（默认关闭）",
    )
    predict.add_argument(
        "--benchmark-warmup-samples",
        type=int,
        default=64,
        help="测速预热样本数；大于清单样本数时循环使用清单样本",
    )
    predict.add_argument(
        "--benchmark-runs",
        type=int,
        default=3,
        help="测速计时的前向 batch 轮数",
    )
    _add_rule_options(predict)
    _add_predict_runtime_options(predict)
    _add_preprocess_options(predict, inherit_from_checkpoint=True)

    demo = subparsers.add_parser("demo", help="一键构建数据库、训练并验证")
    _add_io_root_options(demo)
    demo.add_argument("--sim-per-material", type=int, default=400)
    demo.add_argument("--experiment-limit", type=int, default=2000)
    demo.add_argument(
        "--experiment-dir",
        type=str,
        default="raw/calibration_sweep",
        help="单材料目录，或开启多材料输入时的上层目录",
    )
    demo.add_argument(
        "--experiment-material",
        type=str,
        default="metal_matrix",
        help="一键演示的数据集命名标签；样本材料路由字段取文件夹名",
    )
    demo.add_argument(
        "--multi-material-input",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="扫描 --experiment-dir 的直接子目录作为样本级材料",
    )
    demo.add_argument(
        "--material-split",
        action="append",
        default=[],
        metavar="MATERIAL=TRAIN,VALIDATION,TEST",
        help="多材料模式下单独指定某材料的划分比例；可重复传入",
    )
    demo.add_argument(
        "--waveform-crop-length",
        type=int,
        default=1097,
        help="case 波形保留的原始连续前缀点数；默认 1097，不重采样",
    )
    demo.add_argument("--train-name", type=str, default="")
    demo.add_argument(
        "--separate-materials",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="为多材料集合中的每种样本级材料训练完整 checkpoint",
    )
    _add_dataset_split_options(demo)
    _add_rule_options(demo)
    _add_runtime_options(demo, include_epochs=True)
    _add_preprocess_options(demo)
    return parser


def main() -> None:
    """Parse one command and dispatch it through the versioned project pipelines."""
    parser = _build_parser()
    args = parser.parse_args()
    _reject_legacy_noop_weights(args)
    config = AIModelConfig()
    if hasattr(args, "data_root"):
        config.data_root = _resolve_project_path(config, args.data_root)
    if hasattr(args, "result_root"):
        config.result_root = _resolve_project_path(config, args.result_root)
    config.ensure_dirs()
    if args.command not in {"online-update", "predict"}:
        _apply_standard_runtime_args(config, args)
    builder = DatabaseBuilder(config)

    raw_dimension = str(getattr(args, "rule_dimension", "") or "").strip()
    raw_mode = str(getattr(args, "rule_mode", "") or "").strip()
    raw_material = str(getattr(args, "rule_material", "") or "").strip()
    use_rule = any([raw_dimension, raw_mode, raw_material])
    if use_rule and not all([raw_dimension, raw_mode, raw_material]):
        raise ValueError("使用规则匹配时，--rule-dimension/--rule-mode/--rule-material 必须同时提供")
    resolved_rule: tuple[str, str, str] | None = None
    if use_rule:
        resolved_rule = validate_rule_triplet(raw_dimension, raw_mode, raw_material)

    if args.command == "build-db":
        material_bucket = _sanitize_material_label(str(args.experiment_material or "unknown_material"))
        build_data_root = config.data_root / "data_process" / material_bucket
        build_builder = DatabaseBuilder(config)
        build_builder.database_dir = build_data_root
        experiment_source = _resolve_source_dir(config, args.experiment_dir)
        if str(args.external_test_dir or "").strip() and not bool(args.multi_material_input):
            raise ValueError("独立波形测试集只能与 --multi-material-input 一起使用")
        if bool(args.multi_material_input):
            if not bool(args.skip_simulation) or str(args.external_sim_dir or "").strip():
                raise ValueError("多材料固定节点数据不能与旧规则网格仿真 CSV 混合")
            if not bool(args.split_dataset):
                raise ValueError("多材料建库必须启用独立的 train/validation/test 划分")
            material_roots = discover_material_roots(experiment_source)
            material_splits = _parse_material_split_values(args.material_split)
            external_test_dir = str(args.external_test_dir or "").strip()
            if external_test_dir:
                external_test_material = str(args.external_test_material or "").strip()
                if not external_test_material:
                    raise ValueError("导入独立波形测试集时必须指定 --external-test-material")
                selected_split = material_splits.get(
                    external_test_material,
                    DEFAULT_MATERIAL_SPLIT,
                )
                if abs(float(selected_split[2])) > 1e-9:
                    raise ValueError(
                        f"材料 {external_test_material!r} 使用独立测试集时，"
                        "其 --material-split 测试比例必须为 0"
                    )
            build_data_root = (
                config.data_root
                / "data_process"
                / f"{material_bucket}_multi_material_temperature_field"
            )
            collection_path = build_material_collection(
                experiment_source,
                build_data_root,
                dataset_label=str(args.experiment_material or ""),
                material_splits=material_splits,
                target_points=10000,
                seed=int(args.split_seed),
                waveform_crop_length=int(args.waveform_crop_length),
                limit_per_material=int(args.experiment_limit),
                train_manifest_name=str(args.train_manifest_name),
                validation_manifest_name=str(args.validation_manifest_name),
                test_manifest_name=str(args.test_manifest_name),
            )
            external_test_result: dict[str, object] | None = None
            if external_test_dir:
                collection_payload = load_material_collection(collection_path)
                matching_entries = [
                    entry
                    for entry in collection_payload["materials"]
                    if str(entry.get("material_key", "")).strip() == external_test_material
                ]
                if not matching_entries:
                    available = [
                        str(entry.get("material_key", "")).strip()
                        for entry in collection_payload["materials"]
                    ]
                    raise ValueError(
                        f"外部测试材料 {external_test_material!r} 不在多材料目录中；"
                        f"可用材料为 {available}"
                    )
                reference_manifest = resolve_collection_manifest(
                    collection_path,
                    matching_entries[0],
                    "train",
                )
                external_source = _resolve_source_dir(config, external_test_dir)
                external_dataset_name = (
                    str(args.external_test_dataset_name or "").strip()
                    or external_source.name
                    or f"{external_test_material}_external_test"
                )
                external_output = (
                    build_data_root
                    / "external_tests"
                    / _sanitize_material_label(external_dataset_name)
                )
                external_limit = (
                    None
                    if int(args.external_test_limit) < 0
                    else int(args.external_test_limit)
                )
                external_summary = build_post0_inference_database(
                    source_dir=external_source,
                    output_dir=external_output,
                    reference_manifest=reference_manifest,
                    dataset_name=external_dataset_name,
                    material_key=external_test_material,
                    signal_column=str(args.external_test_signal_column),
                    time_column=str(args.external_test_time_column),
                    time_min_s=float(args.external_test_time_min_s),
                    post0_length=args.external_test_post0_length,
                    limit=external_limit,
                    test_manifest_name=str(args.test_manifest_name),
                )
                external_manifest = Path(
                    str(external_summary["manifests"]["test"])
                )
                attach_external_test_manifest(
                    collection_path,
                    material_key=external_test_material,
                    manifest_path=external_manifest,
                    source_root=external_source,
                )
                external_test_result = {
                    "material_key": external_test_material,
                    "source_dir": str(external_source),
                    "manifest": str(external_manifest),
                    "records": int(external_summary["records"]),
                    "inference_only": True,
                }
            register_materials(
                config.data_root,
                material_roots.keys(),
                source="sample_material_folder",
            )
            collection_payload = json.loads(collection_path.read_text(encoding="utf-8"))
            print(
                json.dumps(
                    {
                        "data_root": str(build_data_root),
                        "material_collection": str(collection_path),
                        "dataset_label": collection_payload.get("dataset_label", ""),
                        "materials": collection_payload.get("materials", []),
                        "external_test": external_test_result,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        case_dirs = discover_cases(experiment_source) if experiment_source.exists() else []
        if case_dirs and int(args.experiment_limit) != 0:
            if not bool(args.skip_simulation) or str(args.external_sim_dir or "").strip():
                raise ValueError("固定节点 case 数据不能与旧规则网格仿真 CSV 静默合并；请保持跳过内置/外部仿真")
            if not bool(args.split_dataset):
                raise ValueError(
                    "固定节点 case 训练需要由训练集拟合温度标准化参数，"
                    "因此建库时必须启用 --split-dataset"
                )
            build_data_root = config.data_root / "data_process" / f"{material_bucket}_case_temperature_field"
            build_data_root.mkdir(parents=True, exist_ok=True)
            combined = build_case_dataset(
                experiment_source,
                build_data_root,
                target_points=10000,
                seed=int(args.split_seed),
                waveform_crop_length=int(args.waveform_crop_length),
                limit=int(args.experiment_limit),
                test_ratio=float(args.split_test_ratio),
                validation_ratio=float(args.split_validation_ratio),
                dataset_label=str(args.experiment_material or ""),
                sample_material_key=experiment_source.name,
                sample_material_name=sample_material_name(experiment_source.name),
                train_manifest_name=str(args.train_manifest_name),
                validation_manifest_name=str(args.validation_manifest_name),
                test_manifest_name=str(args.test_manifest_name),
            )
            case_manifest = json.loads(combined.read_text(encoding="utf-8"))
            register_materials(
                config.data_root,
                [experiment_source.name],
                source="sample_material_folder",
            )
            split_config_path = build_data_root / "split_config.json"
            split_payload = json.loads(split_config_path.read_text(encoding="utf-8"))
            print(json.dumps({
                "data_root": str(build_data_root),
                "combined_manifest": str(combined),
                "split": {
                    "train_manifest": split_payload["manifests"]["train"],
                    "validation_manifest": split_payload["manifests"]["validation"],
                    "test_manifest": split_payload["manifests"]["test"],
                    "split_stats": {k: split_payload[k] for k in ("seed", "ratios", "case_counts", "leakage")},
                },
                "split_config": str(split_config_path),
                "dataset_schema_version": 2,
                "dataset_label": case_manifest.get("dataset_label", ""),
                "sample_material_catalog": case_manifest.get("sample_material_catalog", []),
                "constituent_material_catalog": case_manifest.get(
                    "constituent_material_catalog", []
                ),
            }, ensure_ascii=False, indent=2))
            return
        build_data_root.mkdir(parents=True, exist_ok=True)
        manifest_list: list[Path] = []
        if not bool(args.skip_simulation):
            sim_manifest = build_builder.build_simulation_database(samples_per_material=args.sim_per_material)
            manifest_list.append(sim_manifest)
        if int(args.experiment_limit) != 0:
            exp_manifest = build_builder.import_experimental_csvs(
                source_dir=experiment_source,
                material_key=args.experiment_material,
                limit=args.experiment_limit,
            )
            manifest_list.append(exp_manifest)
        external_sim_dir = str(args.external_sim_dir or "").strip()
        if external_sim_dir and int(args.external_sim_limit) != 0:
            external_material = str(args.external_sim_material or "").strip() or str(args.experiment_material or "").strip()
            external_limit = args.external_sim_limit if int(args.external_sim_limit) >= 0 else None
            external_manifest = build_builder.import_external_simulation_csvs(
                source_dir=_resolve_source_dir(config, external_sim_dir),
                material_key=external_material,
                limit=external_limit,
            )
            manifest_list.append(external_manifest)
        if not manifest_list:
            raise ValueError("没有可合并的数据源：请开启至少一个数据来源（仿真/实验/外部仿真）")
        combined = build_builder.merge_manifests(manifest_list)
        split_result: dict[str, object] | None = None
        if bool(args.split_dataset):
            train_manifest, test_manifest, split_stats = split_manifest_file(
                manifest_path=combined,
                test_ratio=float(args.split_test_ratio),
                seed=int(args.split_seed),
                experiment_policy=str(args.split_experiment_policy),
                train_output_name=str(args.train_manifest_name),
                test_output_name=str(args.test_manifest_name),
            )
            split_result = {
                "train_manifest": str(train_manifest),
                "test_manifest": str(test_manifest),
                "split_stats": split_stats,
            }
        split_config_path = build_data_root / "split_config.json"
        split_config_path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "experiment_material": str(args.experiment_material or ""),
                    "data_root": str(build_data_root),
                    "source_options": {
                        "skip_simulation": bool(args.skip_simulation),
                        "sim_per_material": int(args.sim_per_material),
                        "experiment_dir": str(_resolve_source_dir(config, args.experiment_dir)),
                        "experiment_limit": int(args.experiment_limit),
                        "external_sim_dir": str(_resolve_source_dir(config, external_sim_dir)) if external_sim_dir else "",
                        "external_sim_limit": int(args.external_sim_limit),
                    },
                    "split_params": {
                        "split_dataset": bool(args.split_dataset),
                        "split_test_ratio": float(args.split_test_ratio),
                        "split_seed": int(args.split_seed),
                        "split_experiment_policy": str(args.split_experiment_policy),
                        "train_manifest_name": str(args.train_manifest_name),
                        "test_manifest_name": str(args.test_manifest_name),
                    },
                    "split_stats": split_result.get("split_stats") if split_result else None,
                    "manifests": {
                        "combined": str(combined),
                        "train": str(split_result["train_manifest"]) if split_result else None,
                        "test": str(split_result["test_manifest"]) if split_result else None,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "data_root": str(build_data_root),
                    "combined_manifest": str(combined),
                    "split": split_result,
                    "split_config": str(split_config_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "train":
        _apply_preprocess_args(config, args)
        train_name = str(args.train_name or "").strip()
        checkpoint_name = str(args.checkpoint_name or "").strip()
        if resolved_rule is not None:
            dim, md, mat = resolved_rule
            if not train_name or not checkpoint_name:
                base_name = default_parameter_base_name(
                    dimension=dim,
                    mode=md,
                    material=mat,
                    now=datetime.now(),
                )
                if not train_name:
                    train_name = base_name
                if not checkpoint_name:
                    checkpoint_name = f"{base_name}.pt"
        default_train_manifest = config.database_dir / "train_manifest.json"
        latest_train_manifest = latest_split_manifest(config.data_root, "train")
        latest_collection = latest_material_collection(config.data_root)
        if bool(args.separate_materials):
            latest_training_input = latest_collection
        else:
            training_candidates = [
                path
                for path in (latest_train_manifest, latest_collection)
                if path is not None
            ]
            latest_training_input = (
                max(training_candidates, key=lambda path: path.stat().st_mtime_ns)
                if training_candidates
                else None
            )
        default_manifest = latest_training_input or (
            (default_train_manifest if default_train_manifest.exists() else None)
            or latest_split_manifest(config.data_root, "combined")
            or config.database_dir / "combined_manifest.json"
        )
        training_input = (
            _resolve_project_path(config, args.manifest)
            if args.manifest
            else default_manifest
        )
        manifest_path, validation_manifest_path = resolve_training_manifest_pair(training_input)
        if args.validation_manifest:
            validation_manifest_path = _resolve_project_path(config, args.validation_manifest)
        trainer = ReconstructionTrainer(config)
        manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        is_material_collection = (
            manifest_payload.get("collection_kind") == MATERIAL_COLLECTION_KIND
        )
        if bool(args.separate_materials):
            if not is_material_collection:
                raise ValueError(
                    "--separate-materials 需要 material_collection.json；"
                    "请在建库时启用多材料输入并选择各材料文件夹的上层目录"
                )
            router_path = trainer.train_material_checkpoints(
                manifest_path,
                checkpoint_name=checkpoint_name or "ai_model.pt",
                train_name=train_name or None,
            )
            router_payload = json.loads(router_path.read_text(encoding="utf-8"))
            registrations: list[dict[str, object]] = []
            if resolved_rule is not None:
                dim, md, _dataset_rule_name = resolved_rule
                router_payload, registrations = _register_material_checkpoints_in_rule_table(
                    config,
                    router_path=router_path,
                    dimension=dim,
                    mode=md,
                )
            print(
                json.dumps(
                    {
                        "training_strategy": "separate_material_checkpoints",
                        "material_router": str(router_path.resolve()),
                        "checkpoints": registrations or router_payload.get("checkpoints", []),
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        if is_material_collection:
            collection_path = manifest_path
            manifest_path = build_mixed_collection_manifest(collection_path)
            mixed_train_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
            validation_manifest_path = build_mixed_collection_manifest(
                collection_path,
                output_name="mixed_validation_manifest.json",
                split_kind="validation",
                normalization=mixed_train_payload["normalization"],
            )
        checkpoint = trainer.train(
            manifest_path,
            validation_manifest_path=validation_manifest_path,
            checkpoint_name=checkpoint_name or "ai_model.pt",
            train_name=train_name or None,
        )
        if resolved_rule is not None:
            dim, md, mat = resolved_rule
            _register_checkpoint_in_rule_table(
                config,
                dimension=dim,
                mode=md,
                material=mat,
                checkpoint_path=Path(checkpoint).resolve(),
            )
        print(checkpoint)
        return

    if args.command == "validate":
        automatic_candidates = [
            path
            for path in (
                latest_split_manifest(config.data_root, "combined"),
                latest_material_collection(config.data_root),
            )
            if path is not None
        ]
        latest_validation_input = (
            max(automatic_candidates, key=lambda path: path.stat().st_mtime_ns)
            if automatic_candidates
            else None
        )
        manifest_path = (
            _resolve_project_path(config, args.manifest)
            if args.manifest
            else (
                latest_validation_input
                or config.database_dir / "combined_manifest.json"
            )
        )
        manifest_payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if manifest_payload.get("collection_kind") == MATERIAL_COLLECTION_KIND:
            report = _validate_material_collection(config, manifest_path)
        else:
            report = builder.validate_requirement_33(manifest_path)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.command == "online-update":
        checkpoint_input = str(args.checkpoint or "").strip()
        if not checkpoint_input and resolved_rule is not None:
            dim, md, mat = resolved_rule
            checkpoint_input = str(
                resolve_checkpoint_from_rule(
                    rule_csv_path(config.data_root),
                    dimension=dim,
                    mode=md,
                    material=mat,
                )
            )
        if not checkpoint_input:
            raise ValueError("online-update 需要 --checkpoint，或提供完整 rule 三元组用于自动匹配")

        checkpoint_path = _resolve_project_path(config, checkpoint_input)
        sync_config_for_inference(
            config,
            checkpoint_path,
            rule_csv_path=rule_csv_path(config.data_root),
        )
        _apply_online_update_runtime_overrides(config, args)
        _apply_preprocess_args(config, args)

        output_name = str(args.output_name or "").strip() or None
        updater = OnlineUpdater(config)
        output_path = updater.update(
            _resolve_project_path(config, args.manifest),
            checkpoint_path,
            output_name=output_name,
        )
        output_resolved = Path(output_path).resolve()
        triplet = resolved_rule
        if triplet is None:
            triplet = resolve_rule_triplet_for_checkpoint(
                rule_csv_path(config.data_root),
                checkpoint_path,
            )
        if triplet is None:
            raise ValueError(
                "增量训练完成，但无法解析规则三元组以登记参数；"
                "请提供 --rule-dimension/--rule-mode/--rule-material，"
                "或确保基础 checkpoint 已在 trained_rules.csv 中登记。"
            )
        dim, md, mat = triplet
        registered = _register_checkpoint_in_rule_table(
            config,
            dimension=dim,
            mode=md,
            material=mat,
            checkpoint_path=output_resolved,
        )
        print(
            json.dumps(
                {
                    "checkpoint": str(output_resolved),
                    "rule_registered": registered,
                    "dimension": dim,
                    "mode": md,
                    "material": mat,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "predict":
        router_input = str(args.material_router or "").strip()
        checkpoint_input = str(args.checkpoint or "").strip()
        if not router_input and not checkpoint_input and resolved_rule is not None:
            dim, md, mat = resolved_rule
            checkpoint_input = str(
                resolve_checkpoint_from_rule(
                    rule_csv_path(config.data_root),
                    dimension=dim,
                    mode=md,
                    material=mat,
                )
            )
        if not router_input and not checkpoint_input:
            raise ValueError("predict 需要 --checkpoint、--material-router，或完整 rule 三元组")
        router_path: Path | None = (
            _resolve_project_path(config, router_input) if router_input else None
        )
        checkpoint_path: Path
        if router_path is not None:
            router_payload = json.loads(router_path.read_text(encoding="utf-8"))
            checkpoints = router_payload.get("checkpoints", [])
            if not checkpoints:
                raise ValueError("材料路由文件没有 checkpoints")
            first_checkpoint = Path(str(checkpoints[0].get("checkpoint", "")))
            checkpoint_path = (
                first_checkpoint
                if first_checkpoint.is_absolute()
                else router_path.parent / first_checkpoint
            )
        else:
            checkpoint_path = _resolve_project_path(config, checkpoint_input)
            discovered_router = _find_material_router_for_checkpoint(checkpoint_path)
            if discovered_router is not None:
                router_path = discovered_router
                router_payload = json.loads(router_path.read_text(encoding="utf-8"))
                checkpoints = router_payload.get("checkpoints", [])
                if not checkpoints:
                    raise ValueError("材料路由文件没有 checkpoints")
                first_checkpoint = Path(str(checkpoints[0].get("checkpoint", "")))
                checkpoint_path = (
                    first_checkpoint
                    if first_checkpoint.is_absolute()
                    else router_path.parent / first_checkpoint
                )
        sync_config_for_inference(
            config,
            checkpoint_path,
            rule_csv_path=rule_csv_path(config.data_root),
        )
        _apply_predict_cli_overrides(config, args)
        if args.output_dir:
            output_dir = _resolve_project_path(config, args.output_dir)
        else:
            raw_predict_name = str(args.predict_name or "").strip()
            predict_name = (
                validate_artifact_basename(raw_predict_name, label="推理任务名")
                if raw_predict_name
                else default_predict_output_name(router_path or checkpoint_path)
            )
            root = config.predict_inference_root if args.predict_kind == "inference" else config.predict_batch_root
            output_dir = root / predict_name
        if router_path is not None:
            metrics = predict_with_material_router(
                cfg=config,
                router_path=router_path,
                manifest_path=_resolve_project_path(config, args.manifest),
                output_dir=output_dir,
                sync_config_from_checkpoint=False,
                enable_plots=bool(args.plots),
                num_field_samples=args.num_field_samples,
                prediction_dimension=str(args.prediction_dimension),
                enable_benchmark=bool(args.benchmark),
                benchmark_warmup_samples=args.benchmark_warmup_samples,
                benchmark_runs=args.benchmark_runs,
            )
        else:
            prediction_input = _resolve_project_path(config, args.manifest)
            prediction_payload = json.loads(
                prediction_input.read_text(encoding="utf-8")
            )
            common_predict_options = {
                "cfg": config,
                "checkpoint_path": checkpoint_path,
                "output_dir": output_dir,
                "sync_config_from_checkpoint": False,
                "enable_plots": bool(args.plots),
                "num_field_samples": args.num_field_samples,
                "prediction_dimension": str(args.prediction_dimension),
                "enable_benchmark": bool(args.benchmark),
                "benchmark_warmup_samples": args.benchmark_warmup_samples,
                "benchmark_runs": args.benchmark_runs,
            }
            if prediction_payload.get("collection_kind") == MATERIAL_COLLECTION_KIND:
                metrics = predict_collection_with_checkpoint(
                    collection_path=prediction_input,
                    **common_predict_options,
                )
            else:
                metrics = predict_and_compare(
                    manifest_path=prediction_input,
                    **common_predict_options,
                )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    if args.command == "demo":
        demo_train_name = str(args.train_name or "").strip()
        if not demo_train_name and resolved_rule is not None:
            dim, md, mat = resolved_rule
            demo_train_name = default_parameter_base_name(
                dimension=dim,
                mode=md,
                material=mat,
                now=datetime.now(),
            )
        _apply_preprocess_args(config, args)
        experiment_source = _resolve_source_dir(config, args.experiment_dir)
        material_bucket = _sanitize_material_label(
            str(args.experiment_material or experiment_source.name or "unknown_material")
        )
        if bool(args.multi_material_input):
            if not bool(args.split_dataset):
                raise ValueError("多材料一键训练必须启用独立的 train/validation/test 划分")
            material_roots = discover_material_roots(experiment_source)
            collection_root = (
                config.data_root
                / "data_process"
                / f"{material_bucket}_multi_material_temperature_field"
            )
            collection_path = build_material_collection(
                experiment_source,
                collection_root,
                dataset_label=str(args.experiment_material or ""),
                material_splits=_parse_material_split_values(args.material_split),
                target_points=10000,
                seed=int(args.split_seed),
                waveform_crop_length=int(args.waveform_crop_length),
                limit_per_material=int(args.experiment_limit),
                train_manifest_name=str(args.train_manifest_name),
                validation_manifest_name=str(args.validation_manifest_name),
                test_manifest_name=str(args.test_manifest_name),
            )
            register_materials(
                config.data_root,
                material_roots.keys(),
                source="sample_material_folder",
            )
            trainer = ReconstructionTrainer(config)
            registrations: list[dict[str, object]] = []
            if bool(args.separate_materials):
                checkpoint = trainer.train_material_checkpoints(
                    collection_path,
                    train_name=demo_train_name or None,
                )
                training_strategy = "separate_material_checkpoints"
                if resolved_rule is not None:
                    dim, md, _dataset_rule_name = resolved_rule
                    _router_payload, registrations = _register_material_checkpoints_in_rule_table(
                        config,
                        router_path=checkpoint,
                        dimension=dim,
                        mode=md,
                    )
            else:
                mixed_manifest = build_mixed_collection_manifest(collection_path)
                checkpoint = trainer.train(
                    mixed_manifest,
                    train_name=demo_train_name or None,
                )
                training_strategy = "mixed_material_checkpoint"
                if resolved_rule is not None:
                    dim, md, mat = resolved_rule
                    registered = _register_checkpoint_in_rule_table(
                        config,
                        dimension=dim,
                        mode=md,
                        material=mat,
                        checkpoint_path=Path(checkpoint).resolve(),
                    )
                    registrations.append(
                        {
                            "rule_material": mat,
                            "checkpoint": str(Path(checkpoint).resolve()),
                            "rule_registered": registered,
                        }
                    )
            collection = load_material_collection(collection_path)
            report = _validate_material_collection(config, collection_path)
            print(
                json.dumps(
                    {
                        "data_root": str(collection_root),
                        "material_collection": str(collection_path),
                        "checkpoint": str(checkpoint),
                        "training_strategy": training_strategy,
                        "rule_registrations": registrations,
                        "report": report,
                        "materials": collection["materials"],
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return

        case_dirs = discover_cases(experiment_source) if experiment_source.exists() else []
        if case_dirs and int(args.experiment_limit) != 0:
            if bool(args.separate_materials):
                raise ValueError(
                    "当前选择的是单个样本材料目录；wumu 是一种完整材料，"
                    "不能按 layer_1/layer_2 分开训练。仅在选择多材料上层目录时启用分别训练"
                )
            if not bool(args.split_dataset):
                raise ValueError("固定节点 case 一键训练必须启用 train/validation/test 划分")
            build_data_root = (
                config.data_root
                / "data_process"
                / f"{material_bucket}_case_temperature_field"
            )
            combined = build_case_dataset(
                experiment_source,
                build_data_root,
                target_points=10000,
                seed=int(args.split_seed),
                waveform_crop_length=int(args.waveform_crop_length),
                limit=int(args.experiment_limit),
                test_ratio=float(args.split_test_ratio),
                validation_ratio=float(args.split_validation_ratio),
                dataset_label=str(args.experiment_material or ""),
                sample_material_key=experiment_source.name,
                sample_material_name=sample_material_name(experiment_source.name),
                train_manifest_name=str(args.train_manifest_name),
                validation_manifest_name=str(args.validation_manifest_name),
                test_manifest_name=str(args.test_manifest_name),
            )
            register_materials(
                config.data_root,
                [experiment_source.name],
                source="sample_material_folder",
            )
            split_payload = json.loads(
                (build_data_root / "split_config.json").read_text(encoding="utf-8")
            )
            train_manifest_path = Path(split_payload["manifests"]["train"])
            checkpoint = ReconstructionTrainer(config).train(
                train_manifest_path,
                train_name=demo_train_name or None,
            )
            registrations: list[dict[str, object]] = []
            if resolved_rule is not None:
                dim, md, mat = resolved_rule
                registered = _register_checkpoint_in_rule_table(
                    config,
                    dimension=dim,
                    mode=md,
                    material=mat,
                    checkpoint_path=Path(checkpoint).resolve(),
                )
                registrations.append(
                    {
                        "rule_material": mat,
                        "checkpoint": str(Path(checkpoint).resolve()),
                        "rule_registered": registered,
                    }
                )
            report = DatabaseBuilder(config).validate_requirement_33(combined)
            print(
                json.dumps(
                    {
                        "data_root": str(build_data_root),
                        "manifest": str(combined),
                        "checkpoint": str(checkpoint),
                        "report": report,
                        "training_strategy": "single_material_checkpoint",
                        "rule_registrations": registrations,
                        "split": split_payload["manifests"],
                        "split_config": str(build_data_root / "split_config.json"),
                        "dataset_schema_version": 2,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
            return
        material_bucket = _sanitize_material_label(str(args.experiment_material or "unknown_material"))
        build_data_root = config.data_root / "data_process" / material_bucket
        build_data_root.mkdir(parents=True, exist_ok=True)
        build_builder = DatabaseBuilder(config)
        build_builder.database_dir = build_data_root
        _apply_preprocess_args(config, args)
        sim_manifest = build_builder.build_simulation_database(samples_per_material=args.sim_per_material)
        exp_manifest = build_builder.import_experimental_csvs(
            source_dir=experiment_source,
            material_key=str(args.experiment_material or "metal_matrix"),
            limit=args.experiment_limit,
        )
        combined = build_builder.merge_manifests([sim_manifest, exp_manifest])
        train_manifest_path: Path = combined
        split_result: dict[str, object] | None = None
        if bool(args.split_dataset):
            train_manifest, test_manifest, split_stats = split_manifest_file(
                manifest_path=combined,
                test_ratio=float(args.split_test_ratio),
                seed=int(args.split_seed),
                experiment_policy=str(args.split_experiment_policy),
                train_output_name=str(args.train_manifest_name),
                test_output_name=str(args.test_manifest_name),
            )
            train_manifest_path = train_manifest
            split_result = {
                "train_manifest": str(train_manifest),
                "test_manifest": str(test_manifest),
                "split_stats": split_stats,
            }
        split_config_path = build_data_root / "split_config.json"
        split_config_path.write_text(
            json.dumps(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "experiment_material": str(args.experiment_material or ""),
                    "data_root": str(build_data_root),
                    "source_options": {
                        "skip_simulation": False,
                        "sim_per_material": int(args.sim_per_material),
                        "experiment_dir": str(experiment_source),
                        "experiment_limit": int(args.experiment_limit),
                        "external_sim_dir": "",
                        "external_sim_limit": 0,
                    },
                    "split_params": {
                        "split_dataset": bool(args.split_dataset),
                        "split_test_ratio": float(args.split_test_ratio),
                        "split_validation_ratio": float(args.split_validation_ratio),
                        "split_seed": int(args.split_seed),
                        "split_experiment_policy": str(args.split_experiment_policy),
                        "train_manifest_name": str(args.train_manifest_name),
                        "test_manifest_name": str(args.test_manifest_name),
                        "validation_manifest_name": str(args.validation_manifest_name),
                    },
                    "split_stats": split_result.get("split_stats") if split_result else None,
                    "manifests": {
                        "combined": str(combined),
                        "train": str(split_result["train_manifest"]) if split_result else None,
                        "test": str(split_result["test_manifest"]) if split_result else None,
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if bool(args.separate_materials):
            raise ValueError("--separate-materials 仅支持固定节点 case 数据")
        checkpoint = ReconstructionTrainer(config).train(
            train_manifest_path,
            train_name=demo_train_name or None,
        )
        registration: dict[str, object] | None = None
        if resolved_rule is not None:
            dim, md, mat = resolved_rule
            registered = _register_checkpoint_in_rule_table(
                config,
                dimension=dim,
                mode=md,
                material=mat,
                checkpoint_path=Path(checkpoint).resolve(),
            )
            registration = {
                "rule_material": mat,
                "checkpoint": str(Path(checkpoint).resolve()),
                "rule_registered": registered,
            }
        report = build_builder.validate_requirement_33(combined)
        print(
            json.dumps(
                {
                    "data_root": str(build_data_root),
                    "manifest": str(combined),
                    "checkpoint": str(checkpoint),
                    "rule_registration": registration,
                    "report": report,
                    "split": split_result,
                    "split_config": str(split_config_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
