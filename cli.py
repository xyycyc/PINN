from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from .config import AIModelConfig
from .data_process import (
    SPLIT_EXPERIMENT_POLICIES,
    DatabaseBuilder,
    build_case_dataset,
    discover_cases,
    parse_preprocess_steps,
    split_manifest_file,
)
from .model import (
    OnlineUpdater,
    ReconstructionTrainer,
    apply_training_runtime,
    build_rule_record,
    default_parameter_base_name,
    default_predict_output_name,
    ensure_rule_csv,
    predict_and_compare,
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


FIXED_WEIGHT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("--fixed-weight-cnn", "fixed_weight_cnn"),
    ("--fixed-weight-lstm", "fixed_weight_lstm"),
    ("--fixed-weight-material", "fixed_weight_material"),
    ("--fixed-weight-dimension", "fixed_weight_dimension"),
    ("--fixed-weight-mode", "fixed_weight_mode"),
)


def _add_fixed_weight_options(parser: argparse.ArgumentParser) -> None:
    """fixed 模式下各分支权重的 CLI 入口（不传则保留 AIModelConfig 默认值）。

    注意：--learnable-branch-weights 启用时这些权重不起作用，
    它们仅在 fixed 模式（默认）下被使用。
    """
    parser.add_argument(
        "--fixed-weight-cnn",
        type=float,
        default=None,
        help="CNN 分支固定权重（仅 fixed 模式生效）",
    )
    parser.add_argument(
        "--fixed-weight-lstm",
        type=float,
        default=None,
        help="LSTM 分支固定权重（仅 fixed 模式生效）",
    )
    parser.add_argument(
        "--fixed-weight-material",
        type=float,
        default=None,
        help="material embedding 分支固定权重（仅 fixed 模式生效）",
    )
    parser.add_argument(
        "--fixed-weight-dimension",
        type=float,
        default=None,
        help="dimension embedding 分支固定权重（仅 fixed 模式生效）",
    )
    parser.add_argument(
        "--fixed-weight-mode",
        type=float,
        default=None,
        help="mode embedding 分支固定权重（仅 fixed 模式生效）",
    )


def _add_runtime_options(parser: argparse.ArgumentParser, include_epochs: bool = False) -> None:
    """给需要训练模型的子命令追加统一运行参数。"""
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--learnable-branch-weights",
        action="store_true",
        help="启用分支权重学习；默认关闭，此时各分支权重固定为 1",
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
        parser.add_argument("--epochs", type=int, default=20)


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
        type=int,
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


def _add_preprocess_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--preprocess",
        type=str,
        default=None,
        help="实验波形可选预处理（训练/推理前），逗号分隔: clip,smooth,detrend,robust_norm；z-score 在模型输入前强制",
    )
    parser.add_argument("--clip-quantile", type=float, default=1.0, help="clip 分位数阈值（例如 1.0 表示 [1,99]）")
    parser.add_argument("--smooth-window", type=int, default=11, help="smooth 滑动窗口（自动转为奇数）")


def _apply_preprocess_args(config: AIModelConfig, args: argparse.Namespace) -> None:
    if getattr(args, "preprocess", None) is not None:
        config.preprocess_steps = str(args.preprocess)
    if hasattr(args, "clip_quantile"):
        config.clip_quantile = float(args.clip_quantile)
    if hasattr(args, "smooth_window"):
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
        help="是否在建库后自动划分 train/test manifest（默认启用）",
    )
    parser.add_argument(
        "--split-test-ratio",
        type=float,
        default=0.2,
        help="测试集比例 (0,1)，仅在启用 --split-dataset 时生效",
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


def _sanitize_material_label(material: str) -> str:
    label = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in str(material or "").strip())
    label = label.strip("_")
    return label or "unknown_material"


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
        help="实验数据目录；相对路径默认按 data_root 解析（如 raw/10times）",
    )
    build_db.add_argument("--experiment-material", type=str, default="metal_matrix")
    build_db.add_argument("--experiment-limit", type=int, default=2000)
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
    train.add_argument("--manifest", type=str, default=None)
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
    _add_preprocess_options(update)

    predict = subparsers.add_parser(
        "predict",
        help="基于 checkpoint 对指定 manifest 做推理，输出预测与真实值对比产物",
    )
    _add_io_root_options(predict)
    predict.add_argument("--manifest", type=str, required=True, help="待预测样本的 manifest.json")
    predict.add_argument("--checkpoint", type=str, required=False, default="", help="训练得到的 *.pt 文件")
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
    _add_preprocess_options(predict)

    demo = subparsers.add_parser("demo", help="一键构建数据库、训练并验证")
    _add_io_root_options(demo)
    demo.add_argument("--sim-per-material", type=int, default=400)
    demo.add_argument("--experiment-limit", type=int, default=2000)
    demo.add_argument("--train-name", type=str, default="")
    _add_dataset_split_options(demo)
    _add_runtime_options(demo, include_epochs=True)
    _add_preprocess_options(demo)
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = AIModelConfig()
    if hasattr(args, "data_root"):
        config.data_root = config.resolve_path(args.data_root)
    if hasattr(args, "result_root"):
        config.result_root = config.resolve_path(args.result_root)
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
        build_data_root.mkdir(parents=True, exist_ok=True)
        build_builder = DatabaseBuilder(config)
        build_builder.database_dir = build_data_root
        experiment_source = _resolve_source_dir(config, args.experiment_dir)
        canonical_case_source = config.data_root / "raw" / "calibration_sweep"
        # Preserve the visible GUI default while transparently routing the known
        # calibration case layout into the schema-v1 builder.
        if str(args.experiment_dir).replace("\\", "/").strip("/") == "raw/10times" and discover_cases(canonical_case_source):
            experiment_source = canonical_case_source
        case_dirs = discover_cases(experiment_source) if experiment_source.exists() else []
        if case_dirs and int(args.experiment_limit) != 0:
            if not bool(args.skip_simulation) or str(args.external_sim_dir or "").strip():
                raise ValueError("固定节点 case 数据不能与旧规则网格仿真 CSV 静默合并；请保持跳过内置/外部仿真")
            build_data_root = config.data_root / "data_process" / f"{material_bucket}_case_temperature_field"
            build_data_root.mkdir(parents=True, exist_ok=True)
            combined = build_case_dataset(
                experiment_source,
                build_data_root,
                target_points=10000,
                seed=int(args.split_seed),
                waveform_length=int(config.waveform_length),
                limit=int(args.experiment_limit),
                test_ratio=float(args.split_test_ratio),
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
                "dataset_schema_version": 1,
            }, ensure_ascii=False, indent=2))
            return
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
        if default_train_manifest.exists():
            default_manifest = default_train_manifest
        else:
            default_manifest = config.database_dir / "combined_manifest.json"
        manifest_path = args.manifest or str(default_manifest)
        checkpoint = ReconstructionTrainer(config).train(
            manifest_path,
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
        manifest_path = args.manifest or str(config.database_dir / "combined_manifest.json")
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

        checkpoint_path = Path(checkpoint_input).resolve()
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
            args.manifest,
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
            raise ValueError("predict 需要 --checkpoint，或提供完整 rule 三元组用于自动匹配")
        checkpoint_path = Path(checkpoint_input)
        sync_config_for_inference(
            config,
            checkpoint_path,
            rule_csv_path=rule_csv_path(config.data_root),
        )
        _apply_predict_cli_overrides(config, args)
        if args.output_dir:
            output_dir = Path(args.output_dir)
        else:
            predict_name = str(args.predict_name or "").strip() or default_predict_output_name(
                checkpoint_path
            )
            root = config.predict_inference_root if args.predict_kind == "inference" else config.predict_batch_root
            output_dir = root / predict_name
        metrics = predict_and_compare(
            cfg=config,
            checkpoint_path=checkpoint_path,
            manifest_path=args.manifest,
            output_dir=output_dir,
            sync_config_from_checkpoint=False,
            enable_plots=bool(args.plots),
            num_field_samples=args.num_field_samples,
            enable_benchmark=bool(args.benchmark),
            benchmark_warmup_samples=args.benchmark_warmup_samples,
            benchmark_runs=args.benchmark_runs,
        )
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    if args.command == "demo":
        canonical_case_source = config.data_root / "raw" / "calibration_sweep"
        if discover_cases(canonical_case_source):
            _apply_preprocess_args(config, args)
            build_data_root = config.data_root / "data_process" / "metal_matrix_case_temperature_field"
            combined = build_case_dataset(
                canonical_case_source, build_data_root, target_points=10000,
                seed=int(args.split_seed), waveform_length=int(config.waveform_length),
                limit=int(args.experiment_limit), test_ratio=float(args.split_test_ratio),
            )
            split_payload = json.loads((build_data_root / "split_config.json").read_text(encoding="utf-8"))
            train_manifest_path = Path(split_payload["manifests"]["train"])
            checkpoint = ReconstructionTrainer(config).train(train_manifest_path, train_name=args.train_name or None)
            report = DatabaseBuilder(config).validate_requirement_33(combined)
            print(json.dumps({
                "data_root": str(build_data_root), "manifest": str(combined),
                "checkpoint": str(checkpoint), "report": report,
                "split": {"train_manifest": split_payload["manifests"]["train"],
                          "validation_manifest": split_payload["manifests"]["validation"],
                          "test_manifest": split_payload["manifests"]["test"]},
                "split_config": str(build_data_root / "split_config.json"),
                "dataset_schema_version": 1,
            }, ensure_ascii=False, indent=2))
            return
        material_bucket = _sanitize_material_label("metal_matrix")
        build_data_root = config.data_root / "data_process" / material_bucket
        build_data_root.mkdir(parents=True, exist_ok=True)
        build_builder = DatabaseBuilder(config)
        build_builder.database_dir = build_data_root
        _apply_preprocess_args(config, args)
        sim_manifest = build_builder.build_simulation_database(samples_per_material=args.sim_per_material)
        exp_manifest = build_builder.import_experimental_csvs(
            source_dir=config.data_root / "raw" / "10times",
            material_key="metal_matrix",
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
                    "experiment_material": "metal_matrix",
                    "data_root": str(build_data_root),
                    "source_options": {
                        "skip_simulation": False,
                        "sim_per_material": int(args.sim_per_material),
                        "experiment_dir": str(config.data_root / "raw" / "10times"),
                        "experiment_limit": int(args.experiment_limit),
                        "external_sim_dir": "",
                        "external_sim_limit": 0,
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
        checkpoint = ReconstructionTrainer(config).train(train_manifest_path, train_name=args.train_name or None)
        report = build_builder.validate_requirement_33(combined)
        print(
            json.dumps(
                {
                    "data_root": str(build_data_root),
                    "manifest": str(combined),
                    "checkpoint": str(checkpoint),
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
