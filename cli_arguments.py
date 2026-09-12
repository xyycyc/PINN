"""CLI argument definitions; importing help requires only the standard library."""

from __future__ import annotations

import argparse
import math

from .data_process.split_policy import SPLIT_EXPERIMENT_POLICIES


FIXED_WEIGHT_OPTIONS: tuple[tuple[str, str], ...] = (
    ("--fixed-weight-cnn", "fixed_weight_cnn"),
    ("--fixed-weight-lstm", "fixed_weight_lstm"),
)


LEGACY_NOOP_WEIGHT_OPTIONS: tuple[str, ...] = (
    "fixed_weight_material",
    "fixed_weight_dimension",
    "fixed_weight_mode",
)


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise argparse.ArgumentTypeError("必须是有限数字，不能使用 NaN 或无穷大")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是大于 0 的整数")
    return parsed


def _reject_legacy_noop_weights(args: argparse.Namespace) -> None:
    supplied = [
        name
        for name in LEGACY_NOOP_WEIGHT_OPTIONS
        if getattr(args, name, None) is not None
    ]
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
        type=_finite_float,
        default=None,
        help="CNN 分支权重；固定模式为常量，可学习模式为初始值",
    )
    parser.add_argument(
        "--fixed-weight-lstm",
        type=_finite_float,
        default=None,
        help="LSTM 分支权重；固定模式为常量，可学习模式为初始值",
    )
    parser.add_argument(
        "--fixed-weight-material",
        type=_finite_float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fixed-weight-dimension",
        type=_finite_float,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--fixed-weight-mode",
        type=_finite_float,
        default=None,
        help=argparse.SUPPRESS,
    )


def _add_runtime_options(
    parser: argparse.ArgumentParser, include_epochs: bool = False
) -> None:
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
        type=_finite_float,
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
        type=_finite_float,
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


def _add_preprocess_options(
    parser: argparse.ArgumentParser,
    *,
    inherit_from_checkpoint: bool = False,
) -> None:
    numeric_default: float | int | None = None if inherit_from_checkpoint else 1.0
    inheritance_help = "；不传时继承 checkpoint" if inherit_from_checkpoint else ""
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
        type=_finite_float,
        default=numeric_default,
        help=f"clip 分位数阈值{inheritance_help}",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=None if inherit_from_checkpoint else 11,
        help=f"smooth 滑动窗口（自动转为奇数）{inheritance_help}",
    )


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
        type=_finite_float,
        default=None,
        help="物理残差权重；默认与 checkpoint 一致",
    )
    _add_fixed_weight_options(parser)


def _add_dataset_split_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--split-dataset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在建库后自动划分 manifest（固定节点为 train/validation/test，默认启用）",
    )
    parser.add_argument(
        "--split-test-ratio",
        type=_finite_float,
        default=0.2,
        help="测试集比例 (0,1)，仅在启用 --split-dataset 时生效",
    )
    parser.add_argument(
        "--split-validation-ratio",
        type=_finite_float,
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


def _add_rule_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--rule-dimension", type=str, choices=["one", "two"], default=""
    )
    parser.add_argument(
        "--rule-mode", type=str, choices=["steady", "transient"], default=""
    )
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
        type=_finite_float,
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
    predict.add_argument(
        "--manifest", type=str, required=True, help="待预测样本的 manifest.json"
    )
    predict.add_argument(
        "--checkpoint",
        type=str,
        required=False,
        default="",
        help="训练得到的 *.pt 文件",
    )
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
