"""网络结构、训练与推理（预测）相关实现。"""

from __future__ import annotations

from .checkpoint_runtime import (
    apply_inference_config,
    apply_training_runtime,
    default_training_runtime,
    load_model_state_strict,
    resolve_checkpoint_config_dict,
    resolve_training_runtime,
    runtime_from_rule_row,
    sync_config_for_inference,
)
from .network import AIReconstructionModel, ConvEncoder, LSTMEncoder
from .predict import (
    default_predict_output_name,
    predict_and_compare,
    predict_collection_with_checkpoint,
    predict_with_material_router,
)
from .rule_registry import (
    DIMENSION_CHOICES,
    MODE_CHOICES,
    RuleRecord,
    append_rule_record,
    build_rule_record,
    default_parameter_base_name,
    discover_unregistered_checkpoints,
    ensure_rule_csv,
    find_rule_row_for_checkpoint,
    infer_rule_triplet_from_train_name,
    is_checkpoint_registered,
    register_checkpoint_rule,
    resolve_checkpoint_from_rule,
    resolve_rule_triplet_for_checkpoint,
    rule_csv_path,
    validate_rule_triplet,
)
from .trainer import (
    MATERIAL_ROUTER_KIND,
    OnlineUpdater,
    ReconstructionTrainer,
    _default_device,
)

__all__ = [
    "apply_inference_config",
    "apply_training_runtime",
    "build_rule_record",
    "default_training_runtime",
    "load_model_state_strict",
    "resolve_checkpoint_config_dict",
    "sync_config_for_inference",
    "discover_unregistered_checkpoints",
    "find_rule_row_for_checkpoint",
    "infer_rule_triplet_from_train_name",
    "is_checkpoint_registered",
    "register_checkpoint_rule",
    "resolve_rule_triplet_for_checkpoint",
    "resolve_training_runtime",
    "runtime_from_rule_row",
    "AIReconstructionModel",
    "ConvEncoder",
    "LSTMEncoder",
    "OnlineUpdater",
    "ReconstructionTrainer",
    "MATERIAL_ROUTER_KIND",
    "_default_device",
    "default_predict_output_name",
    "predict_and_compare",
    "predict_collection_with_checkpoint",
    "predict_with_material_router",
    "DIMENSION_CHOICES",
    "MODE_CHOICES",
    "RuleRecord",
    "append_rule_record",
    "default_parameter_base_name",
    "ensure_rule_csv",
    "resolve_checkpoint_from_rule",
    "rule_csv_path",
    "validate_rule_triplet",
]
