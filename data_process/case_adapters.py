"""Source-format detection and normalization for case-level experiment exports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .temperature_field import CASE_RE, CaseKey


LEGACY_CASE_FORMAT = "legacy_nested_config_v1"
DIRECT_CASE_FORMAT = "flat_direct_config_v1"
STRICT_MATERIAL_IDS = "strict"
STABLE_NAME_MATERIAL_IDS = "stable_name_for_invalid"


@dataclass(frozen=True)
class AdaptedCase:
    """The source-specific fields needed by the shared case build core."""

    case_dir: Path
    case_key: str
    source_format: str
    config_path: Path
    referenced_temperature_csv_path: str
    thermal_path: Path
    material_id_policy: str


def expected_thermal_csv(case_dir: str | Path) -> Path:
    return (
        Path(case_dir)
        / "heat"
        / "thermomechanical_steady"
        / "csv"
        / "thermomechanical_steady_nodes.csv"
    )


def _unique_config(case_dir: Path) -> tuple[Path, dict[str, Any]]:
    configs = sorted((case_dir / "configs").glob("*_ultrasonic_config.json"))
    if len(configs) != 1:
        raise ValueError(f"超声配置数量应为 1，实际为 {len(configs)}")
    config_path = configs[0]
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"超声配置根节点必须是对象: {config_path}")
    return config_path, payload


def detect_case_source_format(case_dir: str | Path) -> str:
    """Best-effort format detection for diagnostics before full validation."""

    try:
        _config_path, payload = _unique_config(Path(case_dir))
    except (OSError, ValueError, json.JSONDecodeError):
        return "unknown"
    nested_config = payload.get("config")
    if isinstance(nested_config, dict) and isinstance(nested_config.get("io"), dict):
        return LEGACY_CASE_FORMAT
    if isinstance(payload.get("io"), dict):
        return DIRECT_CASE_FORMAT
    return "unknown"


def _validate_local_pair(
    case_dir: Path,
    referenced: str,
    local_csv: Path,
) -> None:
    normalized = referenced.replace("\\", "/")
    normalized_casefold = normalized.casefold()
    if case_dir.name.casefold() not in normalized_casefold:
        raise ValueError(f"temperature_csv_path 的 case ID 不一致: {referenced}")
    if Path(normalized).name.casefold() != local_csv.name.casefold():
        raise ValueError(
            "temperature_csv_path 的文件名与本地热力 CSV 不一致: "
            f"{referenced}"
        )
    if not local_csv.is_file():
        raise FileNotFoundError(f"temperature_csv_path 本地镜像不存在: {local_csv}")
    if local_csv.stat().st_size <= 0:
        raise ValueError(f"热力 CSV 为空: {local_csv}")


def adapt_case_source(case_dir: str | Path) -> AdaptedCase:
    """Detect a legacy or direct-export case without relying on dataset names."""

    case = Path(case_dir)
    config_path, payload = _unique_config(case)
    nested_config = payload.get("config")
    if isinstance(nested_config, dict) and isinstance(nested_config.get("io"), dict):
        source_format = LEGACY_CASE_FORMAT
        io_payload = nested_config["io"]
        case_key = CaseKey.from_case_dir(case).value
        material_id_policy = STRICT_MATERIAL_IDS
    elif isinstance(payload.get("io"), dict):
        source_format = DIRECT_CASE_FORMAT
        io_payload = payload["io"]
        if case.parent.name.startswith("worker_"):
            case_key = CaseKey.from_case_dir(case).value
        else:
            # Direct exports intentionally have no worker namespace.  The
            # source-format namespace keeps the key stable without inventing a
            # worker number or depending on the dataset directory name.
            if CASE_RE.match(case.name) is None:
                raise ValueError(f"case 目录名无效: {case.name}")
            case_key = f"flat/{case.name}"
        material_id_policy = STABLE_NAME_MATERIAL_IDS
    else:
        raise ValueError(
            "无法识别超声配置格式：需要 config.io（旧格式）或 io（直接导出格式）"
        )

    referenced = str(io_payload.get("temperature_csv_path", "")).strip()
    if not referenced:
        raise ValueError(
            f"{source_format} 超声配置缺少 temperature_csv_path: {config_path}"
        )
    local_csv = expected_thermal_csv(case)
    _validate_local_pair(case, referenced, local_csv)
    return AdaptedCase(
        case_dir=case,
        case_key=case_key,
        source_format=source_format,
        config_path=config_path,
        referenced_temperature_csv_path=referenced,
        thermal_path=local_csv,
        material_id_policy=material_id_policy,
    )
