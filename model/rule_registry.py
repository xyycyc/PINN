from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

RULE_DIR_NAME = "rule"
RULE_CSV_NAME = "trained_rules.csv"
DIMENSION_CHOICES = ("one", "two")
MODE_CHOICES = ("steady", "transient")
MATERIAL_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class RuleRecord:
    created_at: str
    dimension: str
    mode: str
    material: str
    parameter_name: str
    parameter_path: str
    train_name: str
    training_mode: str
    physics_residual_weight: float
    learnable_branch_weights: bool
    fixed_weight_cnn: float
    fixed_weight_lstm: float
    fixed_weight_material: float
    fixed_weight_dimension: float
    fixed_weight_mode: float


def validate_rule_triplet(dimension: str, mode: str, material: str) -> tuple[str, str, str]:
    dim = str(dimension).strip().lower()
    md = str(mode).strip().lower()
    mat = str(material).strip()
    if dim not in DIMENSION_CHOICES:
        raise ValueError(f"dimension must be one of {DIMENSION_CHOICES}, got: {dimension}")
    if md not in MODE_CHOICES:
        raise ValueError(f"mode must be one of {MODE_CHOICES}, got: {mode}")
    if not MATERIAL_PATTERN.fullmatch(mat):
        raise ValueError("material must be English letters/digits/underscore and start with a letter")
    return dim, md, mat


def rule_csv_path(data_root: Path | str) -> Path:
    return Path(data_root) / RULE_DIR_NAME / RULE_CSV_NAME


def ensure_rule_csv(data_root: Path | str) -> Path:
    csv_path = rule_csv_path(data_root)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists():
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "created_at",
                    "dimension",
                    "mode",
                    "material",
                    "parameter_name",
                    "parameter_path",
                    "train_name",
                    "training_mode",
                    "physics_residual_weight",
                    "learnable_branch_weights",
                    "fixed_weight_cnn",
                    "fixed_weight_lstm",
                    "fixed_weight_material",
                    "fixed_weight_dimension",
                    "fixed_weight_mode",
                ],
            )
            writer.writeheader()
    return csv_path


def default_training_time_stamp(now: datetime | None = None) -> str:
    """训练/增量任务时间戳，例如 ``2026_5_15_2349``。"""

    dt = now or datetime.now()
    return f"{dt.year}_{dt.month}_{dt.day}_{dt.strftime('%H%M')}"


def default_parameter_base_name(
    *,
    dimension: str,
    mode: str,
    material: str,
    now: datetime | None = None,
) -> str:
    dim, md, mat = validate_rule_triplet(dimension, mode, material)
    return f"{dim}_{md}_{mat}_{default_training_time_stamp(now)}"


def append_rule_record(csv_path: Path, record: RuleRecord) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists() and csv_path.stat().st_size > 0:
        payload = csv_path.read_bytes()
        if not payload.endswith(b"\n") and not payload.endswith(b"\r\n"):
            with csv_path.open("ab") as raw_out:
                raw_out.write(b"\n")
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "created_at",
                "dimension",
                "mode",
                "material",
                "parameter_name",
                "parameter_path",
                "train_name",
                "training_mode",
                "physics_residual_weight",
                "learnable_branch_weights",
                "fixed_weight_cnn",
                "fixed_weight_lstm",
                "fixed_weight_material",
                "fixed_weight_dimension",
                "fixed_weight_mode",
            ],
        )
        writer.writerow(
            {
                "created_at": record.created_at,
                "dimension": record.dimension,
                "mode": record.mode,
                "material": record.material,
                "parameter_name": record.parameter_name,
                "parameter_path": record.parameter_path,
                "train_name": record.train_name,
                "training_mode": record.training_mode,
                "physics_residual_weight": record.physics_residual_weight,
                "learnable_branch_weights": str(record.learnable_branch_weights),
                "fixed_weight_cnn": record.fixed_weight_cnn,
                "fixed_weight_lstm": record.fixed_weight_lstm,
                "fixed_weight_material": record.fixed_weight_material,
                "fixed_weight_dimension": record.fixed_weight_dimension,
                "fixed_weight_mode": record.fixed_weight_mode,
            }
        )


def infer_rule_triplet_from_train_name(train_name: str) -> tuple[str, str, str] | None:
    """从训练任务目录名（``{dim}_{mode}_{material}_{时间戳}``）反推规则三元组。"""

    name = str(train_name or "").strip()
    if not name:
        return None
    for dim in DIMENSION_CHOICES:
        if not name.startswith(f"{dim}_"):
            continue
        rest = name[len(dim) + 1 :]
        for mode in MODE_CHOICES:
            if not rest.startswith(f"{mode}_"):
                continue
            body = rest[len(mode) + 1 :]
            parts = body.rsplit("_", 4)
            if len(parts) < 5:
                continue
            material = "_".join(parts[:-4])
            if material and MATERIAL_PATTERN.fullmatch(material):
                return dim, mode, material
    return None


def resolve_rule_triplet_for_checkpoint(
    csv_path: Path | str,
    checkpoint_path: Path | str,
) -> tuple[str, str, str] | None:
    """根据已登记路径或任务目录名解析规则三元组。"""

    row = find_rule_row_for_checkpoint(csv_path, checkpoint_path)
    if row is not None:
        return (
            str(row.get("dimension", "")).strip().lower(),
            str(row.get("mode", "")).strip().lower(),
            str(row.get("material", "")).strip(),
        )
    train_name = Path(checkpoint_path).resolve().parent.name
    return infer_rule_triplet_from_train_name(train_name)


def find_rule_row_for_checkpoint(
    csv_path: Path | str,
    checkpoint_path: Path | str,
) -> dict[str, str] | None:
    target = Path(checkpoint_path).resolve()
    if not target.exists():
        return None
    csv_file = Path(csv_path)
    if not csv_file.exists():
        return None

    candidates: list[tuple[str, dict[str, str]]] = []
    with csv_file.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_path = str(row.get("parameter_path", "")).strip()
            if not raw_path:
                continue
            try:
                row_path = Path(raw_path).resolve()
            except OSError:
                continue
            if row_path == target:
                candidates.append((str(row.get("created_at", "")).strip(), dict(row)))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def is_checkpoint_registered(csv_path: Path | str, checkpoint_path: Path | str) -> bool:
    return find_rule_row_for_checkpoint(csv_path, checkpoint_path) is not None


def build_rule_record(
    *,
    dimension: str,
    mode: str,
    material: str,
    checkpoint_path: Path | str,
    training_mode: str,
    physics_residual_weight: float,
    learnable_branch_weights: bool,
    fixed_weight_cnn: float,
    fixed_weight_lstm: float,
    fixed_weight_material: float = 1.0,
    fixed_weight_dimension: float = 1.0,
    fixed_weight_mode: float = 1.0,
    created_at: datetime | None = None,
) -> RuleRecord:
    dim, md, mat = validate_rule_triplet(dimension, mode, material)
    ckpt = Path(checkpoint_path).resolve()
    dt = created_at or datetime.now()
    return RuleRecord(
        created_at=dt.isoformat(timespec="seconds"),
        dimension=dim,
        mode=md,
        material=mat,
        parameter_name=ckpt.name,
        parameter_path=str(ckpt),
        train_name=ckpt.parent.name,
        training_mode=str(training_mode),
        physics_residual_weight=float(physics_residual_weight),
        learnable_branch_weights=bool(learnable_branch_weights),
        fixed_weight_cnn=float(fixed_weight_cnn),
        fixed_weight_lstm=float(fixed_weight_lstm),
        fixed_weight_material=float(fixed_weight_material),
        fixed_weight_dimension=float(fixed_weight_dimension),
        fixed_weight_mode=float(fixed_weight_mode),
    )


def register_checkpoint_rule(csv_path: Path | str, record: RuleRecord) -> bool:
    """将检查点登记到规则表；已存在相同路径时跳过。返回是否新写入。"""

    if is_checkpoint_registered(csv_path, record.parameter_path):
        return False
    append_rule_record(Path(csv_path), record)
    return True


def discover_unregistered_checkpoints(
    csv_path: Path | str,
    checkpoint_root: Path | str,
    *,
    rows: list[dict[str, str]] | None = None,
) -> list[Path]:
    """扫描训练 checkpoint 目录，将磁盘上未登记且可识别三元组的 ``.pt`` 写入规则表。"""

    csv_file = Path(csv_path)
    root = Path(checkpoint_root)
    if not root.is_dir():
        return []

    existing_rows = rows
    if existing_rows is None and csv_file.exists():
        existing_rows = []
        with csv_file.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            existing_rows = [dict(row) for row in reader]

    registered: set[str] = set()
    if existing_rows:
        for row in existing_rows:
            raw = str(row.get("parameter_path", "")).strip()
            if raw:
                try:
                    registered.add(str(Path(raw).resolve()))
                except OSError:
                    registered.add(raw)

    triplet_by_train: dict[str, tuple[str, str, str]] = {}
    if existing_rows:
        for row in existing_rows:
            train_name = str(row.get("train_name", "")).strip()
            if not train_name or train_name in triplet_by_train:
                continue
            triplet = (
                str(row.get("dimension", "")).strip().lower(),
                str(row.get("mode", "")).strip().lower(),
                str(row.get("material", "")).strip(),
            )
            if all(triplet):
                triplet_by_train[train_name] = triplet  # type: ignore[assignment]

    added: list[Path] = []
    for train_dir in sorted(root.iterdir()):
        if not train_dir.is_dir():
            continue
        triplet = triplet_by_train.get(train_dir.name) or infer_rule_triplet_from_train_name(train_dir.name)
        if triplet is None:
            continue
        dim, md, mat = triplet
        for pt_path in sorted(train_dir.glob("*.pt")):
            try:
                resolved = str(pt_path.resolve())
            except OSError:
                resolved = str(pt_path)
            if resolved in registered:
                continue
            try:
                bundle = torch.load(pt_path, map_location="cpu")
            except Exception:
                continue
            if not isinstance(bundle, dict) or "model_state" not in bundle:
                continue
            config_dict = bundle.get("config")
            if isinstance(config_dict, dict):
                training_mode = str(config_dict.get("training_mode", "normal"))
                physics_residual_weight = float(config_dict.get("physics_residual_weight", 0.1))
                learnable = bool(str(config_dict.get("learnable_branch_weights", False)).lower() in {"1", "true", "yes"})
                fw_cnn = float(config_dict.get("fixed_weight_cnn", 0.75))
                fw_lstm = float(config_dict.get("fixed_weight_lstm", 0.75))
                fw_mat = float(config_dict.get("fixed_weight_material", 1.0))
                fw_dim = float(config_dict.get("fixed_weight_dimension", 1.0))
                fw_mode = float(config_dict.get("fixed_weight_mode", 1.0))
            else:
                training_mode = "normal"
                physics_residual_weight = 0.1
                learnable = False
                fw_cnn = fw_lstm = 0.75
                fw_mat = fw_dim = fw_mode = 1.0

            record = build_rule_record(
                dimension=dim,
                mode=md,
                material=mat,
                checkpoint_path=pt_path,
                training_mode=training_mode,
                physics_residual_weight=physics_residual_weight,
                learnable_branch_weights=learnable,
                fixed_weight_cnn=fw_cnn,
                fixed_weight_lstm=fw_lstm,
                fixed_weight_material=fw_mat,
                fixed_weight_dimension=fw_dim,
                fixed_weight_mode=fw_mode,
            )
            if register_checkpoint_rule(csv_file, record):
                registered.add(resolved)
                added.append(pt_path.resolve())
    return added


def resolve_checkpoint_from_rule(
    csv_path: Path,
    *,
    dimension: str,
    mode: str,
    material: str,
) -> Path:
    dim, md, mat = validate_rule_triplet(dimension, mode, material)
    if not csv_path.exists():
        raise FileNotFoundError(f"rule csv not found: {csv_path}")

    candidates: list[tuple[str, Path]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (
                str(row.get("dimension", "")).strip().lower() == dim
                and str(row.get("mode", "")).strip().lower() == md
                and str(row.get("material", "")).strip() == mat
            ):
                path = Path(str(row.get("parameter_path", "")).strip())
                created_at = str(row.get("created_at", "")).strip()
                if path.exists():
                    candidates.append((created_at, path))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint found for rule ({dim}, {md}, {mat}) in {csv_path}")
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]
