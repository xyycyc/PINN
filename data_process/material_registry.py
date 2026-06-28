from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Iterable


MATERIAL_REGISTRY_HEADERS = ("index", "material", "created_at", "source")


def material_csv_path(data_root: str | Path) -> Path:
    return Path(data_root) / "rule" / "material.csv"


def ensure_material_csv(data_root: str | Path) -> Path:
    path = material_csv_path(data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(MATERIAL_REGISTRY_HEADERS))
            writer.writeheader()
    return path


def _normalize_material_name(material: str) -> str:
    name = str(material or "").strip()
    if not name:
        raise ValueError("material 不能为空")
    return name


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, str]] = []
        for row in reader:
            rows.append({key: str(value or "") for key, value in row.items()})
    return rows


def load_material_to_idx(path_or_data_root: str | Path) -> dict[str, int]:
    path = Path(path_or_data_root)
    if path.suffix.lower() != ".csv":
        path = ensure_material_csv(path)
    rows = _read_rows(path)
    mapping: dict[str, int] = {}
    used: set[int] = set()
    fallback_idx = 0
    for row in rows:
        material = str(row.get("material", "")).strip()
        if not material:
            continue
        raw_idx = str(row.get("index", "")).strip()
        try:
            idx = int(raw_idx)
        except ValueError:
            idx = fallback_idx
        while idx in used:
            idx += 1
        mapping[material] = idx
        used.add(idx)
        fallback_idx = max(fallback_idx, idx + 1)
    return mapping


def register_material(
    path_or_data_root: str | Path,
    material: str,
    *,
    source: str = "manual",
) -> int:
    path = Path(path_or_data_root)
    if path.suffix.lower() != ".csv":
        path = ensure_material_csv(path)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(MATERIAL_REGISTRY_HEADERS))
                writer.writeheader()
    material_name = _normalize_material_name(material)
    mapping = load_material_to_idx(path)
    if material_name in mapping:
        return mapping[material_name]
    next_idx = max(mapping.values(), default=-1) + 1
    row = {
        "index": str(next_idx),
        "material": material_name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source": str(source or "manual"),
    }
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(MATERIAL_REGISTRY_HEADERS))
        if path.stat().st_size == 0:
            writer.writeheader()
        writer.writerow(row)
    return next_idx


def register_materials(
    path_or_data_root: str | Path,
    materials: Iterable[str],
    *,
    source: str = "manifest",
) -> dict[str, int]:
    result: dict[str, int] = {}
    for material in materials:
        name = str(material or "").strip()
        if not name:
            continue
        result[name] = register_material(path_or_data_root, name, source=source)
    return result
