"""Shared, dependency-light helpers for the read-only PINN research audit."""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "research_audit"
STATS_ROOT = AUDIT_ROOT / "statistics"
FIGURE_ROOT = AUDIT_ROOT / "figures"
REPORT_ROOT = AUDIT_ROOT / "reports"
DEFAULT_RAW_ROOT = Path("E:/pinn_data")
REPOSITORY_RAW_ROOT = ROOT / "database" / "raw"
CASE_RE = re.compile(r"^case_(\d+)_T(\d+)p(\d+)K$")
POST0_RE = re.compile(r"^T([-+]?\d+(?:\.\d+)?)C(?:_|$)", re.I)
NUMERIC_RE = re.compile(r"^([-+]?\d+(?:\.\d+)?)\.csv$", re.I)


def ensure_output_dirs() -> None:
    for path in (STATS_ROOT, FIGURE_ROOT, REPORT_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def logical_path(path: Path, raw_root: Path = DEFAULT_RAW_ROOT) -> str:
    path = path.resolve()
    try:
        return "RAW_DATA_ROOT/" + path.relative_to(raw_root.resolve()).as_posix()
    except ValueError:
        try:
            return "REPOSITORY_ROOT/" + path.relative_to(ROOT.resolve()).as_posix()
        except ValueError:
            return path.name


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        seen: set[str] = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path, *, comment: str | None = None) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        if comment is None:
            reader = csv.reader(handle)
            rows = list(reader)
        else:
            rows = [row for row in csv.reader(handle) if row and not row[0].startswith(comment)]
    return (rows[0] if rows else []), (rows[1:] if rows else [])


def case_temperature_k(case_name: str) -> float | None:
    match = CASE_RE.match(case_name)
    if not match:
        return None
    return float(f"{match.group(2)}.{match.group(3)}")


def safe_float(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def pearson(x: Iterable[float], y: Iterable[float]) -> float | None:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def spearman(x: Iterable[float], y: Iterable[float]) -> float | None:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    if len(a) < 3:
        return None
    return pearson(rankdata(a), rankdata(b))


def linear_r2(x: Iterable[float], y: Iterable[float]) -> float | None:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    if len(a) < 3 or np.std(a) == 0:
        return None
    design = np.column_stack((np.ones(len(a)), a))
    beta, *_ = np.linalg.lstsq(design, b, rcond=None)
    pred = design @ beta
    denom = np.sum((b - np.mean(b)) ** 2)
    return float(1.0 - np.sum((b - pred) ** 2) / denom) if denom > 0 else None


def mutual_information_binned(x: Iterable[float], y: Iterable[float], bins: int = 8) -> float | None:
    a = np.asarray(list(x), dtype=float)
    b = np.asarray(list(y), dtype=float)
    if len(a) < 8 or np.std(a) == 0 or np.std(b) == 0:
        return None
    bins = max(2, min(int(bins), int(math.sqrt(len(a)))))
    joint, _, _ = np.histogram2d(a, b, bins=bins)
    joint = joint / np.sum(joint)
    px = np.sum(joint, axis=1, keepdims=True)
    py = np.sum(joint, axis=0, keepdims=True)
    expected = px @ py
    mask = joint > 0
    return float(np.sum(joint[mask] * np.log(joint[mask] / expected[mask])))


def read_numeric_columns(path: Path, *, skiprows: int = 0, delimiter: str = ",") -> np.ndarray:
    return np.loadtxt(path, delimiter=delimiter, skiprows=skiprows, dtype=np.float64)


def auc_score(y_true: np.ndarray, scores: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return None
    ranks = rankdata(s)
    return float((np.sum(ranks[y == 1]) - positives * (positives + 1) / 2) / (positives * negatives))


def json_number(value: float | None) -> float | None:
    return None if value is None or not math.isfinite(value) else float(value)
