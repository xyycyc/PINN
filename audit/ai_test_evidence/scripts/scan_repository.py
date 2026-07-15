from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import subprocess
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO = Path(__file__).resolve().parents[3]
AUDIT_ROOT = Path(__file__).resolve().parents[1]
STATUSES = {
    "CONFIRMED",
    "FOUND",
    "NOT_FOUND",
    "AMBIGUOUS",
    "INCONSISTENT",
    "INACCESSIBLE",
    "NOT_EXECUTED",
    "NOT_APPLICABLE",
}


def rel(path: Path | str | None) -> str:
    if path is None:
        return ""
    p = Path(path)
    try:
        return str(p.resolve().relative_to(REPO.resolve())).replace("\\", "/")
    except Exception:
        return str(path).replace("\\", "/")


def run_git(args: list[str]) -> str:
    try:
        cp = subprocess.run(
            ["git", *args],
            cwd=REPO,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        return cp.stdout.strip() if cp.returncode == 0 else cp.stderr.strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def read_text(path: Path, limit: int | None = None) -> str:
    data = path.read_text(encoding="utf-8", errors="replace")
    return data if limit is None else data[:limit]


def load_json(path: Path) -> Any:
    return json.loads(read_text(path))


def line_of(path: Path, needle: str) -> str:
    try:
        for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
            if needle in line:
                return f"L{idx}"
    except Exception:
        pass
    return ""


def excerpt(path: Path, needle: str, max_len: int = 240) -> str:
    try:
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if needle in line:
                s = line.strip()
                return s if len(s) <= max_len else s[: max_len - 3] + "..."
    except Exception:
        pass
    return ""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def item(
    item_id: str,
    item_name: str,
    status: str,
    extracted_value: Any = None,
    source_type: str = "",
    source_path: str = "",
    source_location: str = "",
    evidence_excerpt: str = "",
    extraction_method: str = "",
    notes: str = "",
    requires_new_experiment: bool = False,
    requires_code_change: bool = False,
    requires_manual_confirmation: bool = False,
    category: str = "",
) -> dict[str, Any]:
    if status not in STATUSES:
        raise ValueError(status)
    if status in {"NOT_FOUND", "AMBIGUOUS", "INCONSISTENT", "INACCESSIBLE", "NOT_EXECUTED"} and extracted_value == "":
        extracted_value = None
    return {
        "item_id": item_id,
        "item_name": item_name,
        "status": status,
        "extracted_value": extracted_value,
        "source_type": source_type,
        "source_path": source_path,
        "source_location": source_location,
        "evidence_excerpt": evidence_excerpt,
        "extraction_method": extraction_method,
        "notes": notes,
        "requires_new_experiment": bool(requires_new_experiment),
        "requires_code_change": bool(requires_code_change),
        "requires_manual_confirmation": bool(requires_manual_confirmation),
        "category": category,
    }


def records_from_manifest(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [r for r in payload["records"] if isinstance(r, dict)]
    return []


def resolve_existing(manifest_path: Path, raw_path: str | None) -> tuple[Path | None, bool]:
    if not raw_path:
        return None, False
    p = Path(str(raw_path))
    if not p.is_absolute():
        p = manifest_path.parent / p
    return p, p.exists()


def csv_header(path: Path) -> list[str]:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            return next(reader, [])
    except Exception:
        return []


def count_csv_rows(path: Path) -> int:
    try:
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            return sum(1 for _ in reader)
    except Exception:
        return 0


def inspect_npz_shape(path: Path) -> dict[str, Any]:
    try:
        import numpy as np

        with np.load(path, allow_pickle=False) as data:
            return {key: list(data[key].shape) for key in data.files}
    except Exception as exc:
        return {"error": str(exc)}


def inspect_npy_shape(path: Path) -> list[int] | None:
    try:
        import numpy as np

        return list(np.load(path, mmap_mode="r").shape)
    except Exception:
        return None


def manifest_summary(path: Path) -> dict[str, Any]:
    try:
        payload = load_json(path)
    except Exception as exc:
        return {"path": rel(path), "read_error": str(exc), "records": []}
    records = records_from_manifest(payload)
    temps: list[float] = []
    keys: set[str] = set()
    sources = Counter()
    materials = Counter()
    dims = Counter()
    modes = Counter()
    invalid_wave = 0
    invalid_field = 0
    waveform_count = 0
    field_count = 0
    ids: list[str] = []
    for r in records:
        keys.update(r.keys())
        ids.append(str(r.get("sample_id", "")))
        for key, counter in (("source", sources), ("material_key", materials), ("dimension", dims), ("mode", modes)):
            counter[str(r.get(key, ""))] += 1
        try:
            temps.append(float(r["temperature_k"]))
        except Exception:
            pass
        for key, label in (("waveform_path", "wave"), ("field_path", "field")):
            _, ok = resolve_existing(path, r.get(key))
            if ok:
                if label == "wave":
                    waveform_count += 1
                else:
                    field_count += 1
            else:
                if label == "wave":
                    invalid_wave += 1
                else:
                    invalid_field += 1
    duplicates = sum(v - 1 for v in Counter(ids).values() if v > 1)
    return {
        "path": rel(path),
        "dataset_name": payload.get("dataset_name") if isinstance(payload, dict) else None,
        "dataset_label": payload.get("dataset_label") if isinstance(payload, dict) else None,
        "schema_version": payload.get("schema_version") if isinstance(payload, dict) else None,
        "sample_material_catalog": payload.get("sample_material_catalog") if isinstance(payload, dict) else None,
        "temperature_summary": payload.get("temperature_summary") if isinstance(payload, dict) else None,
        "record_count": len(records),
        "waveform_file_count_existing": waveform_count,
        "temperature_field_file_count_existing": field_count,
        "invalid_waveform_path_count": invalid_wave,
        "invalid_field_path_count": invalid_field,
        "temperature_min_k": min(temps) if temps else None,
        "temperature_max_k": max(temps) if temps else None,
        "near_300k_count": sum(1 for t in temps if 295 <= t <= 305),
        "at_1500k_count": sum(1 for t in temps if abs(t - 1500.0) <= 1e-6),
        "fields": sorted(keys),
        "missing_core_fields": [
            key
            for key in ["sample_id", "waveform_path", "field_path", "temperature_k", "dimension", "mode", "material_key", "source"]
            if key not in keys
        ],
        "duplicate_sample_count": duplicates,
        "source_counts": dict(sources),
        "material_counts": dict(materials),
        "dimension_counts": dict(dims),
        "mode_counts": dict(modes),
        "records": records,
    }


def classify_material(summary: dict[str, Any]) -> str | None:
    text = json.dumps(
        {
            "path": summary["path"],
            "dataset_label": summary.get("dataset_label"),
            "materials": summary.get("material_counts"),
            "catalog": summary.get("sample_material_catalog"),
        },
        ensure_ascii=False,
    ).lower()
    if "wumu" in text or "multilayer" in text or "多层" in text:
        return "multilayer"
    if "metal_matrix" in text or "metal-matrix" in text:
        return "metal_matrix"
    if "carbon_silicon" in text or "silicon" in text or "carbon" in text or "sic" in text:
        return "carbon_silicon"
    return None


def mode_dimension_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    result = {
        "one_steady": 0,
        "one_transient": 0,
        "two_steady": 0,
        "two_transient": 0,
    }
    for r in records:
        dim = str(r.get("dimension", "")).lower()
        mode = str(r.get("mode", "")).lower()
        if dim in {"1d", "one"} and mode == "steady":
            result["one_steady"] += 1
        elif dim in {"1d", "one"} and mode == "transient":
            result["one_transient"] += 1
        elif dim in {"2d", "two"} and mode == "steady":
            result["two_steady"] += 1
        elif dim in {"2d", "two"} and mode == "transient":
            result["two_transient"] += 1
    return result


def collect_prediction_runs() -> list[dict[str, Any]]:
    root = REPO / "result" / "predict"
    runs: list[dict[str, Any]] = []
    if not root.exists():
        return runs
    for metrics_path in sorted(root.rglob("metrics.json")):
        run_dir = metrics_path.parent
        try:
            metrics = load_json(metrics_path)
        except Exception:
            metrics = {}
        metadata_path = run_dir / "metadata.json"
        metadata = load_json(metadata_path) if metadata_path.exists() else {}
        pred_csv = run_dir / "predictions.csv"
        pred_npz = run_dir / "predictions.npz"
        header = csv_header(pred_csv) if pred_csv.exists() else []
        sample_ids: set[str] = set()
        row_count = 0
        if pred_csv.exists():
            try:
                with pred_csv.open("r", encoding="utf-8", errors="replace", newline="") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        row_count += 1
                        if row.get("sample_id"):
                            sample_ids.add(row["sample_id"])
            except Exception:
                pass
        runs.append(
            {
                "run_dir": rel(run_dir),
                "metrics_path": rel(metrics_path),
                "metadata_path": rel(metadata_path) if metadata_path.exists() else "",
                "predictions_csv": rel(pred_csv) if pred_csv.exists() else "",
                "predictions_npz": rel(pred_npz) if pred_npz.exists() else "",
                "prediction_header": header,
                "prediction_rows": row_count,
                "sample_count_from_csv": len(sample_ids),
                "metrics": metrics,
                "metadata": metadata,
                "npz_shapes": inspect_npz_shape(pred_npz) if pred_npz.exists() else {},
            }
        )
    return runs


def collect_training_reports() -> list[dict[str, Any]]:
    report_root = REPO / "result" / "train" / "report"
    rows: list[dict[str, Any]] = []
    if not report_root.exists():
        return rows
    for summary_path in sorted(report_root.rglob("training_summary.json")):
        try:
            summary = load_json(summary_path)
        except Exception:
            summary = {}
        rows.append({"path": rel(summary_path), "run_dir": rel(summary_path.parent), "summary": summary})
    for history_path in sorted(report_root.rglob("*_history.json")):
        if any(row.get("history_path") == rel(history_path) for row in rows):
            continue
        try:
            history = load_json(history_path)
        except Exception:
            history = []
        if isinstance(history, list):
            rows.append(
                {
                    "path": rel(history_path),
                    "run_dir": rel(history_path.parent),
                    "history_epoch_count": len(history),
                    "first_epoch": history[0] if history else None,
                    "last_epoch": history[-1] if history else None,
                }
            )
    return rows


def collect_checkpoint_rows() -> list[dict[str, str]]:
    path = REPO / "database" / "rule" / "trained_rules.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
        return list(csv.DictReader(f))


def checkpoint_metadata(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": rel(path), "exists": path.exists()}
    if not path.exists():
        return result
    result["size_bytes"] = path.stat().st_size
    result["sha256"] = sha256_file(path)
    try:
        import torch

        bundle = torch.load(path, map_location="cpu")
        if isinstance(bundle, dict):
            for key in [
                "checkpoint_version",
                "model_kind",
                "point_count",
                "sample_material_key",
                "base_checkpoint",
                "incremental_stamp",
            ]:
                if key in bundle:
                    result[key] = bundle.get(key)
            if isinstance(bundle.get("model_state"), dict):
                result["parameter_tensor_count"] = len(bundle["model_state"])
                result["parameter_total"] = sum(int(v.numel()) for v in bundle["model_state"].values() if hasattr(v, "numel"))
            if isinstance(bundle.get("config"), dict):
                result["config"] = {
                    key: bundle["config"].get(key)
                    for key in [
                        "waveform_length",
                        "learnable_branch_weights",
                        "fixed_weight_cnn",
                        "fixed_weight_lstm",
                        "training_mode",
                    ]
                    if key in bundle["config"]
                }
    except Exception as exc:
        result["load_error"] = str(exc)
    return result


def code_search(pattern: str, globs: list[str] | None = None) -> list[dict[str, Any]]:
    paths: list[Path] = []
    if globs:
        for g in globs:
            paths.extend(REPO.glob(g))
    else:
        paths = [p for p in REPO.rglob("*") if p.is_file()]
    regex = re.compile(pattern, re.IGNORECASE)
    hits = []
    for path in paths:
        if any(part in {".git", "audit", ".codex_tools", "__pycache__"} for part in path.parts):
            continue
        rel_parts = set(path.relative_to(REPO).parts) if path.is_relative_to(REPO) else set(path.parts)
        if {"database", "raw"} <= rel_parts or {"database", "preprocess"} <= rel_parts:
            continue
        if path.suffix.lower() not in {".py", ".md", ".json", ".csv", ".txt", ".ps1", ".m", ".f90", ".yaml", ".yml"}:
            continue
        try:
            if path.stat().st_size > 2 * 1024 * 1024:
                continue
        except Exception:
            continue
        try:
            for idx, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    hits.append({"path": rel(path), "line": idx, "text": line.strip()[:260]})
        except Exception:
            pass
    return hits


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        out.append("| " + " | ".join(str(x).replace("\n", "<br>") for x in row) + " |")
    return "\n".join(out)


def status_counts(items: list[dict[str, Any]], prefix: str | None = None) -> Counter[str]:
    c = Counter()
    for it in items:
        if prefix is None or it["item_id"].startswith(prefix):
            c[it["status"]] += 1
    return c


def main() -> int:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    scripts_dir = AUDIT_ROOT / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    branch = run_git(["branch", "--show-current"])
    base_commit = run_git(["rev-parse", "HEAD"])
    containing_branches = [
        line.replace("*", "").strip()
        for line in run_git(["branch", "--contains", base_commit]).splitlines()
        if line.strip()
    ]
    base_branch = next(
        (
            name
            for name in containing_branches
            if name != branch and not name.startswith("audit/")
        ),
        branch,
    )
    remote = run_git(["remote", "get-url", "origin"])
    tracked = run_git(["ls-files"]).splitlines()
    untracked = run_git(["ls-files", "--others", "--exclude-standard"]).splitlines()
    status_short = run_git(["status", "--short", "--branch"])
    git_log = run_git(["log", "--oneline", "--decorate", "--all", "--max-count", "80"])

    items: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []

    def add(it: dict[str, Any]) -> None:
        items.append(it)
        evidence_rows.append({k: it.get(k) for k in [
            "item_id",
            "item_name",
            "status",
            "extracted_value",
            "source_type",
            "source_path",
            "source_location",
            "evidence_excerpt",
            "extraction_method",
            "notes",
            "requires_new_experiment",
            "requires_code_change",
            "requires_manual_confirmation",
            "category",
        ]})

    add(item(
        "FACT-001",
        "实验测试材料",
        "CONFIRMED",
        "两层 W30Mo70 钨钼合金多层材料",
        "user_confirmed",
        "user prompt",
        "用户明确确认事实",
        "实验测试材料为：两层 W30Mo70 钨钼合金多层材料",
        "manual user confirmation",
        "该事实按用户要求不得改写为其他材料。",
        category="confirmed_fact",
    ))

    # Model input/output evidence.
    network = REPO / "model" / "network.py"
    point_field = REPO / "model" / "point_field.py"
    trainer = REPO / "model" / "trainer.py"
    predict = REPO / "model" / "predict.py"
    cli = REPO / "cli.py"
    window_app = REPO / "window" / "app.py"
    window_train = REPO / "window" / "tabs" / "train.py"
    window_predict = REPO / "window" / "tabs" / "predict.py"
    window_online = REPO / "window" / "tabs" / "online_update.py"
    dataset_py = REPO / "data_process" / "dataset.py"
    case_pipeline = REPO / "data_process" / "case_pipeline.py"
    readme = REPO / "README.md"

    add(item("A-001", "模型定义文件路径", "FOUND", ["model/network.py", "model/point_field.py"], "source_code", "model/network.py; model/point_field.py", "model/network.py:L45; model/point_field.py:L21", "class AIReconstructionModel(nn.Module); class DirectPointFieldModel(nn.Module)", "static code search", category="A"))
    add(item("A-002", "模型类名称", "FOUND", ["AIReconstructionModel", "DirectPointFieldModel"], "source_code", "model/network.py; model/point_field.py", "model/network.py:L45; model/point_field.py:L21", "class AIReconstructionModel(nn.Module); class DirectPointFieldModel(nn.Module)", "static code search", category="A"))
    add(item("A-003", "实际训练入口", "FOUND", "python -m ai_model train / ReconstructionTrainer.train", "source_code", "cli.py; model/trainer.py", "cli.py:L715; model/trainer.py:L230", "train subparser and ReconstructionTrainer.train", "static code search", category="A"))
    add(item("A-004", "实际推理入口", "FOUND", "python -m ai_model predict / predict_and_compare", "source_code", "cli.py; model/predict.py", "cli.py:L802; model/predict.py:L308", "predict subparser and predict_and_compare", "static code search", category="A"))
    add(item("A-005", "实际 GUI 调用入口", "FOUND", "python -m ai_model.window; window tabs compose CLI commands", "source_code", "README.md; window/app.py; window/tabs/predict.py", "README.md:L133; window/app.py; window/tabs/predict.py", "python -m ai_model.window", "static code search", category="A"))
    add(item("A-006", "模型输入字段", "FOUND", "forward(waveform: torch.Tensor)", "source_code", "model/network.py; model/point_field.py", "model/network.py:L95; model/point_field.py:L46", "def forward(self, waveform: torch.Tensor)", "static code search", category="A"))
    add(item("A-007", "是否直接输入原始超声全波形", "AMBIGUOUS", None, "source_code+manifest", "model/waveform_io.py; manifests", "", "prepare_model_waveform_input(batch['waveform']); waveform_processing fixed_prefix_native_rate", "static code and manifest scan", "代码输入 waveform 张量；W30Mo70 manifest 记录 fixed_prefix_native_rate 和 crop_length=1097。是否等同“原始超声全波形”无法直接证明，因为存在裁剪。", requires_manual_confirmation=True, category="A"))
    add(item("A-008", "是否输入声时/幅值/衰减/振幅谱/中心频率等声学特征", "AMBIGUOUS", {"legacy_builder_acoustic_fields": ["tof", "amplitude", "center_freq"], "model_forward_input": "waveform only"}, "source_code", "data_process/builder.py; model/network.py; model/point_field.py", "data_process/builder.py:L158-L161; model/network.py:L95; model/point_field.py:L46", "\"tof\", \"amplitude\", \"center_freq\" exist in builder acoustic records; model forward accepts waveform only", "static code search", "旧数据构建记录声学字段，但模型 forward 未显示这些字段作为输入。不能确认正式测试中声学特征作为独立输入。", requires_manual_confirmation=True, category="A"))
    add(item("A-009", "是否输入材料参数", "NOT_FOUND", None, "source_code", "model/network.py; model/point_field.py", "forward signatures", "forward(... waveform ...)", "static code search", "未找到材料物性参数进入 forward 的直接证据。", requires_code_change=True, category="A"))
    add(item("A-010", "是否输入空间坐标", "NOT_FOUND", None, "source_code", "model/point_field.py; model/predict.py", "model/point_field.py:L46", "forward(self, waveform: torch.Tensor)", "static code search", "坐标用于输出表和绘图；未找到作为模型输入的直接证据。", requires_code_change=True, category="A"))
    add(item("A-011", "是否输入时间坐标", "NOT_FOUND", None, "source_code", "model/network.py; model/point_field.py", "forward signatures", "forward(... waveform ...)", "static code search", "未找到时间坐标作为独立模型输入的直接证据。", requires_code_change=True, category="A"))
    add(item("A-012", "是否输入 dimension/mode/material 条件", "NOT_FOUND", None, "source_code+README", "README.md; model/point_field.py", "README.md:L346; model/point_field.py:L46", "当前网络没有这三个前向分支", "static code/doc scan", "规则表有 dimension/mode/material，但未作为模型 forward 输入。", requires_code_change=True, category="A"))
    add(item("A-013", "模型输出类型", "FOUND", {"AIReconstructionModel": "field 24x24 + acoustic + temperature", "DirectPointFieldModel": "P fixed-node temperatures"}, "source_code", "model/network.py; model/point_field.py; README.md", "model/network.py:L107-L116; model/point_field.py:L46-L58; README.md:L186-L187", "field_head(...).view(...); DirectPointFieldModel returns concatenated heads; fixed-node manifest -> output 10,000 points", "static code/doc scan", category="A"))

    # Manifests and datasets.
    manifest_paths = sorted(
        [
            p
            for p in REPO.rglob("*.json")
            if "manifest" in p.name.lower()
            and ".git" not in p.parts
            and "audit" not in p.parts
        ]
    )
    manifest_summaries = [manifest_summary(p) for p in manifest_paths]
    training_rows: list[dict[str, Any]] = []
    by_material: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for s in manifest_summaries:
        m = classify_material(s)
        if m:
            by_material[m].append(s)
        if "train_manifest" in s["path"].lower() or "manifest.json" in s["path"].lower():
            counts = mode_dimension_counts(s.get("records", []))
            training_rows.append(
                {
                    "material_category": m or "",
                    "manifest_path": s["path"],
                    "dataset_name": s.get("dataset_name"),
                    "dataset_label": s.get("dataset_label"),
                    "sample_total": s["record_count"],
                    "waveform_file_count": s["waveform_file_count_existing"],
                    "temperature_field_file_count": s["temperature_field_file_count_existing"],
                    "temperature_min_k": s["temperature_min_k"],
                    "temperature_max_k": s["temperature_max_k"],
                    "near_300k_count": s["near_300k_count"],
                    "at_1500k_count": s["at_1500k_count"],
                    **counts,
                    "available_fields": ";".join(s["fields"]),
                    "missing_fields": ";".join(s["missing_core_fields"]),
                    "duplicate_sample_count": s["duplicate_sample_count"],
                    "invalid_path_count": int(s["invalid_waveform_path_count"]) + int(s["invalid_field_path_count"]),
                }
            )

    write_csv(AUDIT_ROOT / "02_simulation_training_sets.csv", training_rows)

    for cat, name in [
        ("multilayer", "多层材料训练集"),
        ("metal_matrix", "金属基复合材料训练集"),
        ("carbon_silicon", "碳基/硅基复合材料训练集"),
    ]:
        summaries = by_material.get(cat, [])
        if summaries:
            best = max(summaries, key=lambda s: int(s.get("record_count") or 0))
            add(item(f"B-{cat}", name, "FOUND", {k: v for k, v in best.items() if k != "records"}, "manifest", best["path"], "JSON records/statistics", json.dumps({k: best.get(k) for k in ["record_count", "temperature_min_k", "temperature_max_k", "material_counts"]}, ensure_ascii=False), "manifest static count", category="B"))
        else:
            add(item(f"B-{cat}", name, "NOT_FOUND", None, "repository_scan", "", "", "", "manifest/code search", f"未找到可归类为 {cat} 的现有 manifest 记录。", requires_new_experiment=True, category="B"))

    sim_total = 0
    sim_temps: list[float] = []
    for s in manifest_summaries:
        for r in s.get("records", []):
            src = str(r.get("source", "")).lower()
            if "sim" in src:
                sim_total += 1
                try:
                    sim_temps.append(float(r["temperature_k"]))
                except Exception:
                    pass
    add(item("B-TOTAL-SIM", "仿真波形总数是否由直接统计证明达到 1000 组", "FOUND" if sim_total else "NOT_FOUND", sim_total if sim_total else None, "manifest", "all manifest files", "records[].source contains sim", str(sim_total), "manifest static count", "仅统计 source 字段含 sim 的记录；不会把 experiment_case 计入仿真。", requires_new_experiment=(sim_total < 1000), category="B"))
    if sim_temps:
        covers = min(sim_temps) <= 300.0 and max(sim_temps) >= 1500.0
        add(item("B-TEMP-COVERAGE", "仿真记录温度是否覆盖室温至 1500 K", "FOUND" if covers else "AMBIGUOUS", {"min_k": min(sim_temps), "max_k": max(sim_temps)}, "manifest", "all manifest files", "records[].temperature_k where source contains sim", "", "manifest static count", "仅基于 source 含 sim 的记录。", requires_new_experiment=not covers, category="B"))
    else:
        add(item("B-TEMP-COVERAGE", "仿真记录温度是否覆盖室温至 1500 K", "NOT_FOUND", None, "manifest", "all manifest files", "records[].temperature_k", "", "manifest static count", "未找到 source 明确为仿真的记录。", requires_new_experiment=True, category="B"))

    # W30Mo70 experiment data.
    wumu_summaries = [s for s in manifest_summaries if classify_material(s) == "multilayer"]
    exp_rows: list[dict[str, Any]] = []
    for s in wumu_summaries:
        for r in s.get("records", []):
            wave_path, wave_ok = resolve_existing(REPO / s["path"], r.get("waveform_path"))
            field_path, field_ok = resolve_existing(REPO / s["path"], r.get("field_path"))
            if len(exp_rows) < 1000:
                exp_rows.append(
                    {
                        "manifest_path": s["path"],
                        "sample_id": r.get("sample_id"),
                        "temperature_K": r.get("temperature_k"),
                        "mode": r.get("mode"),
                        "dimension": r.get("dimension"),
                        "source": r.get("source"),
                        "waveform_path": rel(wave_path) if wave_path else "",
                        "waveform_exists": wave_ok,
                        "field_path": rel(field_path) if field_path else "",
                        "field_exists": field_ok,
                        "original_waveform_path": r.get("meta", {}).get("original_waveform_path") if isinstance(r.get("meta"), dict) else "",
                    }
                )
    write_csv(AUDIT_ROOT / "03_w30mo70_experiment_data.csv", exp_rows)
    wumu_best = max(wumu_summaries, key=lambda s: int(s.get("record_count") or 0), default=None)
    if wumu_best:
        add(item("C-001", "两层 W30Mo70 实验数据 manifest/记录", "FOUND", {"manifest": wumu_best["path"], "records": wumu_best["record_count"], "temperature_min_k": wumu_best["temperature_min_k"], "temperature_max_k": wumu_best["temperature_max_k"]}, "manifest", wumu_best["path"], "records[]", json.dumps({k: wumu_best.get(k) for k in ["dataset_label", "record_count", "temperature_summary"]}, ensure_ascii=False), "manifest static count", "材料身份以用户确认事实为准；manifest 中 material_key 多处为 wumu。", category="C"))
        add(item("C-020", "是否具有不少于 20 组实验采集的直接证据", "FOUND" if wumu_best["record_count"] >= 20 else "NOT_FOUND", wumu_best["record_count"], "manifest", wumu_best["path"], "records[]", f"record_count={wumu_best['record_count']}", "manifest static count with path validation", "计数来自 manifest 记录，不仅凭目录文件数量。", requires_new_experiment=wumu_best["record_count"] < 20, category="C"))
    else:
        add(item("C-001", "两层 W30Mo70 实验数据 manifest/记录", "NOT_FOUND", None, "repository_scan", "", "", "", "manifest scan", "未找到 wumu/multilayer manifest。", requires_manual_confirmation=True, category="C"))
        add(item("C-020", "是否具有不少于 20 组实验采集的直接证据", "NOT_FOUND", None, "repository_scan", "", "", "", "manifest scan", "未找到可审计的 W30Mo70 manifest。", requires_new_experiment=True, category="C"))
    ai_test_refs = [s for s in wumu_summaries if "test_manifest" in s["path"].lower()]
    add(item("C-021", "实验数据是否进入 AI 测试集", "FOUND" if ai_test_refs else "NOT_FOUND", [s["path"] for s in ai_test_refs] or None, "manifest", "; ".join(s["path"] for s in ai_test_refs), "file name test_manifest + records", "", "manifest scan", category="C"))
    inc_refs = [r for r in collect_checkpoint_rows() if "increment" in str(r).lower() or re.search(r"\\\d{4}_\d+_\d+_\d+\.pt$", str(r.get("parameter_path", "")))]
    add(item("C-022", "实验数据是否进入增量训练集", "AMBIGUOUS", None, "trained_rules+reports", "database/rule/trained_rules.csv", "incremental rows", "", "rule table scan", "存在增量 checkpoint 记录，但 rule table 不直接给出新增数据 manifest，不能确认 W30Mo70 实验数据进入增量训练。", requires_manual_confirmation=True, category="C"))

    # Unified database.
    combined = [s for s in manifest_summaries if "combined" in s["path"].lower() or str(s.get("dataset_name", "")).endswith("combined_db")]
    if combined:
        add(item("D-001", "是否存在统一数据库/总 manifest", "FOUND", [s["path"] for s in combined], "manifest", "; ".join(s["path"] for s in combined), "dataset_name/path contains combined", "", "manifest scan", "发现 combined manifest；是否覆盖全部三类材料需逐项看字段统计。", category="D"))
    else:
        add(item("D-001", "是否存在统一数据库/总 manifest", "NOT_FOUND", None, "repository_scan", "", "", "", "manifest scan", "未找到原生统一数据库；本次仅生成审计索引 generated_for_audit_only=true。", requires_manual_confirmation=True, category="D"))
    db_fields = set()
    db_materials = set()
    db_sources = set()
    test_record_total = 0
    for s in combined or manifest_summaries:
        db_fields.update(s.get("fields", []))
        db_materials.update(k for k in s.get("material_counts", {}) if k)
        db_sources.update(k for k in s.get("source_counts", {}) if k)
    for s in manifest_summaries:
        if "test_manifest" in s["path"].lower():
            test_record_total += int(s.get("record_count") or 0)
    for fid, fname, key in [
        ("D-waveform", "是否包含波形路径", "waveform_path"),
        ("D-field", "是否包含温度场路径", "field_path"),
        ("D-temp", "是否包含温度值", "temperature_k"),
        ("D-dim", "是否包含 dimension", "dimension"),
        ("D-mode", "是否包含 mode", "mode"),
        ("D-material", "是否包含 material", "material_key"),
        ("D-source", "是否包含 source", "source"),
        ("D-field-mask", "是否包含 field_mask 或等价字段", "field_mask"),
    ]:
        add(item(fid, fname, "FOUND" if key in db_fields else "NOT_FOUND", key if key in db_fields else None, "manifest", "combined manifests or all manifests fallback", "record keys", key if key in db_fields else "", "manifest field scan", "field_mask 只在数据加载 batch 中出现，manifest 中未必存在。" if key == "field_mask" else "", requires_code_change=(key == "field_mask" and key not in db_fields), category="D"))
    add(item("D-100", "是否能够提取至少 100 条测试记录", "FOUND" if test_record_total >= 100 else "NOT_FOUND", test_record_total if test_record_total else None, "manifest", "test_manifest.json files", "records[]", f"test_record_total={test_record_total}", "manifest static count", requires_new_experiment=test_record_total < 100, category="D"))

    # Prediction results.
    prediction_runs = collect_prediction_runs()
    pred_rows = []
    for run in prediction_runs:
        m = run["metrics"]
        md = run["metadata"]
        pred_rows.append(
            {
                "run_dir": run["run_dir"],
                "predictions_csv": run["predictions_csv"],
                "predictions_npz": run["predictions_npz"],
                "samples_metric": m.get("samples"),
                "point_count": m.get("point_count"),
                "model_kind": m.get("model_kind") or md.get("model_kind"),
                "checkpoint": m.get("checkpoint") or md.get("checkpoint"),
                "manifest": m.get("manifest"),
                "eval_total_seconds": m.get("eval_total_seconds"),
                "eval_latency_ms_per_sample": m.get("eval_latency_ms_per_sample"),
                "benchmark_enabled": m.get("benchmark_enabled") or m.get("enable_benchmark"),
                "mae_k": (m.get("full_field") or {}).get("mae_k") if isinstance(m.get("full_field"), dict) else None,
                "rmse_k": (m.get("full_field") or {}).get("rmse_k") if isinstance(m.get("full_field"), dict) else None,
                "max_abs_k": (m.get("full_field") or {}).get("max_absolute_error_k") if isinstance(m.get("full_field"), dict) else None,
                "field_plot_samples": m.get("field_plot_samples"),
                "prediction_header": ";".join(run["prediction_header"]),
            }
        )
    write_csv(AUDIT_ROOT / "prediction_runs.csv", pred_rows)
    categories = {
        "one_steady": False,
        "one_transient": False,
        "two_steady": False,
        "two_transient": False,
    }
    for row in pred_rows:
        txt = json.dumps(row, ensure_ascii=False).lower()
        if "two" in txt and "steady" in txt:
            categories["two_steady"] = True
        if "one" in txt and "steady" in txt:
            categories["one_steady"] = True
        if "two" in txt and "transient" in txt:
            categories["two_transient"] = True
        if "one" in txt and "transient" in txt:
            categories["one_transient"] = True
    for key, cname in [
        ("one_steady", "一维稳态 AI 预测结果"),
        ("one_transient", "一维瞬态 AI 预测结果"),
        ("two_steady", "二维稳态 AI 预测结果"),
        ("two_transient", "二维瞬态 AI 预测结果"),
    ]:
        add(item(f"E-{key}", cname, "FOUND" if categories[key] else "NOT_FOUND", key if categories[key] else None, "prediction_artifacts", "result/predict", "metrics/metadata/predictions", "", "prediction artifact scan", "仅根据现有预测目录、metrics、metadata 和 predictions 表判断。", requires_new_experiment=not categories[key], category="E"))
    wumu_prediction_runs = [r for r in pred_rows if "wumu" in json.dumps(r, ensure_ascii=False).lower()]
    if wumu_prediction_runs:
        add(item("E-W30MO70", "W30Mo70 现有推理结果", "FOUND", wumu_prediction_runs, "prediction_artifacts", "result/predict/inference", "metrics.json / metadata.json / predictions.csv", "", "artifact scan", "不以目录名推断材料；仅列出路径/metadata/manifest/checkpoint 中出现 wumu 的结果。", category="E"))
    ten_percent = []
    for row in pred_rows:
        if row.get("mae_k") is not None and row.get("max_abs_k") is not None:
            ten_percent.append(row)
    add(item("E-ERR-10PCT", "是否存在相对误差不大于 10% 的直接证据", "NOT_FOUND", None, "prediction_artifacts", "metrics.json files", "full_field metrics", "", "metrics scan", "现有 metrics 提供 K 绝对误差；未发现相对误差 <=10% 的直接字段。", requires_new_experiment=True, category="E"))

    # Five points.
    five_hits = code_search(r"five|5\s*points|测点|measurement")
    add(item("F-001", "五个测点坐标/编号/误差", "NOT_FOUND" if not five_hits else "AMBIGUOUS", None, "repository_scan", "; ".join(h["path"] for h in five_hits[:5]), "; ".join(f"L{h['line']}" for h in five_hits[:5]), "; ".join(h["text"] for h in five_hits[:3]), "text/code search", "未找到明确五个测点坐标、参考温度、预测温度和五点平均相对误差结果的完整证据。", requires_manual_confirmation=True, category="F"))

    # Incremental training and checkpoint metadata.
    ckpt_paths = sorted((REPO / "result" / "train" / "checkpoint").rglob("*.pt")) if (REPO / "result" / "train" / "checkpoint").exists() else []
    ckpt_meta = [checkpoint_metadata(p) for p in ckpt_paths]
    write_csv(AUDIT_ROOT / "checkpoint_inventory.csv", ckpt_meta)
    inc_ckpts = [m for m in ckpt_meta if m.get("base_checkpoint") or re.search(r"/\d{4}_\d+_\d+_\d+\.pt$", str(m.get("path", "")))]
    add(item("G-001", "增量训练入口", "FOUND", "python -m ai_model online-update / ReconstructionTrainer.incremental_train", "source_code", "cli.py; model/trainer.py; window/tabs/online_update.py", "cli.py:L724; model/trainer.py:L715; window/tabs/online_update.py:L33", "online-update; Incremental", "static code search", category="G"))
    add(item("G-002", "已生成增量 checkpoint", "FOUND" if inc_ckpts else "NOT_FOUND", inc_ckpts or None, "checkpoint_inventory", "result/train/checkpoint", "*.pt metadata", "", "checkpoint metadata scan", requires_new_experiment=not inc_ckpts, category="G"))
    inc_reports = [r for r in collect_training_reports() if "/Incremental/" in r["run_dir"].replace("\\", "/")]
    add(item("G-003", "增量训练日志/loss 记录", "FOUND" if inc_reports else "NOT_FOUND", inc_reports or None, "training_report", "result/train/report/**/Incremental", "*_history.json/csv", "", "training report scan", requires_new_experiment=not inc_reports, category="G"))
    comparison_files = sorted(AUDIT_ROOT.glob("checkpoint_comparison_*.json"))
    comparison_summaries: list[dict[str, Any]] = []
    for comparison_file in comparison_files:
        try:
            payload = load_json(comparison_file)
            comparison_summaries.append(
                {
                    "path": rel(comparison_file),
                    "base_checkpoint": payload.get("base_checkpoint"),
                    "updated_checkpoint": payload.get("updated_checkpoint"),
                    "base_sha256": payload.get("base_sha256"),
                    "updated_sha256": payload.get("updated_sha256"),
                    "parameter_total": payload.get("parameter_total"),
                    "same_parameter_count": payload.get("same_parameter_count"),
                    "changed_parameter_count": payload.get("changed_parameter_count"),
                    "max_absolute_parameter_diff": payload.get("max_absolute_parameter_diff"),
                    "mean_absolute_parameter_diff": payload.get("mean_absolute_parameter_diff"),
                    "parameter_norm_change": payload.get("parameter_norm_change"),
                }
            )
        except Exception:
            pass
    if comparison_summaries:
        add(item("G-004", "训练前后权重变化证据", "FOUND", comparison_summaries, "checkpoint_comparison", "; ".join(s["path"] for s in comparison_summaries), "comparison JSON", "", "explicit read-only checkpoint comparison", "比较由 `scripts/compare_checkpoints.py` 单独执行；未修改 checkpoint。", category="G"))
    else:
        add(item("G-004", "训练前后权重变化证据", "NOT_EXECUTED", None, "checkpoint_inventory", "result/train/checkpoint", "", "", "checkpoint comparison script not run by default", "已提供 compare_checkpoints.py；主扫描不默认执行 checkpoint 比较。", requires_manual_confirmation=True, category="G"))

    # Transfer/freezing.
    freeze_hits = code_search(r"requires_grad|freeze|冻结|fixed_weight|learnable_branch")
    add(item("H-001", "迁移学习/参数冻结代码", "AMBIGUOUS" if freeze_hits else "NOT_FOUND", freeze_hits[:20] if freeze_hits else None, "source_code", "; ".join(sorted({h["path"] for h in freeze_hits[:10]})), "grep hits", "; ".join(h["text"] for h in freeze_hits[:3]), "static code search", "找到 fixed_weight/learnable_branch 相关实现和旧规则字段；未找到明确 requires_grad=False 冻结层执行日志。", requires_manual_confirmation=True, category="H"))
    add(item("H-002", "冻结机制实际执行日志", "NOT_FOUND", None, "training_report", "result/train/report", "history/summary", "", "report scan", "未找到直接证明冻结层生效的执行日志。", requires_new_experiment=True, category="H"))

    # Timing.
    ai_timing = [row for row in pred_rows if row.get("eval_total_seconds") is not None or row.get("benchmark_enabled")]
    add(item("I-001", "AI 推理耗时记录", "FOUND" if ai_timing else "NOT_FOUND", ai_timing or None, "metrics", "result/predict/**/metrics.json", "eval_total_seconds/eval_latency_ms_per_sample", "", "metrics scan", "eval_total_seconds 可能包含数据读取、预处理、绘图或保存；benchmark_enabled=false 时不能作为纯前向耗时。", requires_manual_confirmation=True, category="I"))
    target_timing_hits = code_search(r"0\.248766|0\.0622")
    add(item("I-002", "提示中的 4 场 0.248766 s / 0.0622 s 记录", "FOUND" if target_timing_hits else "NOT_FOUND", target_timing_hits if target_timing_hits else None, "repository_scan", "; ".join(h["path"] for h in target_timing_hits), "; ".join(f"L{h['line']}" for h in target_timing_hits), "; ".join(h["text"] for h in target_timing_hits[:3]), "text search", "如果为 NOT_FOUND，则不得在报告中采用提示数值。", requires_new_experiment=not target_timing_hits, category="I"))
    trad_hits = code_search(r"18\.67|11\.10|12\.44|共轭|灵敏度|最速|conjugate|sensitivity|steepest")
    add(item("J-001", "传统算法耗时及比较口径", "AMBIGUOUS" if trad_hits else "NOT_FOUND", trad_hits[:30] if trad_hits else None, "repository_scan", "; ".join(sorted({h["path"] for h in trad_hits[:10]})), "grep hits", "; ".join(h["text"] for h in trad_hits[:3]), "text/code search", "搜索命中可能来自波形 CSV 数值或代码注释；未找到能证明与 AI 同输入/同工况/同输出规模的正式耗时日志。", requires_manual_confirmation=True, category="J"))

    # Fortran/runtime.
    fortran_files = sorted((REPO / "fortran").glob("*")) if (REPO / "fortran").exists() else []
    add(item("K-001", "Windows/Fortran 源文件和编译产物", "FOUND" if fortran_files else "NOT_FOUND", [rel(p) for p in fortran_files], "filesystem", "fortran/", "file list", "; ".join(rel(p) for p in fortran_files), "filesystem scan", category="K"))
    bridge_hits = code_search(r"ctypes|dll|compute_prediction_metrics|average_waveform|backend_name|Fortran", ["fortran/*", "model/predict.py", "README.md", "fortran/README.md"])
    add(item("K-002", "Python/Fortran 调用接口", "FOUND" if bridge_hits else "NOT_FOUND", bridge_hits[:20] if bridge_hits else None, "source_code", "; ".join(sorted({h["path"] for h in bridge_hits[:10]})), "grep hits", "; ".join(h["text"] for h in bridge_hits[:3]), "static code search", "该证据只证明 Fortran 数值/指标组件存在，不证明 AI 主体由 Fortran 编写。", category="K"))

    # Hashes for small relevant files.
    hash_rows = []
    hash_candidates = []
    for pattern in [
        "model/**/*.py",
        "data_process/**/*.py",
        "config/**/*.py",
        "window/**/*.py",
        "fortran/*",
        "database/**/*manifest*.json",
        "database/rule/*.csv",
        "result/train/report/**/*.json",
        "result/train/report/**/*.csv",
        "result/predict/**/*.json",
        "result/predict/**/*.csv",
        "README.md",
        "docs/*.md",
    ]:
        hash_candidates.extend(REPO.glob(pattern))
    seen_hash = set()
    for p in sorted(set(hash_candidates)):
        if not p.is_file() or p in seen_hash:
            continue
        seen_hash.add(p)
        try:
            size = p.stat().st_size
            if size > 20 * 1024 * 1024:
                hash_rows.append({"path": rel(p), "size_bytes": size, "sha256": "", "skipped_reason": "larger_than_20MiB"})
            else:
                hash_rows.append({"path": rel(p), "size_bytes": size, "sha256": sha256_file(p), "skipped_reason": ""})
        except Exception as exc:
            hash_rows.append({"path": rel(p), "size_bytes": "", "sha256": "", "skipped_reason": str(exc)})
    write_csv(AUDIT_ROOT / "file_hashes.csv", hash_rows, ["path", "size_bytes", "sha256", "skipped_reason"])

    # Inventory.
    inventory_md = f"""# Repository inventory

- repository: {remote}
- audit_branch: {branch}
- base_branch: {base_branch}
- base_commit: {base_commit}
- audit_generated_at: {datetime.now(timezone.utc).isoformat()}
- tracked_file_count: {len(tracked)}
- untracked_file_count: {len(untracked)}
- dirty_worktree_at_scan: {"yes" if status_short else "no"}

## Scan scope

- Included: tracked files from `git ls-files`; untracked files from `git ls-files --others --exclude-standard`; JSON/CSV/MD/Python/Fortran/config/result metadata; existing checkpoint metadata.
- Skipped or limited: `.git` object internals; generated audit directory; file content hashing skipped for files larger than 20 MiB; no training, no inference, no data generation.

## Git status at scan

```text
{status_short}
```

## Recent Git history

```text
{git_log}
```
"""
    (AUDIT_ROOT / "00_repository_inventory.md").write_text(inventory_md, encoding="utf-8")

    # Section files.
    def write_section(filename: str, title: str, prefixes: list[str], extra: str = "") -> None:
        rows = [
            [
                it["item_id"],
                it["item_name"],
                it["status"],
                json.dumps(it["extracted_value"], ensure_ascii=False) if it["extracted_value"] is not None else "",
                it["source_path"],
                it["source_location"],
                it["notes"],
            ]
            for it in items
            if any(it["item_id"].startswith(p) for p in prefixes)
        ]
        body = f"# {title}\n\n"
        if extra:
            body += extra + "\n\n"
        body += table(["item_id", "item_name", "status", "extracted_value", "source_path", "source_location", "notes"], rows)
        body += "\n"
        (AUDIT_ROOT / filename).write_text(body, encoding="utf-8")

    write_section("01_model_input_output.md", "AI 模型实际输入和输出", ["A-"])
    write_section("02_simulation_training_sets.md", "三种材料的 AI 仿真训练集", ["B-"], "详细统计见 `02_simulation_training_sets.csv`。")
    write_section("03_w30mo70_experiment_data.md", "两层 W30Mo70 实验数据", ["C-"], "逐条记录索引见 `03_w30mo70_experiment_data.csv`。材料事实由用户确认。")
    write_section("04_temperature_acoustic_database.md", "温度声学参数数据库", ["D-"], "若使用本目录生成的索引，均为 `generated_for_audit_only: true`。")
    write_section("05_prediction_results.md", "AI 四类温度场预测结果", ["E-"], "预测运行汇总见 `prediction_runs.csv`。")
    write_section("06_five_measurement_points.md", "五个测点", ["F-"])
    write_section("07_incremental_training.md", "在线增量训练", ["G-"], "主扫描不默认执行 checkpoint 参数比较；比较工具见 `scripts/compare_checkpoints.py`。")
    write_section("08_transfer_learning_and_freezing.md", "迁移学习和参数冻结", ["H-"])
    write_section("09_timing_evidence.md", "AI 与传统算法耗时证据", ["I-", "J-"])
    write_section("10_fortran_and_runtime_evidence.md", "Windows、Fortran 和跨语言调用", ["K-"])

    unresolved = [
        it
        for it in items
        if it["status"] in {"NOT_FOUND", "AMBIGUOUS", "INCONSISTENT", "INACCESSIBLE", "NOT_EXECUTED"}
    ]
    unresolved_rows = [
        [it["item_id"], it["item_name"], it["status"], it["source_path"], it["notes"]]
        for it in unresolved
    ]
    (AUDIT_ROOT / "11_missing_ambiguous_inconsistent_items.md").write_text(
        "# Missing / ambiguous / inconsistent items\n\n"
        + table(["编号", "内容", "状态", "直接证据位置", "说明"], unresolved_rows)
        + "\n",
        encoding="utf-8",
    )

    next_rows = []
    for it in unresolved:
        if it["requires_new_experiment"]:
            cls = "A. 需要补充实验"
            reason = "当前仓库没有可直接证明的数据、结果或日志。"
        elif it["requires_code_change"]:
            cls = "B. 需要修改代码"
            reason = "当前代码或输出形式未显示支持该要求。"
        else:
            cls = "C. 只需补充文件或人工确认"
            reason = "现有信息不足以唯一确认；可能需要上传外部文件或人工确认口径。"
        next_rows.append([it["item_id"], it["item_name"], it["status"], it["source_path"] or it["evidence_excerpt"], cls, reason])
    (AUDIT_ROOT / "12_next_step_classification.md").write_text(
        "# Next step classification\n\n"
        + table(["编号", "缺失内容", "当前状态", "直接证据", "分类", "原因"], next_rows)
        + "\n",
        encoding="utf-8",
    )

    # README and audit summary.
    counts = Counter(it["status"] for it in items)
    category_counts = {}
    for category in sorted(set(it["category"] for it in items if it.get("category"))):
        category_counts[category] = dict(status_counts(items, None if category == "" else None))
    summary = {
        "repository": remote,
        "audit_branch": branch,
        "base_branch": base_branch,
        "base_commit": base_commit,
        "audit_commit": None,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_for_audit_only": True,
        "scan_scope": {
            "tracked_file_count": len(tracked),
            "untracked_file_count": len(untracked),
            "manifest_count": len(manifest_summaries),
            "prediction_run_count": len(prediction_runs),
            "checkpoint_count": len(ckpt_meta),
        },
        "status_counts": dict(counts),
        "direct_evidence_count": counts["CONFIRMED"] + counts["FOUND"],
        "requires_new_experiment_count": sum(1 for it in unresolved if it["requires_new_experiment"]),
        "requires_code_change_count": sum(1 for it in unresolved if it["requires_code_change"]),
        "requires_manual_confirmation_or_file_count": sum(1 for it in unresolved if not it["requires_new_experiment"] and not it["requires_code_change"]),
        "items": items,
    }
    (AUDIT_ROOT / "audit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(AUDIT_ROOT / "evidence_index.csv", evidence_rows)
    write_csv(AUDIT_ROOT / "manifest_summary.csv", [{k: v for k, v in s.items() if k != "records"} for s in manifest_summaries])

    readme_text = f"""# AI test evidence audit

This directory contains a static audit of existing evidence related to AI-based temperature-field inversion test reporting.

- Generated for audit only: `true`
- Repository: {remote}
- Branch at scan time: `{branch}`
- Base commit: `{base_commit}`
- Generated at: {summary["generated_at"]}

The audit did not run training, inference, data generation, or destructive Git commands. Existing files outside this directory were not modified by the audit scripts.

Primary entry point: `FINAL_AUDIT_REPORT.md`.
"""
    (AUDIT_ROOT / "README.md").write_text(readme_text, encoding="utf-8")

    # Final report.
    overview_rows = []
    for label, prefixes in [
        ("A. AI 模型输入与输出", ["A-"]),
        ("B. 三种材料仿真训练集", ["B-"]),
        ("C. W30Mo70 实验数据", ["C-"]),
        ("D. 温度声学参数数据库", ["D-"]),
        ("E. 四类 AI 预测结果", ["E-"]),
        ("F. 五个测点", ["F-"]),
        ("G. 增量训练", ["G-"]),
        ("H. 迁移学习和冻结", ["H-"]),
        ("I/J. 效率证据", ["I-", "J-"]),
        ("K. Windows/Fortran", ["K-"]),
    ]:
        sub = [it for it in items if any(it["item_id"].startswith(p) for p in prefixes)]
        c = Counter(it["status"] for it in sub)
        overview_rows.append([label, c["CONFIRMED"] + c["FOUND"], c["NOT_FOUND"], c["AMBIGUOUS"], c["INCONSISTENT"], c["INACCESSIBLE"]])

    usable = [
        it
        for it in items
        if it["status"] in {"CONFIRMED", "FOUND"}
        and it["item_id"] in {"FACT-001", "A-001", "A-002", "A-003", "A-004", "A-005", "A-006", "A-013", "C-001", "C-020", "D-100", "I-001", "K-001", "K-002"}
    ]
    usable_rows = [
        [
            it["item_id"],
            f"可写入：{it['item_name']} = {json.dumps(it['extracted_value'], ensure_ascii=False)}",
            it["source_path"],
            it["source_location"],
            "仅限按证据原样引用；不得扩展为未证明指标。",
        ]
        for it in usable
    ]
    file_index_rows = []
    for p in sorted(AUDIT_ROOT.rglob("*")):
        if p.is_file():
            file_index_rows.append([rel(p), "audit output" if "scripts" not in p.parts else "audit script"])

    final = f"""# AI 温度场测试证据审计总报告

## 1. 审计基本信息

- 仓库地址：{remote}
- 审计分支：{branch}
- 基准分支：{base_branch}
- 基准 commit：{base_commit}
- 审计 commit：待提交后由 Git 记录；`audit_summary.json` 当前为生成时快照。
- 审计时间：{summary["generated_at"]}
- 扫描范围：tracked files、untracked project files、manifest、CSV/JSON/MD/Python/Fortran、已有训练/预测报告、已有 checkpoint 元信息、Git log。
- 未扫描范围及原因：`.git` object internals 未展开；大文件内容未全文哈希；未运行训练、推理、数据生成；未默认执行 checkpoint 参数比较。

## 2. 已确认事实

- 实验测试材料为两层 W30Mo70 钨钼合金多层材料。
  - source_type: user_confirmed
  - status: CONFIRMED

## 3. 现有证据总览

{table(["检查类别", "CONFIRMED/FOUND", "NOT_FOUND", "AMBIGUOUS", "INCONSISTENT", "INACCESSIBLE"], overview_rows)}

## 4. AI 模型输入与输出

详见 `01_model_input_output.md`。关键点：模型类、训练入口、推理入口、GUI 入口、forward 输入和输出均按源代码行证据列出；材料参数、空间坐标、时间坐标、dimension/mode/material 作为模型输入未找到直接证据。

## 5. 三种材料仿真训练集

详见 `02_simulation_training_sets.md` 和 `02_simulation_training_sets.csv`。统计来自现有 manifest 记录；配置中的温度范围未作为实际覆盖证据。

## 6. W30Mo70 实验数据

详见 `03_w30mo70_experiment_data.md` 和 `03_w30mo70_experiment_data.csv`。材料身份采用用户确认事实；manifest 和记录只作为数据路径、温度、维度、模式和来源字段证据。

## 7. 温度声学参数数据库

详见 `04_temperature_acoustic_database.md`。本审计生成的 `manifest_summary.csv`、`prediction_runs.csv`、`checkpoint_inventory.csv` 均为审计索引：

```yaml
generated_for_audit_only: true
```

## 8. 四类 AI 预测结果

详见 `05_prediction_results.md` 和 `prediction_runs.csv`。报告区分代码支持、已有预测结果、已有真值、已有误差字段；未将单点或示例结果扩展为完整温度场结论。

## 9. 五个测点

详见 `06_five_measurement_points.md`。未找到明确五个测点坐标、参考温度、预测温度和五点平均相对误差的完整证据。

## 10. 增量训练与迁移学习

详见 `07_incremental_training.md` 和 `08_transfer_learning_and_freezing.md`。已区分增量训练入口、增量 checkpoint/报告存在性、以及参数实际变化证据。主扫描不默认执行 checkpoint 比较。

## 11. 效率证据

详见 `09_timing_evidence.md`。AI metrics 中存在 `eval_total_seconds`/`eval_latency_ms_per_sample` 等字段，但 benchmark 多处为 false；未找到能证明与传统算法同输入、同工况、同输出规模的正式比较口径。未给出正式 speedup。

## 12. Windows、Fortran 与跨语言调用

详见 `10_fortran_and_runtime_evidence.md`。Fortran 源文件、DLL 和 Python bridge 存在；该证据不表示 AI 主体由 Fortran 编写。

## 13. 仍需补充的事项

### 13.1 需要补充实验

{table(["编号", "缺失内容", "当前状态", "已搜索位置", "为什么证据不足"], [[it["item_id"], it["item_name"], it["status"], it["source_path"], it["notes"]] for it in unresolved if it["requires_new_experiment"]])}

### 13.2 需要修改代码

{table(["编号", "缺失内容", "当前状态", "已搜索位置", "为什么证据不足"], [[it["item_id"], it["item_name"], it["status"], it["source_path"], it["notes"]] for it in unresolved if it["requires_code_change"]])}

### 13.3 只需补充文件

{table(["编号", "缺失内容", "当前状态", "已搜索位置", "为什么证据不足"], [[it["item_id"], it["item_name"], it["status"], it["source_path"], it["notes"]] for it in unresolved if not it["requires_new_experiment"] and not it["requires_code_change"] and "上传" in it["notes"]])}

### 13.4 只需人工确认

{table(["编号", "缺失内容", "当前状态", "已搜索位置", "为什么证据不足"], [[it["item_id"], it["item_name"], it["status"], it["source_path"], it["notes"]] for it in unresolved if not it["requires_new_experiment"] and not it["requires_code_change"]])}

## 14. 可直接用于测试报告的内容

{table(["编号", "推荐报告表述", "原始证据路径", "证据位置", "使用限制"], usable_rows)}

## 15. 文件索引

{table(["文件", "用途"], file_index_rows)}
"""
    (AUDIT_ROOT / "FINAL_AUDIT_REPORT.md").write_text(final, encoding="utf-8")
    print(AUDIT_ROOT)
    print(json.dumps({"status_counts": dict(counts), "direct_evidence_count": summary["direct_evidence_count"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
