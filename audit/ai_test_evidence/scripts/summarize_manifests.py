from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any


AUDIT_ROOT = Path(__file__).resolve().parents[1]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8", errors="replace"))


def records_from(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, dict) and isinstance(payload.get("records"), list):
        return [r for r in payload["records"] if isinstance(r, dict)]
    return []


def summarize_manifest(path: Path, repo: Path) -> dict[str, Any]:
    try:
        payload = load_json(path)
    except Exception as exc:
        return {"path": str(path.relative_to(repo)), "status": "UNREADABLE", "error": str(exc)}
    records = records_from(payload)
    temps: list[float] = []
    invalid_wave = 0
    invalid_field = 0
    keys: set[str] = set()
    sources: Counter[str] = Counter()
    materials: Counter[str] = Counter()
    dimensions: Counter[str] = Counter()
    modes: Counter[str] = Counter()
    sample_ids: list[str] = []
    for record in records:
        keys.update(record.keys())
        sample_ids.append(str(record.get("sample_id", "")))
        for key, counter in (
            ("source", sources),
            ("material_key", materials),
            ("dimension", dimensions),
            ("mode", modes),
        ):
            counter[str(record.get(key, ""))] += 1
        try:
            temps.append(float(record["temperature_k"]))
        except Exception:
            pass
        for key, label in (("waveform_path", "wave"), ("field_path", "field")):
            raw = record.get(key)
            if not raw:
                if label == "wave":
                    invalid_wave += 1
                else:
                    invalid_field += 1
                continue
            candidate = Path(str(raw))
            if not candidate.is_absolute():
                candidate = path.parent / candidate
            if not candidate.exists():
                if label == "wave":
                    invalid_wave += 1
                else:
                    invalid_field += 1
    dup_count = sum(v - 1 for v in Counter(sample_ids).values() if v > 1)
    return {
        "path": str(path.relative_to(repo)),
        "status": "READ",
        "dataset_name": payload.get("dataset_name") if isinstance(payload, dict) else None,
        "dataset_label": payload.get("dataset_label") if isinstance(payload, dict) else None,
        "schema_version": payload.get("schema_version") if isinstance(payload, dict) else None,
        "record_count": len(records),
        "temperature_min_k": min(temps) if temps else None,
        "temperature_max_k": max(temps) if temps else None,
        "near_300k_records": sum(1 for t in temps if 295.0 <= t <= 305.0),
        "records_at_1500k": sum(1 for t in temps if abs(t - 1500.0) <= 1e-6),
        "available_fields": "|".join(sorted(keys)),
        "sources": json.dumps(dict(sources), ensure_ascii=False),
        "materials": json.dumps(dict(materials), ensure_ascii=False),
        "dimensions": json.dumps(dict(dimensions), ensure_ascii=False),
        "modes": json.dumps(dict(modes), ensure_ascii=False),
        "duplicate_sample_id_count": dup_count,
        "invalid_waveform_path_count": invalid_wave,
        "invalid_field_path_count": invalid_field,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize existing manifest JSON files.")
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--output", default=str(AUDIT_ROOT / "manifest_summary.csv"))
    args = parser.parse_args()
    repo = Path(args.repo).resolve()
    output = Path(args.output)
    if not output.resolve().is_relative_to(AUDIT_ROOT.resolve()):
        raise ValueError("output must be under audit/ai_test_evidence")
    manifest_paths = [
        path
        for path in repo.rglob("*.json")
        if "manifest" in path.name.lower()
        and ".git" not in path.parts
        and "audit" not in path.parts
    ]
    rows = [summarize_manifest(path, repo) for path in sorted(manifest_paths)]
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
