from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


AUDIT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_state(path: Path) -> dict[str, Any]:
    import torch

    bundle = torch.load(path, map_location="cpu")
    if isinstance(bundle, dict) and isinstance(bundle.get("model_state"), dict):
        return dict(bundle["model_state"])
    if isinstance(bundle, dict) and all(hasattr(v, "shape") for v in bundle.values()):
        return dict(bundle)
    raise ValueError(f"no readable model_state in checkpoint: {path}")


def tensor_abs_stats(a: Any, b: Any) -> dict[str, float | int]:
    import torch

    if a.shape != b.shape:
        return {
            "same_shape": 0,
            "parameter_count": 0,
            "changed": 1,
            "max_abs_diff": float("nan"),
            "mean_abs_diff": float("nan"),
            "norm_diff": float("nan"),
        }
    diff = (a.detach().cpu().float() - b.detach().cpu().float()).abs()
    count = int(diff.numel())
    if count == 0:
        return {
            "same_shape": 1,
            "parameter_count": 0,
            "changed": 0,
            "max_abs_diff": 0.0,
            "mean_abs_diff": 0.0,
            "norm_diff": 0.0,
        }
    max_abs = float(diff.max().item())
    mean_abs = float(diff.mean().item())
    norm_diff = float(torch.linalg.vector_norm(diff.reshape(-1)).item())
    return {
        "same_shape": 1,
        "parameter_count": count,
        "changed": int(max_abs != 0.0),
        "max_abs_diff": max_abs,
        "mean_abs_diff": mean_abs,
        "norm_diff": norm_diff,
    }


def compare(base: Path, updated: Path) -> dict[str, Any]:
    base_state = load_state(base)
    updated_state = load_state(updated)
    base_keys = set(base_state)
    updated_keys = set(updated_state)
    common = sorted(base_keys & updated_keys)
    rows: list[dict[str, Any]] = []
    total_params = 0
    same_params = 0
    changed_params = 0
    max_abs = 0.0
    weighted_abs_sum = 0.0
    norm_sq = 0.0
    for key in common:
        stats = tensor_abs_stats(base_state[key], updated_state[key])
        count = int(stats["parameter_count"])
        total_params += count
        if int(stats["same_shape"]) and not int(stats["changed"]):
            same_params += count
        else:
            changed_params += count
        if count:
            max_abs = max(max_abs, float(stats["max_abs_diff"]))
            weighted_abs_sum += float(stats["mean_abs_diff"]) * count
            norm_sq += float(stats["norm_diff"]) ** 2
        rows.append({"parameter": key, **stats})
    return {
        "base_checkpoint": str(base),
        "updated_checkpoint": str(updated),
        "base_sha256": sha256_file(base),
        "updated_sha256": sha256_file(updated),
        "base_only_keys": sorted(base_keys - updated_keys),
        "updated_only_keys": sorted(updated_keys - base_keys),
        "common_parameter_tensors": len(common),
        "parameter_total": total_params,
        "same_parameter_count": same_params,
        "changed_parameter_count": changed_params,
        "max_absolute_parameter_diff": max_abs,
        "mean_absolute_parameter_diff": weighted_abs_sum / total_params if total_params else None,
        "parameter_norm_change": norm_sq ** 0.5,
        "per_tensor": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read-only checkpoint comparison. Writes only under audit/ai_test_evidence/."
    )
    parser.add_argument("--base", required=True)
    parser.add_argument("--updated", required=True)
    parser.add_argument("--name", default="checkpoint_comparison")
    args = parser.parse_args()
    base = Path(args.base).resolve()
    updated = Path(args.updated).resolve()
    if not base.is_file() or not updated.is_file():
        raise FileNotFoundError(f"checkpoint not found: {base} or {updated}")
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    result = compare(base, updated)
    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.name)
    json_path = AUDIT_ROOT / f"{safe_name}.json"
    csv_path = AUDIT_ROOT / f"{safe_name}.csv"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "parameter",
            "same_shape",
            "parameter_count",
            "changed",
            "max_abs_diff",
            "mean_abs_diff",
            "norm_diff",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(result["per_tensor"])
    print(json_path)
    print(csv_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
