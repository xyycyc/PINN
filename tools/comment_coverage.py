"""Audit module, class, and function documentation coverage for project code."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


SOURCE_DIRECTORIES = (
    "batch",
    "config",
    "data_process",
    "fortran",
    "model",
    "tools",
    "window",
)
EXTRA_SOURCE_FILES = (
    "database/raw/waveforms_by_file/build_post0_database.py",
)


def discover_source_files(root: Path) -> list[Path]:
    """Return production Python files while excluding tests and generated data."""
    files = list(root.glob("*.py"))
    for directory in SOURCE_DIRECTORIES:
        source_root = root / directory
        if source_root.is_dir():
            files.extend(
                path
                for path in source_root.rglob("*.py")
                if "__pycache__" not in path.parts
            )
    for relative_path in EXTRA_SOURCE_FILES:
        path = root / relative_path
        if path.is_file():
            files.append(path)
    return sorted(set(path.resolve() for path in files))


def documentation_counts(path: Path) -> tuple[int, int]:
    """Count documented and total module/class/function symbols in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    symbols: list[ast.AST] = [tree]
    symbols.extend(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
    )
    documented = sum(bool(ast.get_docstring(node, clean=False)) for node in symbols)
    return documented, len(symbols)


def audit(root: Path) -> tuple[int, int, list[tuple[Path, int, int]]]:
    """Aggregate documentation counts and retain per-file audit details."""
    details: list[tuple[Path, int, int]] = []
    for path in discover_source_files(root):
        documented, total = documentation_counts(path)
        details.append((path, documented, total))
    return (
        sum(item[1] for item in details),
        sum(item[2] for item in details),
        details,
    )


def parse_args() -> argparse.Namespace:
    """Parse the repository root and minimum accepted coverage."""
    parser = argparse.ArgumentParser(
        description=(
            "Measure the percentage of production modules, classes, and functions "
            "that contain Python docstrings."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project package root; defaults to the parent of tools/.",
    )
    parser.add_argument(
        "--minimum",
        type=float,
        default=0.30,
        help="Minimum accepted fraction in [0,1]; default: 0.30.",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Print per-file counts after the aggregate result.",
    )
    return parser.parse_args()


def main() -> None:
    """Run the audit and return a failing process code below the threshold."""
    args = parse_args()
    if not 0.0 <= args.minimum <= 1.0:
        raise ValueError("--minimum must be within [0,1]")
    root = args.root.resolve()
    documented, total, details = audit(root)
    coverage = documented / total if total else 1.0
    print(
        f"documented_symbols={documented} total_symbols={total} "
        f"coverage={coverage:.2%} minimum={args.minimum:.2%}"
    )
    if args.details:
        for path, file_documented, file_total in details:
            relative = path.relative_to(root)
            file_coverage = file_documented / file_total if file_total else 1.0
            print(
                f"{file_coverage:7.2%} "
                f"{file_documented:4d}/{file_total:4d} {relative}"
            )
    if coverage < args.minimum:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
