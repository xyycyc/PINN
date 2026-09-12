"""Resolve project paths consistently, independent of the process directory."""

from __future__ import annotations

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def resolve_project_path(
    path_value: str | Path, *, repo_root: str | Path = PACKAGE_ROOT
) -> Path:
    """Accept absolute, package-relative and legacy package-prefixed paths."""
    root = Path(repo_root).expanduser().resolve()
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        base = (
            root.parent
            if path.parts and path.parts[0].casefold() == root.name.casefold()
            else root
        )
        path = base / path
    return path.resolve()
