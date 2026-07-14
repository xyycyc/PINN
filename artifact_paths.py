"""Portable naming rules for artifacts written below managed project roots."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


_PORTABLE_INVALID_CHARS = frozenset('<>:"|?*')
_WINDOWS_DEVICE_NAMES = frozenset(
    {
        "con",
        "prn",
        "aux",
        "nul",
        "conin$",
        "conout$",
        *(f"com{index}" for index in range(1, 10)),
        *(f"lpt{index}" for index in range(1, 10)),
    }
)


def validate_artifact_basename(
    value: str,
    *,
    label: str,
    suffix: str | None = None,
    allow_empty: bool = False,
) -> str:
    """Return one safe path component, rejecting traversal and nested paths."""

    text = str(value or "").strip()
    if not text:
        if allow_empty:
            return ""
        raise ValueError(f"{label}不能为空")
    device_stem = text.split(".", 1)[0].casefold()
    if (
        text in {".", ".."}
        or "/" in text
        or "\\" in text
        or Path(text).is_absolute()
        or any(char in _PORTABLE_INVALID_CHARS or ord(char) < 32 for char in text)
        or text.endswith((".", " "))
        or device_stem in _WINDOWS_DEVICE_NAMES
    ):
        raise ValueError(f"{label}必须是单个文件名/目录名，不能包含路径或非法字符: {text!r}")
    if suffix is not None and Path(text).suffix.casefold() != suffix.casefold():
        raise ValueError(f"{label}必须使用 {suffix} 后缀: {text!r}")
    return text


def validate_distinct_artifact_basenames(
    values: Iterable[str],
    *,
    label: str,
    reserved: Iterable[str] = (),
) -> tuple[str, ...]:
    """Reject case-insensitive output-name collisions on every supported OS."""

    names = tuple(str(value).strip() for value in values)
    folded = tuple(name.casefold() for name in names)
    if len(set(folded)) != len(folded):
        raise ValueError(f"{label}必须互不相同")
    reserved_names = {str(value).strip().casefold() for value in reserved}
    collisions = [name for name in names if name.casefold() in reserved_names]
    if collisions:
        raise ValueError(
            f"{label}不能覆盖保留文件: {', '.join(collisions)}"
        )
    return names


def normalize_pt_filename(
    value: str,
    *,
    label: str,
    allow_empty: bool = False,
) -> str:
    """Validate one filename and append ``.pt`` when no suffix was supplied."""

    text = validate_artifact_basename(value, label=label, allow_empty=allow_empty)
    if not text:
        return ""
    if not Path(text).suffix:
        text = f"{text}.pt"
    if Path(text).suffix.casefold() != ".pt":
        raise ValueError(f"{label}必须使用 .pt 后缀: {text!r}")
    return text
