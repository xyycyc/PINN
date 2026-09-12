"""Tkinter GUI; window creation and training dependencies load on demand."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "AiModelApp": (".app", "AiModelApp"),
    "launch": (".app", "launch"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
