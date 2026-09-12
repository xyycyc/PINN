"""Lightweight CLI entrypoint; training dependencies load only for execution."""

from __future__ import annotations

from collections.abc import Sequence

from .cli_arguments import _build_parser, _reject_legacy_noop_weights


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _reject_legacy_noop_weights(args)
    from .commands import run_command

    run_command(args)


def __getattr__(name: str):
    """Keep historical ``ai_model.cli`` imports working after the split."""
    if name.startswith("__"):
        raise AttributeError(name)
    from importlib import import_module

    for module_name in (".cli_arguments", ".commands"):
        module = import_module(module_name, __package__)
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
