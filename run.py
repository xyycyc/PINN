"""Run from the checkout root: ``python run.py --help`` or ``--gui``."""

from __future__ import annotations

import sys
from pathlib import Path

# File execution must make the containing package importable after moving drives.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> None:
    if sys.argv[1:] == ["--gui"]:
        from ai_model.window import launch

        launch()
    else:
        from ai_model.cli import main as cli_main

        cli_main()


if __name__ == "__main__":
    main()
