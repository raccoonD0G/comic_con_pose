"""Application entry point with CLI and GUI front-ends."""
from __future__ import annotations

import sys

from app.args import parse_args
from pipeline import run_pipeline
from ui.control_panel import ControlPanelApp


def main() -> None:
    if len(sys.argv) > 1:
        args = parse_args()
        run_pipeline(args)
        return

    try:
        app = ControlPanelApp()
    except RuntimeError as exc:
        print(f"[WARN] {exc}. Falling back to CLI mode.", file=sys.stderr)
        args = parse_args()
        run_pipeline(args)
        return

    app.run()


if __name__ == "__main__":
    main()
