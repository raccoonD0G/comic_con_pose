"""Path helpers used by both CLI and GUI entry points."""
from __future__ import annotations

import os
import sys


def resource_path(relative_path: str) -> str:
    """Return an absolute path that works for PyInstaller bundles."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)  # type: ignore[attr-defined]
    return os.path.join(os.path.abspath("."), relative_path)
