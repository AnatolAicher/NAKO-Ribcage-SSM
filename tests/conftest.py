"""Shared test setup: ensure ``src/`` is importable when running pytest from
the repo root without ``pip install -e .``."""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
