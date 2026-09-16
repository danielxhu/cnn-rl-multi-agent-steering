"""Import shim: make the un-packaged ``sim/`` modules importable from ``perception/``.

``sim/`` modules import each other by bare name (``from config import Config``),
so the only way to use them unchanged is to put ``<repo>/sim`` on ``sys.path``.
Every module in ``perception/`` starts with ``import _paths  # noqa: F401``
before any ``sim`` import.

Invariants:
- ``SIM_DIR`` is *appended*, never inserted, so ``perception/`` (the script
  directory) stays first and its modules shadow nothing in ``sim/``.
- No module in ``perception/`` may share a name with one in ``sim/``
  (``config``, ``scene``, ``render``, ``physics``, ``geometry``, ``layouts``,
  ``reachability``, ``generate``); see DESIGN.md §11.3.
"""
from __future__ import annotations

import sys
from pathlib import Path

PERCEPTION_DIR = Path(__file__).resolve().parent
REPO = PERCEPTION_DIR.parent
SIM_DIR = REPO / "sim"

if str(SIM_DIR) not in sys.path:
    sys.path.append(str(SIM_DIR))
