"""Import shim: make the un-packaged ``sim/`` modules importable from ``policy/``.

``sim/`` modules import each other by bare name (``from config import Config``),
so the only way to use them unchanged is to put ``<repo>/sim`` on ``sys.path``.
Every module in ``policy/`` starts with ``import _paths  # noqa: F401`` before
any ``sim`` import.

Invariants:
- ``SIM_DIR`` is *appended*, never inserted, so ``policy/`` (the script
  directory) stays first and its modules shadow nothing in ``sim/``.
- No module in ``policy/`` may share a name with one in ``sim/`` (``config,
  scene, render, physics, geometry, layouts, reachability, generate, sketch``);
  hence ``policyconfig.py`` and ``nets.py``. DESIGN.md §8.1 and §11.3.
- Spawned worker processes re-import this module, so the path is set there too.
"""
from __future__ import annotations

import sys
from pathlib import Path

POLICY_DIR = Path(__file__).resolve().parent
REPO = POLICY_DIR.parent
SIM_DIR = REPO / "sim"

if str(POLICY_DIR) not in sys.path:
    sys.path.insert(0, str(POLICY_DIR))
if str(SIM_DIR) not in sys.path:
    sys.path.append(str(SIM_DIR))
