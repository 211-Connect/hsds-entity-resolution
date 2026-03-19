"""Pytest bootstrap for project-local import paths."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONSUMER = ROOT / "consumer"
for path in (ROOT, CONSUMER):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)
