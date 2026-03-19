"""Test package bootstrap helpers."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_CONSUMER = _ROOT / "consumer"
for _path in (_ROOT, _CONSUMER):
    text_path = str(_path)
    if text_path not in sys.path:
        sys.path.insert(0, text_path)
