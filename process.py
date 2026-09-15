#!/usr/bin/env python3
"""Build camera/object CSVs and a dense cloud.ply from a recorded run.

Example:
    python process.py --run runs/slide1 --table-height 0.75 --detector hsv
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from oak_track.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["process", *sys.argv[1:]]))
