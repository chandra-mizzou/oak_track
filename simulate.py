#!/usr/bin/env python3
"""Write a synthetic table-slide run (no OAK-D required).

Example:
    python simulate.py --out runs/sim --table-height 0.75 --object-x 0.05 --object-y 0.90
    python process.py --run runs/sim --table-height 0.75
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
    raise SystemExit(main(["simulate", *sys.argv[1:]]))
