#!/usr/bin/env python3
"""Record RGB, stereo depth, and IMU from a connected OAK-D Pro W.

Example:
    python record.py --out runs/slide1 --table-height 0.75 --optical-height 0.03
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
    raise SystemExit(main(["record", *sys.argv[1:]]))
