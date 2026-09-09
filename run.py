#!/usr/bin/env python3
"""Record a table slide and process it in one step.

Logs video, per-frame timestamps, and IMU while you slide the camera, then
writes camera_imu.csv and object.csv in the same folder.

Example:
    python run.py --table-height 0.75 --optical-height 0.03
    python run.py --out runs/slide1 --table-height 0.75 --duration 8
    python run.py --simulate --out runs/sim --table-height 0.75
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
    raise SystemExit(main(["run", *sys.argv[1:]]))
