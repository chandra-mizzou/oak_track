"""Run-folder layout, calibration, and the 2-column CSV the experiment asked for."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np


def format_xyz(xyz: Sequence[float], digits: int = 6) -> str:
    x, y, z = (float(v) for v in xyz)
    return f"({x:.{digits}f}, {y:.{digits}f}, {z:.{digits}f})"


def parse_xyz(text: str) -> np.ndarray:
    s = text.strip().strip("()")
    parts = [p.strip() for p in s.split(",")]
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float64)


def write_frame_xyz_csv(path: Path, frame_numbers: Iterable[int], xyz: np.ndarray) -> None:
    """Two columns: frame number and (x,y,z) in metres."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_number", "xyz"])
        for i, p in zip(frame_numbers, xyz):
            writer.writerow([int(i), format_xyz(p)])


def read_frame_xyz_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    frames = []
    xyz = []
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            frames.append(int(row["frame_number"]))
            xyz.append(parse_xyz(row["xyz"]))
    return np.asarray(frames), np.asarray(xyz)


def write_imu_raw_csv(
    path: Path,
    t: np.ndarray,
    accel: np.ndarray,
    gyro: np.ndarray,
    quat: Optional[np.ndarray] = None,
    linear_accel: Optional[np.ndarray] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["t", "ax", "ay", "az", "gx", "gy", "gz"]
    if quat is not None:
        header += ["qw", "qx", "qy", "qz"]
    if linear_accel is not None:
        header += ["lax", "lay", "laz"]
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        n = len(t)
        for i in range(n):
            row = [f"{t[i]:.9f}", *accel[i].tolist(), *gyro[i].tolist()]
            if quat is not None:
                row += quat[i].tolist()
            if linear_accel is not None:
                row += linear_accel[i].tolist()
            writer.writerow(row)


def read_imu_raw_csv(path: Path) -> dict[str, np.ndarray]:
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    keys = rows[0].keys() if rows else []

    def col(*names: str) -> np.ndarray:
        return np.array([[float(r[n]) for n in names] for r in rows], dtype=np.float64)

    out: dict[str, np.ndarray] = {
        "t": np.array([float(r["t"]) for r in rows], dtype=np.float64),
        "accel": col("ax", "ay", "az"),
        "gyro": col("gx", "gy", "gz"),
    }
    if "qw" in keys:
        out["quat"] = col("qw", "qx", "qy", "qz")
    if "lax" in keys:
        out["linear_accel"] = col("lax", "lay", "laz")
    return out


def write_frames_csv(path: Path, frame_idx: Sequence[int], t_color: Sequence[float], t_depth: Optional[Sequence[float]] = None) -> None:
    path = Path(path)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_number", "t_color", "t_depth"])
        for i, tc in enumerate(t_color):
            td = t_depth[i] if t_depth is not None else tc
            writer.writerow([int(frame_idx[i]), f"{tc:.9f}", f"{td:.9f}"])


def read_frames_csv(path: Path) -> dict[str, np.ndarray]:
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return {
        "frame_number": np.array([int(r["frame_number"]) for r in rows], dtype=int),
        "t_color": np.array([float(r["t_color"]) for r in rows], dtype=np.float64),
        "t_depth": np.array([float(r["t_depth"]) for r in rows], dtype=np.float64),
    }


@dataclass
class Calibration:
    K: list[list[float]]
    dist: list[float]
    width: int
    height: int
    baseline_m: float = 0.075
    imu_to_cam: list[list[float]] | None = None
    K_left: list[list[float]] | None = None
    inverted: bool = False
    undistorted: bool = False
    K_raw: list[list[float]] | None = None
    dist_raw: list[float] | None = None

    def K_np(self) -> np.ndarray:
        return np.asarray(self.K, dtype=np.float64)

    def dist_np(self) -> np.ndarray:
        return np.asarray(self.dist, dtype=np.float64)


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2))


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text())


def save_calibration(path: Path, calib: Calibration) -> None:
    write_json(path, asdict(calib))


def load_calibration(path: Path) -> Calibration:
    d = read_json(path)
    allowed = {f.name for f in fields(Calibration)}
    return Calibration(**{k: v for k, v in d.items() if k in allowed})


def run_paths(run_dir: Path) -> dict[str, Path]:
    run_dir = Path(run_dir)
    return {
        "root": run_dir,
        "color": run_dir / "color.mp4",
        "color_dir": run_dir / "color_frames",
        "depth_dir": run_dir / "depth",
        "imu": run_dir / "imu.csv",
        "frames": run_dir / "frames.csv",
        "calibration": run_dir / "calibration.json",
        "meta": run_dir / "meta.json",
        "camera_imu": run_dir / "camera_imu.csv",
        "camera_vo": run_dir / "camera_vo.csv",
        "object": run_dir / "object.csv",
        "object_fused": run_dir / "object_fused.csv",
        "detections": run_dir / "detections.csv",
        "preview": run_dir / "preview.mp4",
        "ground_truth": run_dir / "ground_truth.json",
        "cloud": run_dir / "cloud.ply",
        "cloud_meta": run_dir / "cloud.json",
        "cloud_preview": run_dir / "cloud_preview.png",
    }
