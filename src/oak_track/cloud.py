"""Dense RGB-D point cloud from stored frames, depth maps, and camera poses.

Generation uses numpy + OpenCV only (already in requirements.txt). The output is
a binary PLY that MeshLab, CloudCompare, and Open3D can open. Open3D is optional
and only needed if you want an interactive viewer:

    python -m pip install open3d
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from oak_track.geometry import Pose, transform_points
from oak_track.io_utils import write_json


def depth_to_meters(depth: np.ndarray) -> np.ndarray:
    """OAK depth PNGs are 16-bit millimetres; keep float metres as-is."""
    if depth.dtype == np.uint16 or depth.max() > 20.0:
        return depth.astype(np.float32) * 0.001
    return depth.astype(np.float32)


def _pack_voxel_keys(keys: np.ndarray) -> np.ndarray:
    """Pack int (N,3) voxel indices into uint64. 21 bits/axis, origin offset 2^20."""
    off = np.int64(1 << 20)
    x = keys[:, 0].astype(np.int64) + off
    y = keys[:, 1].astype(np.int64) + off
    z = keys[:, 2].astype(np.int64) + off
    if np.any((x < 0) | (y < 0) | (z < 0) | (x >= (1 << 21)) | (y >= (1 << 21)) | (z >= (1 << 21))):
        raise ValueError("Point cloud extent is too large for 1 cm-class voxel packing")
    return (x | (y << 21) | (z << 42)).astype(np.uint64)


def voxel_downsample(xyz: np.ndarray, rgb: np.ndarray, voxel_m: float) -> tuple[np.ndarray, np.ndarray]:
    """Mean position and colour per voxel. ``voxel_m <= 0`` keeps every point."""
    xyz = np.asarray(xyz, dtype=np.float64)
    rgb = np.asarray(rgb, dtype=np.float64)
    if len(xyz) == 0 or voxel_m <= 0:
        return xyz, np.clip(np.round(rgb), 0, 255).astype(np.uint8)
    keys = np.floor(xyz / float(voxel_m)).astype(np.int64)
    packed = _pack_voxel_keys(keys)
    uniq, inv, counts = np.unique(packed, return_inverse=True, return_counts=True)
    n = len(uniq)
    xyz_acc = np.zeros((n, 3), dtype=np.float64)
    rgb_acc = np.zeros((n, 3), dtype=np.float64)
    np.add.at(xyz_acc, inv, xyz)
    np.add.at(rgb_acc, inv, rgb)
    counts = counts.astype(np.float64)[:, None]
    xyz_acc /= counts
    rgb_acc /= counts
    return xyz_acc, np.clip(np.round(rgb_acc), 0, 255).astype(np.uint8)


class _VoxelAccum:
    """Running mean per voxel so multi-frame fusion stays bounded in memory."""

    def __init__(self, voxel_m: float) -> None:
        self.voxel_m = float(voxel_m)
        self.xyz_sum: dict[int, np.ndarray] = {}
        self.rgb_sum: dict[int, np.ndarray] = {}
        self.count: dict[int, int] = {}

    def add(self, xyz: np.ndarray, rgb: np.ndarray) -> None:
        if len(xyz) == 0:
            return
        if self.voxel_m <= 0:
            # Keep unique keys by point index so finalize still works.
            start = len(self.count)
            for i in range(len(xyz)):
                k = start + i
                self.xyz_sum[k] = xyz[i].astype(np.float64)
                self.rgb_sum[k] = rgb[i].astype(np.float64)
                self.count[k] = 1
            return
        keys = np.floor(xyz / self.voxel_m).astype(np.int64)
        packed = _pack_voxel_keys(keys)
        uniq, inv, counts = np.unique(packed, return_inverse=True, return_counts=True)
        xyz_acc = np.zeros((len(uniq), 3), dtype=np.float64)
        rgb_acc = np.zeros((len(uniq), 3), dtype=np.float64)
        np.add.at(xyz_acc, inv, xyz.astype(np.float64))
        np.add.at(rgb_acc, inv, rgb.astype(np.float64))
        for i, k in enumerate(uniq.tolist()):
            ki = int(k)
            n = int(counts[i])
            if ki in self.count:
                self.xyz_sum[ki] += xyz_acc[i]
                self.rgb_sum[ki] += rgb_acc[i]
                self.count[ki] += n
            else:
                self.xyz_sum[ki] = xyz_acc[i]
                self.rgb_sum[ki] = rgb_acc[i]
                self.count[ki] = n

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if not self.count:
            return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
        keys = list(self.count.keys())
        n = len(keys)
        xyz = np.empty((n, 3), dtype=np.float64)
        rgb = np.empty((n, 3), dtype=np.float64)
        for i, k in enumerate(keys):
            c = float(self.count[k])
            xyz[i] = self.xyz_sum[k] / c
            rgb[i] = self.rgb_sum[k] / c
        return xyz, np.clip(np.round(rgb), 0, 255).astype(np.uint8)

    def __len__(self) -> int:
        return len(self.count)


def backproject_rgbd(
    color_bgr: np.ndarray,
    depth: np.ndarray,
    K: np.ndarray,
    pose: Pose,
    pixel_stride: int = 2,
    depth_min_m: float = 0.3,
    depth_max_m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Camera-frame RGB-D → world XYZ + RGB. Empty arrays if nothing is valid."""
    if color_bgr is None or depth is None:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    if depth.shape[:2] != color_bgr.shape[:2]:
        depth = cv2.resize(depth, (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    z = depth_to_meters(depth)
    h, w = z.shape[:2]
    stride = max(int(pixel_stride), 1)
    vs = np.arange(0, h, stride, dtype=np.int32)
    us = np.arange(0, w, stride, dtype=np.int32)
    uu, vv = np.meshgrid(us, vs)
    zz = z[vv, uu]
    valid = np.isfinite(zz) & (zz >= depth_min_m) & (zz <= depth_max_m)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0, 3), dtype=np.uint8)
    ui = uu[valid]
    vi = vv[valid]
    zz = zz[valid].astype(np.float64)
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    x = (ui.astype(np.float64) - cx) / fx * zz
    y = (vi.astype(np.float64) - cy) / fy * zz
    pts_cam = np.stack([x, y, zz], axis=1)
    pts_w = transform_points(pose.R, pose.t, pts_cam)
    bgr = color_bgr[vi, ui]
    rgb = bgr[:, ::-1].copy()
    return pts_w, rgb


def write_ply_xyzrgb(path: Path, xyz: np.ndarray, rgb: np.ndarray) -> None:
    """Binary little-endian PLY with float XYZ and uchar RGB."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(xyz, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb, dtype=np.uint8).reshape(-1, 3)
    n = int(xyz.shape[0])
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment oak_track dense RGB-D fusion\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    verts = np.empty(n, dtype=dt)
    if n:
        verts["x"] = xyz[:, 0]
        verts["y"] = xyz[:, 1]
        verts["z"] = xyz[:, 2]
        verts["r"] = rgb[:, 0]
        verts["g"] = rgb[:, 1]
        verts["b"] = rgb[:, 2]
    with path.open("wb") as f:
        f.write(header.encode("ascii"))
        verts.tofile(f)


def read_ply_xyzrgb(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a PLY written by :func:`write_ply_xyzrgb` (tests / sanity checks)."""
    raw = Path(path).read_bytes()
    marker = b"end_header\n"
    idx = raw.find(marker)
    if idx < 0:
        raise ValueError(f"No PLY header in {path}")
    header = raw[: idx + len(marker)].decode("ascii")
    n = 0
    for line in header.splitlines():
        if line.startswith("element vertex"):
            n = int(line.split()[-1])
    body = raw[idx + len(marker) :]
    dt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    verts = np.frombuffer(body, dtype=dt, count=n)
    xyz = np.stack([verts["x"], verts["y"], verts["z"]], axis=1).astype(np.float64)
    rgb = np.stack([verts["r"], verts["g"], verts["b"]], axis=1).astype(np.uint8)
    return xyz, rgb


def write_cloud_preview_xy(path: Path, xyz: np.ndarray, rgb: np.ndarray, width: int = 960) -> None:
    """Top-down XY scatter (table plane) as a PNG, no extra libraries."""
    path = Path(path)
    if len(xyz) == 0:
        cv2.imwrite(str(path), np.zeros((480, width, 3), dtype=np.uint8))
        return
    x, y = xyz[:, 0], xyz[:, 1]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    span_x = max(xmax - xmin, 1e-3)
    span_y = max(ymax - ymin, 1e-3)
    height = max(int(round(width * span_y / span_x)), 64)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    px = np.clip(((x - xmin) / span_x * (width - 1)).astype(np.int32), 0, width - 1)
    py = np.clip(((ymax - y) / span_y * (height - 1)).astype(np.int32), 0, height - 1)
    img[py, px] = rgb[:, ::-1]
    cv2.imwrite(str(path), img)


def fuse_dense_cloud(
    color_frames: Sequence[np.ndarray],
    depths: Sequence[Optional[np.ndarray]],
    poses: Sequence[Pose],
    K: np.ndarray,
    pixel_stride: int = 2,
    frame_stride: int = 1,
    voxel_m: float = 0.01,
    depth_min_m: float = 0.3,
    depth_max_m: float = 3.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fuse RGB-D frames into a voxel-downsampled world cloud."""
    n = min(len(color_frames), len(depths), len(poses))
    step = max(int(frame_stride), 1)
    accum = _VoxelAccum(voxel_m)
    n_used = 0
    n_raw = 0
    for i in range(0, n, step):
        depth = depths[i]
        if depth is None:
            continue
        xyz, rgb = backproject_rgbd(
            color_frames[i],
            depth,
            K,
            poses[i],
            pixel_stride=pixel_stride,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
        )
        if len(xyz) == 0:
            continue
        n_used += 1
        n_raw += len(xyz)
        accum.add(xyz, rgb)
    xyz, rgb = accum.finalize()
    meta = {
        "n_points": int(len(xyz)),
        "n_raw_points": int(n_raw),
        "n_frames_used": int(n_used),
        "pixel_stride": int(pixel_stride),
        "frame_stride": int(step),
        "voxel_m": float(voxel_m),
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
    }
    return xyz, rgb, meta


def write_run_cloud(
    run_dir: Path,
    color_frames: Sequence[np.ndarray],
    depths: Sequence[Optional[np.ndarray]],
    poses: Sequence[Pose],
    K: np.ndarray,
    pixel_stride: int = 2,
    frame_stride: int = 1,
    voxel_m: float = 0.01,
    depth_min_m: float = 0.3,
    depth_max_m: float = 3.0,
) -> Optional[dict]:
    """Write ``cloud.ply``, ``cloud.json``, and ``cloud_preview.png`` into the run folder."""
    if not any(d is not None for d in depths):
        return None
    xyz, rgb, meta = fuse_dense_cloud(
        color_frames,
        depths,
        poses,
        K,
        pixel_stride=pixel_stride,
        frame_stride=frame_stride,
        voxel_m=voxel_m,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    run_dir = Path(run_dir)
    ply = run_dir / "cloud.ply"
    write_ply_xyzrgb(ply, xyz, rgb)
    write_cloud_preview_xy(run_dir / "cloud_preview.png", xyz, rgb)
    meta["cloud_ply"] = str(ply)
    meta["cloud_preview"] = str(run_dir / "cloud_preview.png")
    write_json(run_dir / "cloud.json", meta)
    return meta
