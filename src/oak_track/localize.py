"""Precise object (x, y, z) on the table from multi-view stereo + slide prior.

IMU translation is too noisy to be the primary pose for Part B. This module:

1. Back-projects each detection with stereo depth into the world.
2. Robustly averages those points onto the table plane z = h.
3. Jointly refines camera slide X(t) and a static object (X, Y, h) with
   reprojection + depth residuals (the slide itself is a large baseline).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from oak_track.geometry import Pose, backproject, project_point, world_from_cam_point


@dataclass
class ObjectEstimate:
    xyz: np.ndarray  # fused (x, y, h)
    per_frame: np.ndarray  # (N, 3), nan where missing
    rms_reproj_px: float
    n_used: int


def detections_to_world(
    poses: list[Pose],
    uvs: np.ndarray,
    depths: np.ndarray,
    K: np.ndarray,
    table_height: float,
) -> np.ndarray:
    """Per-frame object points in world; z forced to table height when finite."""
    n = len(poses)
    out = np.full((n, 3), np.nan)
    for i in range(n):
        if not np.isfinite(uvs[i]).all():
            continue
        if not np.isfinite(depths[i]) or depths[i] <= 0.05:
            continue
        p_cam = backproject(K, uvs[i], float(depths[i]))
        p_w = world_from_cam_point(poses[i], p_cam)
        p_w[2] = table_height
        out[i] = p_w
    return out


def robust_median_xyz(points: np.ndarray) -> Optional[np.ndarray]:
    pts = points[np.isfinite(points).all(axis=1)]
    if len(pts) == 0:
        return None
    med = np.median(pts, axis=0)
    d = np.linalg.norm(pts - med, axis=1)
    keep = pts[d <= max(np.median(d) * 3.0, 0.04)]
    if len(keep) == 0:
        keep = pts
    return np.median(keep, axis=0)


def _huber(r: np.ndarray, k: float) -> np.ndarray:
    r = np.nan_to_num(np.asarray(r, dtype=np.float64), nan=0.0, posinf=25.0, neginf=-25.0)
    a = np.abs(r)
    out = np.empty_like(r)
    small = a <= k
    out[small] = r[small]
    large = ~small
    out[large] = np.sqrt(np.maximum(k * (2.0 * a[large] - k), 0.0)) * np.sign(r[large])
    return out


def refine_slide_and_object(
    poses: list[Pose],
    uvs: np.ndarray,
    depths: np.ndarray,
    K: np.ndarray,
    table_height: float,
    optical_height: float,
    depth_weight: float = 0.25,
    smooth_weight: float = 5.0,
    slide_y_weight: float = 8.0,
) -> ObjectEstimate:
    n = len(poses)
    uvs = np.asarray(uvs, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    valid = np.isfinite(uvs).all(axis=1)
    if valid.sum() < 3:
        pts = detections_to_world(poses, uvs, depths, K, table_height)
        est = robust_median_xyz(pts)
        xyz = est if est is not None else np.array([np.nan, np.nan, table_height])
        xyz[2] = table_height
        return ObjectEstimate(xyz=xyz, per_frame=pts, rms_reproj_px=np.nan, n_used=int(valid.sum()))

    cam_t = np.stack([p.t.copy() for p in poses])
    cam_R = np.stack([p.R.copy() for p in poses])
    pts0 = detections_to_world(poses, uvs, depths, K, table_height)
    seed = robust_median_xyz(pts0)
    if seed is None:
        seed = np.array([0.0, 0.9, table_height])

    # Unknowns: cam_x[n], cam_y[n], obj_x, obj_y
    x0 = np.concatenate([cam_t[:, 0], cam_t[:, 1], seed[:2]])
    z_cam = table_height + optical_height
    idx = np.where(valid)[0]

    def pack(v):
        cx, cy = v[:n], v[n : 2 * n]
        ox, oy = v[-2], v[-1]
        return cx, cy, ox, oy

    def residuals(v):
        cx, cy, ox, oy = pack(v)
        obj = np.array([ox, oy, table_height])
        res = []
        for i in idx:
            C = np.array([cx[i], cy[i], z_cam])
            R = cam_R[i]
            p_cam = R.T @ (obj - C)
            if p_cam[2] <= 0.05:
                res.extend([50.0, 50.0])
                continue
            uv_hat = project_point(K, p_cam)
            ruv = uvs[i] - uv_hat
            res.extend(_huber(ruv, 4.0).tolist())
            if np.isfinite(depths[i]) and depths[i] > 0.05:
                res.append(depth_weight * _huber(np.array([p_cam[2] - depths[i]]), 0.08)[0])
        # Slide prior: stay near initial VO/IMU path and y ≈ 0.
        for i in range(n):
            res.append(slide_y_weight * cy[i])
            res.append(1.5 * (cx[i] - cam_t[i, 0]))
            res.append(1.5 * (cy[i] - cam_t[i, 1]))
        for i in range(1, n):
            res.append(smooth_weight * ((cx[i] - cx[i - 1]) - (cam_t[i, 0] - cam_t[i - 1, 0])))
            res.append(smooth_weight * (cy[i] - cy[i - 1]))
        return np.asarray(res, dtype=np.float64)

    sol = least_squares(residuals, x0, method="trf", max_nfev=200, ftol=1e-8, xtol=1e-8)
    cx, cy, ox, oy = pack(sol.x)
    xyz = np.array([ox, oy, table_height], dtype=np.float64)

    # Per-frame world points using refined camera translation, original rotation.
    per = np.full((n, 3), np.nan)
    reproj = []
    for i in idx:
        pose = Pose(cam_R[i], np.array([cx[i], cy[i], z_cam]))
        if np.isfinite(depths[i]) and depths[i] > 0.05:
            per[i] = world_from_cam_point(pose, backproject(K, uvs[i], depths[i]))
            per[i, 2] = table_height
        C = pose.t
        p_cam = pose.R.T @ (xyz - C)
        if p_cam[2] > 0.05:
            err = uvs[i] - project_point(K, p_cam)
            if np.isfinite(err).all():
                reproj.append(np.linalg.norm(err))
    rms = float(np.sqrt(np.mean(np.square(reproj)))) if reproj else float("nan")
    return ObjectEstimate(xyz=xyz, per_frame=per, rms_reproj_px=rms, n_used=len(idx))


def _path_score(
    poses: list[Pose],
    uvs: np.ndarray,
    depths: np.ndarray,
    K: np.ndarray,
    table_height: float,
) -> float:
    """Lower is better: object should look static at a plausible table range."""
    pts = detections_to_world(poses, uvs, depths, K, table_height)
    valid = pts[np.isfinite(pts).all(axis=1)]
    if len(valid) < 5:
        return 1e9
    med = np.median(valid, axis=0)
    if not (0.35 < med[1] < 1.8):
        return 1e9
    spread = float(np.median(np.linalg.norm(valid[:, :2] - med[:2], axis=1)))
    span = abs(float(poses[-1].t[0] - poses[0].t[0]))
    if span > 2.5:
        return 1e9
    # Tiny bonus for having actually slid; huge penalty for a frozen camera.
    motion_pen = 0.15 if span < 0.04 else 0.0
    return spread + motion_pen


def fuse_object(
    imu_poses: list[Pose],
    vo_poses: Optional[list[Pose]],
    uvs: np.ndarray,
    depths: np.ndarray,
    K: np.ndarray,
    table_height: float,
    optical_height: float,
    depth_weight: float = 0.25,
    smooth_weight: float = 5.0,
    slide_y_weight: float = 8.0,
) -> tuple[ObjectEstimate, list[Pose]]:
    """Pick the camera path that keeps the object most stationary, then refine."""
    poses = imu_poses
    if vo_poses is not None and len(vo_poses) == len(imu_poses):
        s_imu = _path_score(imu_poses, uvs, depths, K, table_height)
        s_vo = _path_score(vo_poses, uvs, depths, K, table_height)
        if s_vo < s_imu:
            poses = vo_poses
    est = refine_slide_and_object(
        poses,
        uvs,
        depths,
        K,
        table_height,
        optical_height,
        depth_weight=depth_weight,
        smooth_weight=smooth_weight,
        slide_y_weight=slide_y_weight,
    )
    return est, poses
