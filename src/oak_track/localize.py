"""Precise object (x, y, z) on a known ground/table plane.

Object localisation is **not** IMU-only and does **not** estimate a homography
from the object. Steps:

1. Detect the object in pixels (HSV/ArUco/depth blob; optional lock-first).
2. Camera pose from RGB-D **keypoint** visual odometry (PnP on stereo depth),
   with IMU used for Part A logging and as a fallback path.
3. Each detection → world point via **plane-induced homography** (ray ∩ z = h)
   gated by stereo depth so background blobs are rejected.
4. Wide-baseline triangulation + joint refine of camera X(t) and one static
   object (X, Y, h).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import least_squares

from oak_track.geometry import Pose, backproject, project_point, world_from_cam_point
from oak_track.ranging import (
    GroundPlaneConfig,
    detections_on_plane,
    pose_source_vo_or_imu,
    triangulate_static_object,
)


@dataclass
class ObjectEstimate:
    xyz: np.ndarray  # fused (x, y, h)
    per_frame: np.ndarray  # (N, 3), nan where missing
    rms_reproj_px: float
    n_used: int
    pose_source: str = "imu"
    n_rejected_background: int = 0
    frame_sources: list[str] | None = None


def detections_to_world(
    poses: list[Pose],
    uvs: np.ndarray,
    depths: np.ndarray,
    K: np.ndarray,
    table_height: float,
    cfg: GroundPlaneConfig | None = None,
) -> np.ndarray:
    """Per-frame object points in world; z forced to table height when finite."""
    if cfg is None:
        cfg = GroundPlaneConfig.table(table_height)
    xyz, _ = detections_on_plane(poses, uvs, depths, K, cfg)
    return xyz


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
    cfg: GroundPlaneConfig | None = None,
    pose_source: str = "imu",
) -> ObjectEstimate:
    if cfg is None:
        cfg = GroundPlaneConfig.table(table_height)
    n = len(poses)
    uvs = np.asarray(uvs, dtype=np.float64)
    depths = np.asarray(depths, dtype=np.float64)
    pts0, sources = detections_on_plane(poses, uvs, depths, K, cfg)
    n_rej = int(sum(s.startswith("rejected") for s in sources))
    valid = np.isfinite(uvs).all(axis=1) & np.isfinite(pts0).all(axis=1)
    if valid.sum() < 3:
        # Fall back to any pixel with a detection if the plane gate wiped them.
        valid_uv = np.isfinite(uvs).all(axis=1)
        est = robust_median_xyz(pts0)
        if est is None:
            raw = np.full((n, 3), np.nan)
            for i in np.where(valid_uv)[0]:
                if np.isfinite(depths[i]) and depths[i] > 0.05:
                    p = world_from_cam_point(poses[i], backproject(K, uvs[i], float(depths[i])))
                    p[2] = table_height
                    raw[i] = p
            est = robust_median_xyz(raw)
            pts0 = raw
        xyz = est if est is not None else np.array([np.nan, np.nan, table_height])
        xyz[2] = table_height
        return ObjectEstimate(
            xyz=xyz,
            per_frame=pts0,
            rms_reproj_px=np.nan,
            n_used=int(np.isfinite(pts0).all(axis=1).sum()),
            pose_source=pose_source,
            n_rejected_background=n_rej,
            frame_sources=sources,
        )

    cam_t = np.stack([p.t.copy() for p in poses])
    cam_R = np.stack([p.R.copy() for p in poses])
    seed = robust_median_xyz(pts0)
    tri = triangulate_static_object(poses, uvs, K, table_height)
    if seed is None and tri is not None:
        seed = tri
    if seed is None:
        seed = np.array([0.0, 0.9, table_height])
    if tri is not None and np.isfinite(tri).all():
        # Blend DLT range into the seed (helps when stereo latched on background).
        seed = np.array([0.5 * (seed[0] + tri[0]), 0.5 * (seed[1] + tri[1]), table_height])

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
            if np.isfinite(pts0[i]).all():
                res.append(2.0 * _huber(np.array([ox - pts0[i, 0]]), 0.08)[0])
                res.append(2.0 * _huber(np.array([oy - pts0[i, 1]]), 0.08)[0])
            if cfg.use_stereo and np.isfinite(depths[i]) and 0.05 < depths[i] <= cfg.stereo_max_m:
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

    per = np.full((n, 3), np.nan)
    reproj = []
    for i in idx:
        pose = Pose(cam_R[i], np.array([cx[i], cy[i], z_cam]))
        if np.isfinite(pts0[i]).all():
            per[i] = pts0[i]
        elif cfg.use_stereo and np.isfinite(depths[i]) and depths[i] > 0.05:
            per[i] = world_from_cam_point(pose, backproject(K, uvs[i], depths[i]))
            per[i, 2] = table_height
        C = pose.t
        p_cam = pose.R.T @ (xyz - C)
        if p_cam[2] > 0.05:
            err = uvs[i] - project_point(K, p_cam)
            if np.isfinite(err).all():
                reproj.append(np.linalg.norm(err))
    rms = float(np.sqrt(np.mean(np.square(reproj)))) if reproj else float("nan")
    return ObjectEstimate(
        xyz=xyz,
        per_frame=per,
        rms_reproj_px=rms,
        n_used=len(idx),
        pose_source=pose_source,
        n_rejected_background=n_rej,
        frame_sources=sources,
    )


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
    cfg: GroundPlaneConfig | None = None,
) -> tuple[ObjectEstimate, list[Pose]]:
    """Use VO camera path when the slide is recovered, then refine."""
    if cfg is None:
        cfg = GroundPlaneConfig.table(table_height)
    poses, source = pose_source_vo_or_imu(imu_poses, vo_poses, uvs, depths, K, cfg)
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
        cfg=cfg,
        pose_source=source,
    )
    return est, poses
