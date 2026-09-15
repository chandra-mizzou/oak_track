"""Ground-plane object localisation shared by table-top and long-range modes.

This is **not** IMU-only, and it is **not** a homography estimated from the
object itself.

Per frame the object pixel is turned into a 3D point in two independent ways:

1. **Plane-induced homography** (ray ∩ ground/table plane ``z = h``).
   Needs camera pose + intrinsics. Same geometry at 1 m or 3 km.
2. **Stereo / rangefinder depth** back-projection (OAK-D only, useful to ~8 m).

Those per-frame points are then fused across the slide (wide-baseline
bearings). IMU **translation** is not the primary pose; RGB-D visual
odometry (keypoints + PnP) is.

Why the same code cannot be dropped onto a 3 km test unchanged
--------------------------------------------------------------
OAK-D stereo baseline is 7.5 cm. Depth error grows as Z² / (f B). At 3 km
that error is kilometres. IMU double integration is worse.

At 3 km you keep this module's **ray ∩ ground plane** and **multi-view
triangulation**, but you must replace:

| Table-top (this repo) | 3 km |
| --- | --- |
| Observer pose from RGB-D VO + table slide | GNSS/RTK + INS/attitude |
| Plane ``z = h`` (table) | WGS-84 / local ENU + DEM |
| OAK stereo as range check | laser/radar, or two observers km apart |
| 80–100 cm object, 20–50 cm slide | milliradian bearings, km baseline |

A 1 mrad (~0.06°) pitch error is ~1 mm at 1 m and **3 m** at 3 km. Long-range
needs a calibrated long lens, surveyed observer pose, and preferably two
lines of sight — not the OAK-D Pro W IMU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from oak_track.geometry import (
    Pose,
    backproject,
    pixel_to_plane,
    pixel_ray_world,
    projection_matrix,
    triangulate_dlt,
    world_from_cam_point,
)

# OAK-D Pro W stereo is specified useful roughly 0.7–8 m (800p).
OAK_STEREO_MAX_M = 8.0


@dataclass
class GroundPlaneConfig:
    """Scene-scale knobs. Table-top and long-range share the same geometry."""

    plane_z: float
    range_min_m: float = 0.30
    range_max_m: float = 3.0
    assume_on_plane: bool = True
    consistency_m: float = 0.30
    stereo_max_m: float = OAK_STEREO_MAX_M
    use_stereo: bool = True

    @classmethod
    def table(cls, plane_z: float, **kwargs) -> "GroundPlaneConfig":
        kw = dict(range_min_m=0.30, range_max_m=3.0, assume_on_plane=True, use_stereo=True)
        kw.update(kwargs)
        return cls(plane_z=plane_z, **kw)

    @classmethod
    def longrange(cls, plane_z: float, **kwargs) -> "GroundPlaneConfig":
        """3 km-class: plane + bearings only. Stereo is ignored."""
        kw = dict(
            range_min_m=50.0,
            range_max_m=20000.0,
            assume_on_plane=True,
            use_stereo=False,
            stereo_max_m=0.0,
            consistency_m=50.0,
        )
        kw.update(kwargs)
        return cls(plane_z=plane_z, **kw)


@dataclass
class FrameFix:
    xyz: Optional[np.ndarray]
    source: str  # plane | depth | fused | rejected_background | grazing_depth | none
    plane_xyz: Optional[np.ndarray] = None
    depth_xyz: Optional[np.ndarray] = None


def _in_range(p: np.ndarray, cfg: GroundPlaneConfig) -> bool:
    r = float(np.linalg.norm(p[:2]))
    y = float(p[1])
    return cfg.range_min_m <= y <= cfg.range_max_m and r < 1.5 * cfg.range_max_m


def depth_to_world(pose: Pose, K: np.ndarray, uv: np.ndarray, depth_m: float, plane_z: float) -> Optional[np.ndarray]:
    if not np.isfinite(depth_m) or depth_m <= 0.05:
        return None
    p_w = world_from_cam_point(pose, backproject(K, uv, float(depth_m)))
    p_w = p_w.astype(np.float64)
    p_w[2] = plane_z
    return p_w


def localize_uv(
    pose: Pose,
    K: np.ndarray,
    uv: np.ndarray,
    depth_m: Optional[float],
    cfg: GroundPlaneConfig,
) -> FrameFix:
    """One pixel → table/ground point, with stereo as a consistency gate."""
    _, direction = pixel_ray_world(pose, K, uv)
    grazing = abs(float(direction[2])) < 0.02
    p_plane = None if grazing else pixel_to_plane(K, pose, uv, cfg.plane_z)
    p_depth = None
    if cfg.use_stereo and depth_m is not None and np.isfinite(depth_m) and depth_m <= cfg.stereo_max_m:
        p_depth = depth_to_world(pose, K, uv, float(depth_m), cfg.plane_z)

    if p_plane is not None and not _in_range(p_plane, cfg):
        p_plane = None
    if p_depth is not None and not _in_range(p_depth, cfg):
        # Keep depth if stereo is the only cue (grazing plane).
        if p_plane is not None:
            p_depth = None

    if p_plane is None and p_depth is None:
        src = "grazing_depth" if grazing and cfg.use_stereo else "none"
        return FrameFix(xyz=None, source=src, plane_xyz=None, depth_xyz=p_depth)

    # Looking above the table (wall / distant clutter): no plane hit.
    if p_plane is None and cfg.assume_on_plane and not grazing and float(direction[2]) >= 0:
        return FrameFix(xyz=None, source="rejected_not_on_plane", plane_xyz=None, depth_xyz=p_depth)

    if p_plane is None:
        return FrameFix(xyz=p_depth, source="depth", plane_xyz=None, depth_xyz=p_depth)
    if p_depth is None:
        return FrameFix(xyz=p_plane, source="plane", plane_xyz=p_plane, depth_xyz=None)

    gap = float(np.linalg.norm(p_plane[:2] - p_depth[:2]))
    if gap <= cfg.consistency_m:
        xyz = 0.5 * (p_plane + p_depth)
        xyz[2] = cfg.plane_z
        return FrameFix(xyz=xyz, source="fused", plane_xyz=p_plane, depth_xyz=p_depth)

    # Stereo much farther than the table along this ray → background blob.
    if cfg.assume_on_plane and float(p_depth[1]) > float(p_plane[1]) + cfg.consistency_m:
        return FrameFix(xyz=None, source="rejected_background", plane_xyz=p_plane, depth_xyz=p_depth)

    xyz = p_plane if cfg.assume_on_plane else p_depth
    return FrameFix(xyz=xyz, source="plane" if cfg.assume_on_plane else "depth", plane_xyz=p_plane, depth_xyz=p_depth)


def detections_on_plane(
    poses: list[Pose],
    uvs: np.ndarray,
    depths: np.ndarray,
    K: np.ndarray,
    cfg: GroundPlaneConfig,
) -> tuple[np.ndarray, list[str]]:
    n = len(poses)
    xyz = np.full((n, 3), np.nan)
    sources: list[str] = []
    for i in range(n):
        if not np.isfinite(uvs[i]).all():
            sources.append("none")
            continue
        z = depths[i] if i < len(depths) else np.nan
        fix = localize_uv(poses[i], K, uvs[i], z, cfg)
        sources.append(fix.source)
        if fix.xyz is not None:
            xyz[i] = fix.xyz
    return xyz, sources


def triangulate_static_object(
    poses: list[Pose],
    uvs: np.ndarray,
    K: np.ndarray,
    plane_z: float,
) -> Optional[np.ndarray]:
    """Wide-baseline DLT from the two frames with the largest camera X span."""
    idx = np.where(np.isfinite(uvs).all(axis=1))[0]
    if len(idx) < 2:
        return None
    xs = np.array([poses[i].t[0] for i in idx])
    i0 = int(idx[int(np.argmin(xs))])
    i1 = int(idx[int(np.argmax(xs))])
    if abs(poses[i1].t[0] - poses[i0].t[0]) < 0.04:
        return None
    P1 = projection_matrix(K, poses[i0])
    P2 = projection_matrix(K, poses[i1])
    p = triangulate_dlt(P1, P2, uvs[i0], uvs[i1])
    if not np.isfinite(p).all():
        return None
    p = np.asarray(p, dtype=np.float64).reshape(3)
    p[2] = plane_z
    return p


def camera_path_score(
    poses: list[Pose],
    uvs: np.ndarray,
    depths: np.ndarray,
    K: np.ndarray,
    cfg: GroundPlaneConfig,
) -> float:
    """Lower is better: object should look static and the camera should have slid."""
    pts, _ = detections_on_plane(poses, uvs, depths, K, cfg)
    valid = pts[np.isfinite(pts).all(axis=1)]
    if len(valid) < 5:
        return 1e9
    med = np.median(valid, axis=0)
    spread = float(np.median(np.linalg.norm(valid[:, :2] - med[:2], axis=1)))
    span = abs(float(poses[-1].t[0] - poses[0].t[0]))
    if span > 8.0:
        return 1e9
    motion_pen = 0.15 if span < 0.04 else 0.0
    return spread + motion_pen


def pose_source_vo_or_imu(
    imu_poses: list[Pose],
    vo_poses: Optional[list[Pose]],
    uvs: Optional[np.ndarray] = None,
    depths: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    cfg: Optional[GroundPlaneConfig] = None,
) -> tuple[list[Pose], str]:
    """Prefer the camera path that keeps the object most stationary.

    RGB-D VO (keypoints + PnP) wins when it actually recovered the slide.
    IMU translation is the fallback. No hard 1.8 m object-range gate.
    """
    if vo_poses is None or len(vo_poses) != len(imu_poses):
        return imu_poses, "imu"
    if uvs is not None and depths is not None and K is not None and cfg is not None:
        s_imu = camera_path_score(imu_poses, uvs, depths, K, cfg)
        s_vo = camera_path_score(vo_poses, uvs, depths, K, cfg)
        if s_vo < s_imu:
            return vo_poses, "vo"
        return imu_poses, "imu"
    span_vo = abs(float(vo_poses[-1].t[0] - vo_poses[0].t[0]))
    if 0.04 <= span_vo <= 8.0:
        return vo_poses, "vo"
    return imu_poses, "imu"
