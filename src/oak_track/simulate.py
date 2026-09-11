"""Synthetic table-slide run used to test the pipeline without an OAK device."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from oak_track.geometry import (
    G,
    Pose,
    R_WORLD_FROM_CAM,
    camera_matrix,
    project_point,
    quat_from_rotation_matrix,
)
from oak_track.io_utils import (
    Calibration,
    run_paths,
    save_calibration,
    write_frames_csv,
    write_imu_raw_csv,
    write_json,
)


def camera_x_profile(t: np.ndarray, still: float, slide: float, distance: float) -> np.ndarray:
    x, _ = camera_motion(t, still, slide, distance)
    return x


def camera_motion(t: np.ndarray, still: float, slide: float, distance: float) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(t, dtype=np.float64)
    span = max(float(slide), 1e-9)
    u = (t - still) / span
    u_c = np.clip(u, 0.0, 1.0)
    s = u_c * u_c * (3.0 - 2.0 * u_c)
    x = distance * s
    ax = np.zeros_like(t)
    inside = (u > 0.0) & (u < 1.0)
    ax[inside] = distance * (6.0 - 12.0 * u[inside]) / (span * span)
    return x, ax


def table_depth_mm(K: np.ndarray, pose: Pose, width: int, height: int, table_height: float) -> np.ndarray:
    """Dense 16-bit millimetre depth of the table plane z = h, from one camera pose."""
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    us = np.arange(width, dtype=np.float64)
    vs = np.arange(height, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs)
    dirs = np.stack([(uu - cx) / fx, (vv - cy) / fy, np.ones_like(uu)], axis=-1)
    dirs_w = dirs @ pose.R.T
    denom = dirs_w[:, :, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        z_cam = (table_height - float(pose.t[2])) / denom
    p_w = dirs_w * z_cam[..., None] + pose.t.reshape(1, 1, 3)
    valid = (
        np.isfinite(z_cam)
        & (z_cam > 0.08)
        & (z_cam < 8.0)
        & (p_w[:, :, 1] > 0.05)
        & (np.abs(p_w[:, :, 0]) < 4.0)
    )
    out = np.zeros((height, width), dtype=np.uint16)
    mm = np.clip(np.round(z_cam * 1000.0), 1, 65535)
    out[valid] = mm[valid].astype(np.uint16)
    return out


def simulate_run(
    out_dir: Path,
    table_height: float = 0.75,
    optical_height: float = 0.03,
    object_xy=(0.05, 0.90),
    slide_m: float = 0.40,
    still_s: float = 1.0,
    slide_s: float = 2.5,
    fps: int = 30,
    imu_hz: int = 200,
    width: int = 640,
    height: int = 360,
    fx: float = 420.0,
    seed: int = 0,
) -> Path:
    """Write a run folder with color, depth, IMU, and ground truth."""
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = run_paths(out_dir)
    paths["depth_dir"].mkdir(parents=True, exist_ok=True)
    paths["color_dir"].mkdir(parents=True, exist_ok=True)

    duration = still_s + slide_s + 0.4
    t_imu = np.arange(0.0, duration, 1.0 / imu_hz)
    t_frm = np.arange(0.0, duration, 1.0 / fps)
    x_imu, ax_imu = camera_motion(t_imu, still_s, slide_s, slide_m)
    x_frm, _ = camera_motion(t_frm, still_s, slide_s, slide_m)

    R = R_WORLD_FROM_CAM
    g_world = np.array([0.0, 0.0, -G])
    accel = np.zeros((len(t_imu), 3))
    gyro = np.zeros((len(t_imu), 3))
    quat = np.tile(quat_from_rotation_matrix(R), (len(t_imu), 1))
    for i, ax in enumerate(ax_imu):
        a_world = np.array([ax, 0.0, 0.0])
        f_world = a_world - g_world
        accel[i] = R.T @ f_world
        gyro[i] = rng.normal(0.0, 0.0005, size=3)
        accel[i] += rng.normal(0.0, 0.01, size=3)

    K = camera_matrix(fx, fx, width / 2.0 - 0.5, height / 2.0 - 0.5)
    obj = np.array([object_xy[0], object_xy[1], table_height], dtype=np.float64)
    z_cam = table_height + optical_height

    writer = cv2.VideoWriter(
        str(paths["color"]), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height)
    )
    for i, (t, xc) in enumerate(zip(t_frm, x_frm)):
        pose = Pose(R.copy(), np.array([xc, 0.0, z_cam]))
        p_cam = pose.R.T @ (obj - pose.t)
        uv = project_point(K, p_cam)
        img = np.full((height, width, 3), 36, dtype=np.uint8)
        # Wooden-table-like grain plus a grid of larger dots so VO can track.
        yy, xx = np.mgrid[0:height, 0:width]
        img[:, :, 0] = np.clip(36 + (xx * 3 + yy) % 17, 0, 255)
        img[:, :, 1] = np.clip(48 + (yy * 2) % 21, 0, 255)
        img[:, :, 2] = np.clip(70 + (xx + 2 * yy) % 25, 0, 255)
        depth = table_depth_mm(K, pose, width, height, table_height)
        for gx in np.linspace(-0.15, 0.55, 12):
            for gy in np.linspace(0.35, 1.15, 10):
                pw = np.array([gx, gy, table_height])
                pc = pose.R.T @ (pw - pose.t)
                uvd = project_point(K, pc)
                if np.isfinite(uvd).all() and pc[2] > 0.05:
                    u, v = int(round(uvd[0])), int(round(uvd[1]))
                    if 0 <= u < width and 0 <= v < height:
                        cv2.circle(img, (u, v), 3, (40, 90, 140), -1)
                        cv2.circle(depth, (u, v), 3, int(np.clip(pc[2] * 1000.0, 1, 65535)), -1)
        if np.isfinite(uv).all() and p_cam[2] > 0.05:
            u, v = uv
            radius = max(8, int(22.0 / max(p_cam[2], 0.2)))
            cv2.circle(img, (int(round(u)), int(round(v))), radius, (0, 0, 220), -1)
            zz = int(np.clip(p_cam[2] * 1000.0, 1, 65535))
            cv2.circle(depth, (int(round(u)), int(round(v))), radius, zz, -1)
        writer.write(img)
        cv2.imwrite(str(paths["depth_dir"] / f"{i:06d}.png"), depth)
        cv2.imwrite(str(paths["color_dir"] / f"{i:06d}.png"), img)
    writer.release()

    write_imu_raw_csv(paths["imu"], t_imu, accel, gyro, quat=quat)
    write_frames_csv(paths["frames"], list(range(len(t_frm))), t_frm, t_frm)
    save_calibration(
        paths["calibration"],
        Calibration(K=K.tolist(), dist=[0.0] * 5, width=width, height=height, baseline_m=0.075),
    )
    write_json(
        paths["meta"],
        {
            "table_height_m": table_height,
            "camera_optical_height_m": optical_height,
            "fps": fps,
            "simulated": True,
            "still_time_s": still_s,
            "first_frame_origin": [0.0, 0.0, table_height],
        },
    )
    write_json(
        paths["ground_truth"],
        {
            "object_xyz": [float(obj[0]), float(obj[1]), float(obj[2])],
            "camera_x": x_frm.tolist(),
            "slide_m": slide_m,
            "optical_height_m": optical_height,
        },
    )
    return out_dir
