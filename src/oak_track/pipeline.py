"""Offline processing: IMU camera CSV (Part A) and object XYZ (Part B)."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from oak_track.geometry import Pose
from oak_track.imu import IMUOdometryConfig, IMUSample, integrate_imu, poses_at_times
from oak_track.io_utils import (
    format_xyz,
    load_calibration,
    read_frames_csv,
    read_imu_raw_csv,
    read_json,
    run_paths,
    write_frame_xyz_csv,
    write_json,
)
from oak_track.cloud import write_run_cloud
from oak_track.localize import fuse_object
from oak_track.vision import ObjectTracker, stereo_visual_odometry


def _load_color(paths: dict, n_hint: int) -> list[np.ndarray]:
    color_dir = paths.get("color_dir")
    if color_dir is not None and Path(color_dir).exists():
        files = sorted(Path(color_dir).glob("*.png"))
        if files:
            return [cv2.imread(str(p)) for p in files]
    return _load_video(paths["color"])


def _load_video(path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frm = cap.read()
        if not ok:
            break
        frames.append(frm)
    cap.release()
    return frames


def _load_depth(depth_dir: Path, n: int) -> list[Optional[np.ndarray]]:
    out: list[Optional[np.ndarray]] = []
    for i in range(n):
        p = depth_dir / f"{i:06d}.png"
        if p.exists():
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            out.append(img)
        else:
            out.append(None)
    return out


def _imu_samples(raw: dict) -> list[IMUSample]:
    samples = []
    quat = raw.get("quat")
    lin = raw.get("linear_accel")
    for i in range(len(raw["t"])):
        q = None
        if quat is not None and np.isfinite(quat[i]).all():
            q = quat[i]
        la = None
        if lin is not None and np.isfinite(lin[i]).all():
            la = lin[i]
        samples.append(
            IMUSample(
                t=float(raw["t"][i]),
                accel=raw["accel"][i],
                gyro=raw["gyro"][i],
                quat_body_to_world=q,
                linear_accel=la,
            )
        )
    return samples


def process_run(
    run_dir: Path,
    table_height: Optional[float] = None,
    optical_height: Optional[float] = None,
    detector: str = "hsv",
    still_time_s: float = 1.0,
    write_preview: bool = True,
    slide_distance: Optional[float] = None,
    write_cloud: bool = True,
    cloud_pixel_stride: int = 2,
    cloud_frame_stride: int = 1,
    cloud_voxel: float = 0.01,
    cloud_depth_min: float = 0.3,
    cloud_depth_max: float = 3.0,
    **det_kwargs,
) -> dict:
    run_dir = Path(run_dir)
    paths = run_paths(run_dir)
    meta = read_json(paths["meta"]) if paths["meta"].exists() else {}
    h = float(table_height if table_height is not None else meta.get("table_height_m", 0.75))
    z_off = float(
        optical_height
        if optical_height is not None
        else meta.get("camera_optical_height_m", 0.03)
    )
    calib = load_calibration(paths["calibration"])
    K = calib.K_np()
    frames_info = read_frames_csv(paths["frames"])
    frame_nos = frames_info["frame_number"]
    t_color = frames_info["t_color"]

    color = _load_color(paths, len(frame_nos))
    n = min(len(color), len(frame_nos))
    color = color[:n]
    frame_nos = frame_nos[:n]
    t_color = t_color[:n]
    depths = _load_depth(paths["depth_dir"], n)

    raw = read_imu_raw_csv(paths["imu"])
    samples = _imu_samples(raw)
    cfg = IMUOdometryConfig(
        table_height=h,
        optical_height=z_off,
        still_time_s=float(meta.get("still_time_s", still_time_s)),
    )
    imu_state = integrate_imu(samples, cfg)
    imu_poses = poses_at_times(imu_state, t_color)
    if slide_distance is not None:
        dx = imu_poses[-1].t[0] - imu_poses[0].t[0]
        if abs(dx) > 1e-4:
            scale = float(slide_distance) / dx
            imu_poses = [
                Pose(p.R, np.array([p.t[0] * scale, p.t[1], p.t[2]])) for p in imu_poses
            ]
    cam_xyz = np.stack([p.t.copy() for p in imu_poses])
    cam_xyz[:, 2] = h
    cam_xyz[0] = np.array([0.0, 0.0, h])
    write_frame_xyz_csv(paths["camera_imu"], frame_nos, cam_xyz)

    tracker_kwargs = {
        k: v
        for k, v in det_kwargs.items()
        if k in {"aruco_id", "hsv_lower", "hsv_upper", "hsv_lower2", "hsv_upper2", "depth_min_m", "depth_max_m"}
    }
    tracker = ObjectTracker(mode=detector, **tracker_kwargs)
    uvs = np.full((n, 2), np.nan)
    zdet = np.full(n, np.nan)
    for i in range(n):
        det = tracker.detect(color[i], depths[i])
        if det is None:
            continue
        uvs[i] = det.uv
        if det.depth_m is not None:
            zdet[i] = det.depth_m

    vo_poses = None
    if any(d is not None for d in depths):
        try:
            vo_poses = stereo_visual_odometry(color, depths, K, h, z_off)
            vo_xyz = np.stack([p.t.copy() for p in vo_poses])
            vo_xyz[:, 2] = h
            write_frame_xyz_csv(paths["camera_vo"], frame_nos, vo_xyz[:n])
        except Exception:
            vo_poses = None

    est, used_poses = fuse_object(
        imu_poses,
        vo_poses if vo_poses is not None else None,
        uvs,
        zdet,
        K,
        h,
        z_off,
        depth_weight=float(det_kwargs.get("depth_weight", 0.25)),
        smooth_weight=float(det_kwargs.get("smooth_weight", 5.0)),
        slide_y_weight=float(det_kwargs.get("slide_y_weight", 8.0)),
    )
    obj_frames = np.tile(est.xyz, (n, 1))
    # Per-frame measurements where available, fused value elsewhere.
    per = est.per_frame
    for i in range(n):
        if np.isfinite(per[i]).all():
            obj_frames[i] = per[i]
            obj_frames[i, 2] = h
    write_frame_xyz_csv(paths["object"], frame_nos, obj_frames)
    write_frame_xyz_csv(paths["object_fused"], [0], est.xyz.reshape(1, 3))

    det_rows = []
    for i in range(n):
        det_rows.append(
            {
                "frame_number": int(frame_nos[i]),
                "u": None if not np.isfinite(uvs[i, 0]) else float(uvs[i, 0]),
                "v": None if not np.isfinite(uvs[i, 1]) else float(uvs[i, 1]),
                "depth_m": None if not np.isfinite(zdet[i]) else float(zdet[i]),
            }
        )
    write_json(paths["detections"], det_rows)

    if write_preview and color:
        hgt, wdt = color[0].shape[:2]
        wr = cv2.VideoWriter(
            str(paths["preview"]), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (wdt, hgt)
        )
        for i, img in enumerate(color):
            vis = img.copy()
            if np.isfinite(uvs[i]).all():
                cv2.circle(vis, (int(uvs[i, 0]), int(uvs[i, 1])), 8, (0, 255, 0), 2)
            cv2.putText(
                vis,
                f"cam {format_xyz(cam_xyz[i], 3)}",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                vis,
                f"obj {format_xyz(est.xyz, 3)}",
                (12, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            wr.write(vis)
        wr.release()

    cloud_meta = None
    if write_cloud:
        cloud_meta = write_run_cloud(
            run_dir,
            color,
            depths,
            used_poses,
            K,
            pixel_stride=cloud_pixel_stride,
            frame_stride=cloud_frame_stride,
            voxel_m=cloud_voxel,
            depth_min_m=cloud_depth_min,
            depth_max_m=cloud_depth_max,
        )

    summary = {
        "camera_imu_csv": str(paths["camera_imu"]),
        "object_csv": str(paths["object"]),
        "object_fused_xyz": est.xyz.tolist(),
        "object_rms_reproj_px": est.rms_reproj_px,
        "n_object_detections": int(est.n_used),
        "table_height_m": h,
        "first_frame_origin": [0.0, 0.0, h],
        "cloud_ply": None if cloud_meta is None else cloud_meta.get("cloud_ply"),
        "cloud_n_points": None if cloud_meta is None else cloud_meta.get("n_points"),
    }
    write_json(run_dir / "summary.json", summary)
    return summary


def capture_then_process(
    out_dir: Path,
    table_height: float,
    optical_height: float = 0.03,
    fps: int = 30,
    color_size=(1280, 720),
    mono_resolution: str = "800p",
    imu_rate_hz: int = 200,
    ir_dot_projector: bool = True,
    save_depth: bool = True,
    duration_s: Optional[float] = None,
    detector: str = "hsv",
    still_time_s: float = 1.0,
    write_preview: bool = True,
    slide_distance: Optional[float] = None,
    simulate: bool = False,
    object_xy=(0.05, 0.90),
    slide_m: float = 0.40,
    still_s: float = 1.0,
    slide_s: float = 2.5,
    orientation: str = "auto",
    undistort_alpha: float = 0.0,
    write_cloud: bool = True,
    cloud_pixel_stride: int = 2,
    cloud_frame_stride: int = 1,
    cloud_voxel: float = 0.01,
    cloud_depth_min: float = 0.3,
    cloud_depth_max: float = 3.0,
    **det_kwargs,
) -> dict:
    """Record (or simulate) a slide, then write camera/object CSVs in the same folder."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if simulate:
        from oak_track.simulate import simulate_run

        simulate_run(
            out_dir=out_dir,
            table_height=table_height,
            optical_height=optical_height,
            object_xy=object_xy,
            slide_m=slide_m,
            still_s=still_s,
            slide_s=slide_s,
        )
    else:
        from oak_track.capture import record_oak

        record_oak(
            out_dir=out_dir,
            table_height=table_height,
            fps=fps,
            color_size=color_size,
            mono_resolution=mono_resolution,
            imu_rate_hz=imu_rate_hz,
            ir_dot_projector=ir_dot_projector,
            save_depth=save_depth,
            duration_s=duration_s,
            camera_optical_height_m=optical_height,
            orientation=orientation,
            undistort_alpha=undistort_alpha,
        )
    return process_run(
        run_dir=out_dir,
        table_height=table_height,
        optical_height=optical_height,
        detector=detector,
        still_time_s=still_time_s,
        write_preview=write_preview,
        slide_distance=slide_distance,
        write_cloud=write_cloud,
        cloud_pixel_stride=cloud_pixel_stride,
        cloud_frame_stride=cloud_frame_stride,
        cloud_voxel=cloud_voxel,
        cloud_depth_min=cloud_depth_min,
        cloud_depth_max=cloud_depth_max,
        **det_kwargs,
    )
