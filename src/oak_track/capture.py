"""DepthAI capture for OAK-D Pro W: color, aligned depth, IMU, timestamps."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import numpy as np

from oak_track.io_utils import (
    Calibration,
    run_paths,
    save_calibration,
    write_frames_csv,
    write_imu_raw_csv,
    write_json,
)


def _get_imu_name(device) -> str:
    try:
        return str(device.getConnectedIMU())
    except Exception:
        return "UNKNOWN"


def _enable_imu(imu, dai, imu_name: str, rate_hz: int) -> None:
    name = imu_name.upper()
    fused = "BNO" in name
    if fused:
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER, rate_hz)
        try:
            imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_CALIBRATED, min(rate_hz, 400))
        except Exception:
            imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, rate_hz)
        try:
            imu.enableIMUSensor(dai.IMUSensor.ROTATION_VECTOR, min(rate_hz, 400))
        except Exception:
            pass
        try:
            imu.enableIMUSensor(dai.IMUSensor.LINEAR_ACCELERATION, min(rate_hz, 400))
        except Exception:
            pass
    else:
        imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, rate_hz)
        imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, rate_hz)
    imu.setBatchReportThreshold(1)
    imu.setMaxBatchReports(10)


def _set_ir(device, enabled: bool) -> None:
    if not enabled:
        return
    for fn_name, args in (
        ("setIrLaserDotProjectorIntensity", (1.0,)),
        ("setIrLaserDotProjectorBrightness", (800,)),
        ("setIrLaserDotProjectorBrightness", (1, 800)),
    ):
        fn = getattr(device, fn_name, None)
        if fn is None:
            continue
        try:
            fn(*args)
            return
        except TypeError:
            continue
        except Exception:
            continue


def _mono_res(dai, name: str):
    name = name.lower()
    if name in ("800p", "800", "the_800_p"):
        return dai.MonoCameraProperties.SensorResolution.THE_800_P
    return dai.MonoCameraProperties.SensorResolution.THE_400_P


def build_pipeline(dai, fps: int, color_size, mono_resolution: str, imu_rate_hz: int, save_depth: bool):
    pipeline = dai.Pipeline()
    cam = pipeline.create(dai.node.ColorCamera)
    cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
    cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
    cam.setFps(fps)
    cam.setPreviewSize(int(color_size[0]), int(color_size[1]))
    cam.setInterleaved(False)
    cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
    cam.setPreviewKeepAspectRatio(True)

    xout_color = pipeline.create(dai.node.XLinkOut)
    xout_color.setStreamName("color")
    cam.preview.link(xout_color.input)

    imu = pipeline.create(dai.node.IMU)
    # RAW accel/gyro works on both BMI270 and BNO085; host fusion is in imu.py.
    _enable_imu(imu, dai, "BMI270", imu_rate_hz)
    xout_imu = pipeline.create(dai.node.XLinkOut)
    xout_imu.setStreamName("imu")
    imu.out.link(xout_imu.input)

    if save_depth:
        left = pipeline.create(dai.node.MonoCamera)
        right = pipeline.create(dai.node.MonoCamera)
        left.setBoardSocket(dai.CameraBoardSocket.CAM_B)
        right.setBoardSocket(dai.CameraBoardSocket.CAM_C)
        res = _mono_res(dai, mono_resolution)
        left.setResolution(res)
        right.setResolution(res)
        left.setFps(fps)
        right.setFps(fps)

        stereo = pipeline.create(dai.node.StereoDepth)
        try:
            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
        except Exception:
            pass
        stereo.setLeftRightCheck(True)
        try:
            stereo.setSubpixel(True)
        except Exception:
            pass
        # Object is ~0.8-1.0 m; 800P minZ is ~70 cm so this is in range.
        try:
            stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        except Exception:
            pass
        left.out.link(stereo.left)
        right.out.link(stereo.right)
        xout_depth = pipeline.create(dai.node.XLinkOut)
        xout_depth.setStreamName("depth")
        stereo.depth.link(xout_depth.input)

    return pipeline


def _calibration_from_device(device, dai, width: int, height: int) -> Calibration:
    calib = device.readCalibration()
    socket = dai.CameraBoardSocket.CAM_A
    try:
        K = np.array(calib.getCameraIntrinsics(socket, width, height), dtype=np.float64)
    except Exception:
        K = np.array(calib.getCameraIntrinsics(socket), dtype=np.float64)
    try:
        dist = list(calib.getDistortionCoefficients(socket))
    except Exception:
        dist = [0.0] * 14
    imu_to_cam = None
    try:
        imu_to_cam = calib.getImuToCameraExtrinsics(socket)
    except Exception:
        imu_to_cam = None
    K_left = None
    try:
        K_left = calib.getCameraIntrinsics(dai.CameraBoardSocket.CAM_B, 1280, 800)
    except Exception:
        pass
    return Calibration(
        K=K.tolist(),
        dist=[float(x) for x in dist],
        width=width,
        height=height,
        baseline_m=0.075,
        imu_to_cam=imu_to_cam,
        K_left=K_left,
    )


def _packet_vec(report, keys=("x", "y", "z")) -> Optional[np.ndarray]:
    if report is None:
        return None
    try:
        return np.array([float(getattr(report, k)) for k in keys], dtype=np.float64)
    except Exception:
        return None


def _packet_quat(report) -> Optional[np.ndarray]:
    if report is None:
        return None
    try:
        # DepthAI rotation vector: i,j,k,real
        return np.array(
            [float(report.real), float(report.i), float(report.j), float(report.k)],
            dtype=np.float64,
        )
    except Exception:
        return None


UDEV_HELP = """
Linux cannot talk to the OAK-D (X_LINK_UNBOOTED / insufficient USB permissions).
Install DepthAI udev rules once, then unplug and replug the camera:

  echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules
  sudo udevadm control --reload-rules && sudo udevadm trigger

Unplug the OAK-D, plug it back into a USB3 port, wait a few seconds, then rerun.
Do not use sudo python; the udev rule is what grants your user access.
""".strip()

BUSY_HELP = """
The OAK-D is already used by another process (X_LINK_DEVICE_ALREADY_IN_USE).

Find and stop it (a leftover run.py, depthai demo, ROS node, or hung Python):

  pgrep -af 'python|depthai'
  pkill -f 'run.py|record.py|depthai'

Then unplug the camera, wait 3 seconds, plug it back into USB3, and rerun.
Before starting, `lsusb | grep 03e7` should show 03e7:2485 (unbooted).
While recording it typically becomes 03e7:f63b.
""".strip()


def _list_oak_devices(dai) -> list:
    try:
        return list(dai.Device.getAllAvailableDevices())
    except Exception:
        return []


def _device_label(info) -> str:
    mx = getattr(info, "mxid", None) or getattr(info, "getMxId", lambda: "?")()
    proto = str(getattr(info, "protocol", "USB"))
    name = getattr(info, "name", None) or "OAK"
    return f"{name}  mxid={mx}  {proto}"


def _select_oak_info(dai):
    found = _list_oak_devices(dai)
    if not found:
        raise SystemExit("No usable OAK camera on USB.\n\n" + UDEV_HELP + "\n\n" + BUSY_HELP)
    info = found[0]
    print(f"Detected {len(found)} OAK device(s). Using: {_device_label(info)}")
    for extra in found[1:]:
        print(f"  (not used) {_device_label(extra)}")
    return info


def _connect_device(dai, pipeline, info):
    try:
        try:
            return dai.Device(pipeline, info)
        except TypeError:
            return dai.Device(pipeline)
    except RuntimeError as exc:
        text = str(exc)
        if "ALREADY_IN_USE" in text or "another process" in text.lower():
            raise SystemExit(f"{exc}\n\n{BUSY_HELP}") from exc
        if "UNBOOTED" in text or "permission" in text.lower() or "No available" in text:
            raise SystemExit(f"{exc}\n\n{UDEV_HELP}") from exc
        raise SystemExit(f"{exc}\n\n{BUSY_HELP}\n\n{UDEV_HELP}") from exc


def record_oak(
    out_dir: Path,
    table_height: float,
    fps: int = 30,
    color_size=(1280, 720),
    mono_resolution: str = "800p",
    imu_rate_hz: int = 200,
    ir_dot_projector: bool = True,
    save_depth: bool = True,
    duration_s: Optional[float] = None,
    camera_optical_height_m: float = 0.03,
) -> Path:
    """Record until Ctrl+C or duration_s. Hold still ~1 s at the start for IMU bias."""
    try:
        import cv2
        import depthai as dai
    except ImportError as e:
        raise SystemExit(
            "Recording requires 'depthai' and a connected OAK-D. "
            "Install with: pip install depthai"
        ) from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = run_paths(out_dir)
    paths["depth_dir"].mkdir(parents=True, exist_ok=True)

    info = _select_oak_info(dai)
    pipeline = build_pipeline(
        dai, fps, color_size, mono_resolution, imu_rate_hz, save_depth
    )
    with _connect_device(dai, pipeline, info) as device:
        imu_name = _get_imu_name(device)
        _set_ir(device, ir_dot_projector)

        q_color = device.getOutputQueue("color", maxSize=8, blocking=False)
        q_imu = device.getOutputQueue("imu", maxSize=64, blocking=False)
        q_depth = device.getOutputQueue("depth", maxSize=8, blocking=False) if save_depth else None

        calib = _calibration_from_device(device, dai, int(color_size[0]), int(color_size[1]))
        save_calibration(paths["calibration"], calib)

        writer = None
        imu_t, imu_a, imu_g = [], [], []
        imu_q, imu_la = [], []
        have_q = False
        have_la = False
        frame_idx, t_color, t_depth = [], [], []
        n = 0
        t_start = time.time()
        print(
            f"Recording on {imu_name}. Hold still 1s, then slide along the table width. Ctrl+C to stop."
        )
        try:
            while True:
                if duration_s is not None and (time.time() - t_start) >= duration_s:
                    break
                imu_msg = q_imu.tryGet()
                if imu_msg is not None:
                    for pkt in imu_msg.packets:
                        acc = _packet_vec(getattr(pkt, "acceleroMeter", None))
                        gyr = _packet_vec(getattr(pkt, "gyroscope", None))
                        if acc is None and gyr is None:
                            continue
                        ts_src = getattr(pkt, "acceleroMeter", None) or getattr(pkt, "gyroscope", None)
                        ts = float(ts_src.getTimestamp().total_seconds()) if ts_src is not None else time.time()
                        if acc is None:
                            acc = np.zeros(3)
                        if gyr is None:
                            gyr = np.zeros(3)
                        imu_t.append(ts)
                        imu_a.append(acc)
                        imu_g.append(gyr)
                        qv = _packet_quat(getattr(pkt, "rotationVector", None))
                        if qv is not None:
                            have_q = True
                            imu_q.append(qv)
                        else:
                            imu_q.append(np.array([np.nan, np.nan, np.nan, np.nan]))
                        la = _packet_vec(getattr(pkt, "linearAcceleroMeter", None) or getattr(pkt, "linearAcceleration", None))
                        if la is not None:
                            have_la = True
                            imu_la.append(la)
                        else:
                            imu_la.append(np.array([np.nan, np.nan, np.nan]))

                color_frame = q_color.tryGet()
                if color_frame is None:
                    time.sleep(0.001)
                    continue
                img = color_frame.getCvFrame()
                if writer is None:
                    h, w = img.shape[:2]
                    writer = cv2.VideoWriter(
                        str(paths["color"]),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(fps),
                        (w, h),
                    )
                writer.write(img)
                tc = float(color_frame.getTimestamp().total_seconds())
                td = tc
                if q_depth is not None:
                    depth_frame = q_depth.tryGet()
                    if depth_frame is not None:
                        depth = depth_frame.getFrame()
                        td = float(depth_frame.getTimestamp().total_seconds())
                        cv2.imwrite(
                            str(paths["depth_dir"] / f"{n:06d}.png"),
                            depth.astype(np.uint16),
                        )
                frame_idx.append(n)
                t_color.append(tc)
                t_depth.append(td)
                n += 1
                if n % max(fps, 1) == 0:
                    print(f"  {n} frames, {len(imu_t)} IMU samples")
        except KeyboardInterrupt:
            print("Stopped.")
        if writer is not None:
            writer.release()

    quat = np.vstack(imu_q) if have_q else None
    lin = np.vstack(imu_la) if have_la else None
    if quat is not None and np.isnan(quat).all():
        quat = None
    if lin is not None and np.isnan(lin).all():
        lin = None
    write_imu_raw_csv(
        paths["imu"],
        np.asarray(imu_t),
        np.vstack(imu_a) if imu_a else np.zeros((0, 3)),
        np.vstack(imu_g) if imu_g else np.zeros((0, 3)),
        quat=quat,
        linear_accel=lin,
    )
    write_frames_csv(paths["frames"], frame_idx, t_color, t_depth)
    write_json(
        paths["meta"],
        {
            "table_height_m": table_height,
            "camera_optical_height_m": camera_optical_height_m,
            "fps": fps,
            "color_size": list(color_size),
            "mono_resolution": mono_resolution,
            "imu_rate_hz": imu_rate_hz,
            "imu_name": imu_name,
            "save_depth": save_depth,
            "n_frames": n,
            "n_imu": len(imu_t),
            "first_frame_origin": [0.0, 0.0, table_height],
            "world_frame": {
                "x": "table width / slide, camera-right at t=0",
                "y": "toward object / camera-forward at t=0",
                "z": "up, table plane z = table_height_m",
            },
        },
    )
    print(f"Wrote {n} frames to {out_dir}")
    return out_dir
