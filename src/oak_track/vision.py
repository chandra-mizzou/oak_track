"""Object detection and stereo visual odometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from oak_track.geometry import Pose


@dataclass
class Detection:
    uv: np.ndarray  # (u, v) pixel
    depth_m: Optional[float]
    score: float = 1.0
    bbox: Optional[np.ndarray] = None


def _red_mask(bgr: np.ndarray, lo1, hi1, lo2, hi2) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m1 = cv2.inRange(hsv, np.array(lo1), np.array(hi1))
    m2 = cv2.inRange(hsv, np.array(lo2), np.array(hi2))
    mask = cv2.bitwise_or(m1, m2)
    mask = cv2.medianBlur(mask, 5)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def _largest_blob(mask: np.ndarray) -> Optional[tuple[np.ndarray, np.ndarray]]:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 20:
        return None
    m = cv2.moments(c)
    if m["m00"] <= 1e-6:
        return None
    u = m["m10"] / m["m00"]
    v = m["m01"] / m["m00"]
    x, y, w, h = cv2.boundingRect(c)
    return np.array([u, v], dtype=np.float64), np.array([x, y, w, h], dtype=np.float64)


def median_depth_m(depth_mm: Optional[np.ndarray], uv: np.ndarray, bbox: Optional[np.ndarray] = None) -> Optional[float]:
    if depth_mm is None:
        return None
    h, w = depth_mm.shape[:2]
    if bbox is not None:
        x, y, bw, bh = [int(v) for v in bbox]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + bw, w), min(y + bh, h)
    else:
        u, v = int(round(uv[0])), int(round(uv[1]))
        rad = 8
        x0, y0 = max(u - rad, 0), max(v - rad, 0)
        x1, y1 = min(u + rad + 1, w), min(v + rad + 1, h)
    roi = depth_mm[y0:y1, x0:x1].astype(np.float64)
    valid = roi[(roi > 100) & (roi < 10000)]
    if valid.size < 5:
        return None
    return float(np.median(valid) / 1000.0)


def detect_hsv(
    bgr: np.ndarray,
    depth_mm: Optional[np.ndarray],
    lo1=(0, 120, 70),
    hi1=(10, 255, 255),
    lo2=(170, 120, 70),
    hi2=(180, 255, 255),
) -> Optional[Detection]:
    mask = _red_mask(bgr, lo1, hi1, lo2, hi2)
    blob = _largest_blob(mask)
    if blob is None:
        return None
    uv, bbox = blob
    return Detection(uv=uv, depth_m=median_depth_m(depth_mm, uv, bbox), bbox=bbox, score=1.0)


def detect_depth_blob(
    bgr: np.ndarray,
    depth_mm: np.ndarray,
    zmin: float = 0.55,
    zmax: float = 1.40,
) -> Optional[Detection]:
    z = depth_mm.astype(np.float64) / 1000.0
    mask = ((z >= zmin) & (z <= zmax)).astype(np.uint8) * 255
    # Prefer compact foreground near image center (object on table, camera looking at it).
    h, w = mask.shape
    ys, xs = np.ogrid[:h, :w]
    dist = ((xs - w / 2.0) / max(w / 2.0, 1)) ** 2 + ((ys - h / 3.0) / max(h / 2.0, 1)) ** 2
    mask[dist > 1.2] = 0
    mask = cv2.medianBlur(mask, 7)
    blob = _largest_blob(mask)
    if blob is None:
        return None
    uv, bbox = blob
    return Detection(uv=uv, depth_m=median_depth_m(depth_mm, uv, bbox), bbox=bbox)


def detect_aruco(bgr: np.ndarray, depth_mm: Optional[np.ndarray], marker_id: int = 0) -> Optional[Detection]:
    if not hasattr(cv2, "aruco"):
        return None
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    try:
        params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(bgr)
    except Exception:
        corners, ids, _ = cv2.aruco.detectMarkers(bgr, dictionary)
    if ids is None:
        return None
    ids = ids.flatten()
    if marker_id not in ids:
        return None
    i = int(np.where(ids == marker_id)[0][0])
    c = corners[i][0]
    uv = c.mean(axis=0)
    x0, y0 = c.min(axis=0)
    x1, y1 = c.max(axis=0)
    bbox = np.array([x0, y0, x1 - x0, y1 - y0])
    return Detection(uv=uv, depth_m=median_depth_m(depth_mm, uv, bbox), bbox=bbox, score=1.0)


class ObjectTracker:
    def __init__(self, mode: str = "hsv", **kwargs) -> None:
        self.mode = mode
        self.kwargs = kwargs
        self._tracker = None
        self._ok = False

    def _init_csrt(self, bgr, bbox) -> None:
        x, y, w, h = [int(v) for v in bbox]
        try:
            self._tracker = cv2.TrackerCSRT_create()
        except Exception:
            self._tracker = cv2.legacy.TrackerCSRT_create() if hasattr(cv2, "legacy") else None
        if self._tracker is not None:
            self._ok = self._tracker.init(bgr, (x, y, w, h))

    def detect(self, bgr: np.ndarray, depth_mm: Optional[np.ndarray]) -> Optional[Detection]:
        mode = self.mode
        det = None
        if mode == "aruco":
            det = detect_aruco(bgr, depth_mm, int(self.kwargs.get("aruco_id", 0)))
        elif mode == "depth_blob":
            if depth_mm is not None:
                det = detect_depth_blob(
                    bgr,
                    depth_mm,
                    float(self.kwargs.get("depth_min_m", 0.55)),
                    float(self.kwargs.get("depth_max_m", 1.40)),
                )
        else:
            det = detect_hsv(
                bgr,
                depth_mm,
                self.kwargs.get("hsv_lower", (0, 120, 70)),
                self.kwargs.get("hsv_upper", (10, 255, 255)),
                self.kwargs.get("hsv_lower2", (170, 120, 70)),
                self.kwargs.get("hsv_upper2", (180, 255, 255)),
            )
        if det is not None and self._tracker is None and det.bbox is not None:
            self._init_csrt(bgr, det.bbox)
            return det
        if det is None and self._tracker is not None:
            ok, box = self._tracker.update(bgr)
            if ok:
                x, y, w, h = box
                uv = np.array([x + w / 2.0, y + h / 2.0])
                bbox = np.array([x, y, w, h])
                return Detection(uv=uv, depth_m=median_depth_m(depth_mm, uv, bbox), bbox=bbox, score=0.6)
        return det


def stereo_visual_odometry(
    frames_bgr: list[np.ndarray],
    depths_mm: list[Optional[np.ndarray]],
    K: np.ndarray,
    table_height: float,
    optical_height: float,
) -> list[Pose]:
    """Incremental RGB-D PnP odometry, first pose at (0, 0, h+offset), nominal axes."""
    n = len(frames_bgr)
    poses = [Pose(np.eye(3), np.zeros(3)) for _ in range(n)]
    from oak_track.geometry import R_WORLD_FROM_CAM

    z = table_height + optical_height
    poses[0] = Pose(R_WORLD_FROM_CAM.copy(), np.array([0.0, 0.0, z]))
    if n < 2:
        return poses

    prev_gray = cv2.cvtColor(frames_bgr[0], cv2.COLOR_BGR2GRAY)
    prev_pts = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=400, qualityLevel=0.01, minDistance=8, blockSize=7
    )
    T_w_c = poses[0]

    for i in range(1, n):
        gray = cv2.cvtColor(frames_bgr[i], cv2.COLOR_BGR2GRAY)
        if prev_pts is None or len(prev_pts) < 12:
            prev_pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=400, qualityLevel=0.01, minDistance=8, blockSize=7
            )
            prev_gray = gray
            poses[i] = T_w_c
            continue
        nxt, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, prev_pts, None, winSize=(21, 21), maxLevel=3)
        if nxt is None:
            poses[i] = T_w_c
            prev_gray = gray
            continue
        good_old = prev_pts[st.flatten() == 1]
        good_new = nxt[st.flatten() == 1]
        depth = depths_mm[i - 1]
        obj, img = [], []
        if depth is not None:
            h, w = depth.shape[:2]
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
            for p0, p1 in zip(good_old, good_new):
                u, v = p0.ravel()
                iu, iv = int(round(u)), int(round(v))
                if not (0 <= iu < w and 0 <= iv < h):
                    continue
                z_m = float(depth[iv, iu]) / 1000.0
                if z_m < 0.2 or z_m > 8.0:
                    continue
                x = (u - cx) / fx * z_m
                y = (v - cy) / fy * z_m
                obj.append([x, y, z_m])
                img.append(p1.ravel())
        if len(obj) >= 12:
            obj = np.asarray(obj, dtype=np.float32)
            img = np.asarray(img, dtype=np.float32)
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                obj,
                img,
                K.astype(np.float64),
                None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                reprojectionError=2.5,
                iterationsCount=100,
                confidence=0.99,
            )
            if ok:
                R_rel, _ = cv2.Rodrigues(rvec)
                t_rel = tvec.reshape(3)
                # p_prev = R_rel p_curr + t_rel  => T_prev_from_curr
                T_prev_curr = Pose(R_rel, t_rel).inverse()
                T_w_c = T_w_c.compose(T_prev_curr)
                # Keep camera on the table plane.
                t = T_w_c.t.copy()
                t[2] = z
                T_w_c = Pose(T_w_c.R, t)
        poses[i] = T_w_c
        prev_gray = gray
        prev_pts = good_new.reshape(-1, 1, 2) if len(good_new) else None

    # Rebase in case PnP drifted the origin off (0,0,*)
    t0 = poses[0].t.copy()
    R0 = poses[0].R.copy()
    R_align = R_WORLD_FROM_CAM @ R0.T
    t_target = np.array([0.0, 0.0, z])
    t_align = t_target - R_align @ t0
    out = []
    for p in poses:
        Rn = R_align @ p.R
        tn = R_align @ p.t + t_align
        tn[2] = z
        out.append(Pose(Rn, tn))
    return out
