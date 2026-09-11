"""Undistort RGB/depth and correct an inverted (Y-up) OAK mount."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# 180° about the optical axis: x'=-x, y'=-y, z'=z (image upside-down → upright).
R_INVERT = np.diag([-1.0, -1.0, 1.0]).astype(np.float64)


@dataclass
class ImageCorrection:
    map1: np.ndarray | None
    map2: np.ndarray | None
    K: np.ndarray
    inverted: bool
    width: int
    height: int


def detect_inverted_from_accel(accel: np.ndarray) -> bool:
    """RDF y-down: rest ay<0 if upright, ay>0 if the module Y axis points up."""
    a = np.asarray(accel, dtype=np.float64)
    if a.ndim == 1:
        mean = a
    else:
        if len(a) == 0:
            return True
        mean = np.mean(a, axis=0)
    return float(mean[1]) > 0.0


def rotate_k_180(K: np.ndarray, width: int, height: int) -> np.ndarray:
    Kn = np.array(K, dtype=np.float64, copy=True)
    Kn[0, 2] = width - 1.0 - Kn[0, 2]
    Kn[1, 2] = height - 1.0 - Kn[1, 2]
    return Kn


def apply_invert_vec(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64).reshape(3)
    return R_INVERT @ v


def apply_invert_quat(q: np.ndarray) -> np.ndarray:
    """Body 180° about Z: q' = q ⊗ (0, 0, 0, 1)."""
    q = np.asarray(q, dtype=np.float64).reshape(4)
    if not np.isfinite(q).all():
        return q
    # Hamilton: q * qz180, qz180 = [0, 0, 0, 1]
    w, x, y, z = q
    # q ⊗ (0, 0, 0, 1)  — 180° about body Z
    return np.array([-z, y, -x, w], dtype=np.float64)


def invert_imu_to_cam(T: Optional[list]) -> Optional[list]:
    if T is None:
        return None
    M = np.asarray(T, dtype=np.float64).reshape(4, 4)
    F = np.eye(4)
    F[:3, :3] = R_INVERT
    return (F @ M).tolist()


def build_undistort_maps(K, dist, width: int, height: int, alpha: float = 0.0):
    import cv2

    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.asarray(dist, dtype=np.float64).reshape(-1)
    if dist.size == 0 or np.allclose(dist, 0):
        return None, None, K
    newK, _ = cv2.getOptimalNewCameraMatrix(K, dist, (width, height), float(alpha), (width, height))
    map1, map2 = cv2.initUndistortRectifyMap(K, dist, None, newK, (width, height), cv2.CV_16SC2)
    return map1, map2, np.asarray(newK, dtype=np.float64)


def correct_bgr(cv2, img: np.ndarray, maps: ImageCorrection) -> np.ndarray:
    out = img
    if maps.map1 is not None:
        out = cv2.remap(out, maps.map1, maps.map2, interpolation=cv2.INTER_LINEAR)
    if maps.inverted:
        out = cv2.rotate(out, cv2.ROTATE_180)
    return out


def correct_depth(cv2, depth: np.ndarray, maps: ImageCorrection) -> np.ndarray:
    out = depth
    if maps.map1 is not None:
        out = cv2.remap(out, maps.map1, maps.map2, interpolation=cv2.INTER_NEAREST)
    if maps.inverted:
        out = cv2.rotate(out, cv2.ROTATE_180)
    return out
