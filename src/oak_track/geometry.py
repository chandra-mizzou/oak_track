"""World / camera / IMU geometry for a table-slide capture.

World frame (right-handed):
  origin at the first-frame camera position, reported as (0, 0, h)
  +X  along the table width (slide direction, camera-right at t=0)
  +Y  along the table toward the object (camera forward at t=0)
  +Z  up
  table plane is z = h

Camera / RDF IMU frame (OpenCV / Luxonis):
  +x right, +y down, +z forward
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np

G = 9.80665

# Maps camera/RDF axes into the world frame defined above.
R_WORLD_FROM_CAM = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v)
    if n < eps:
        return v
    return v / n


def wrap_angle(a: float) -> float:
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, both [w, x, y, z]."""
    w1, x1, y1, z1 = quat_normalize(q1)
    w2, x2, y2, z2 = quat_normalize(q2)
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quat_from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Quaternion rotating unit vector a onto unit vector b."""
    a = normalize(a)
    b = normalize(b)
    c = np.dot(a, b)
    if c < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = normalize(axis)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    v = np.cross(a, b)
    q = np.array([1.0 + c, v[0], v[1], v[2]])
    return quat_normalize(q)


def quat_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return quat_normalize(np.array([w, x, y, z]))


def rotation_matrix_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    return rotation_matrix_from_quat(q) @ np.asarray(v, dtype=np.float64).reshape(3)


def slerp(q0: np.ndarray, q1: np.ndarray, u: float) -> np.ndarray:
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return quat_normalize(q0 + u * (q1 - q0))
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    so = np.sin(theta)
    return quat_normalize(
        (np.sin((1.0 - u) * theta) / so) * q0 + (np.sin(u * theta) / so) * q1
    )


def quat_from_gravity(accel: np.ndarray) -> np.ndarray:
    """Orientation that maps body specific-force direction to world +Z (z-up rest)."""
    # At rest the accelerometer reads ~+g opposite gravity, i.e. body "up".
    up_body = normalize(accel)
    return quat_from_two_vectors(up_body, np.array([0.0, 0.0, 1.0]))


@dataclass
class Pose:
    """p_world = R @ p_frame + t, with t the frame origin in world."""

    R: np.ndarray
    t: np.ndarray

    def inverse(self) -> "Pose":
        Rt = self.R.T
        return Pose(Rt, -Rt @ self.t)

    def apply(self, p: np.ndarray) -> np.ndarray:
        return (self.R @ np.asarray(p, dtype=np.float64).reshape(3).T).T + self.t

    def compose(self, other: "Pose") -> "Pose":
        return Pose(self.R @ other.R, self.R @ other.t + self.t)

    def as_rt(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.R.copy(), self.t.copy()


def nominal_cam_to_world(position: np.ndarray) -> Pose:
    return Pose(R_WORLD_FROM_CAM.copy(), np.asarray(position, dtype=np.float64).reshape(3))


def rebase_first_pose(
    R_world_from_body: np.ndarray,
    t_world_body: np.ndarray,
    table_height: float,
    optical_height: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (R_align, t_align) so aligned pose of sample 0 is nominal at (0,0,h+offset)."""
    R0 = np.asarray(R_world_from_body, dtype=np.float64).reshape(3, 3)
    t0 = np.asarray(t_world_body, dtype=np.float64).reshape(3)
    R_target = R_WORLD_FROM_CAM
    R_align = R_target @ R0.T
    t_target = np.array([0.0, 0.0, table_height + optical_height])
    t_align = t_target - R_align @ t0
    return R_align, t_align


def apply_rebase(
    R_align: np.ndarray, t_align: np.ndarray, R: np.ndarray, t: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    Rn = R_align @ R
    tn = R_align @ t + t_align
    return Rn, tn


def camera_matrix(fx: float, fy: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def project_point(K: np.ndarray, p_cam: np.ndarray) -> np.ndarray:
    p = np.asarray(p_cam, dtype=np.float64).reshape(3)
    if p[2] <= 1e-9:
        return np.array([np.nan, np.nan])
    u = K @ p
    return np.array([u[0] / u[2], u[1] / u[2]])


def backproject(K: np.ndarray, uv: np.ndarray, z: float) -> np.ndarray:
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    u, v = np.asarray(uv, dtype=np.float64).reshape(2)
    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return np.array([x, y, z], dtype=np.float64)


def pixel_ray_cam(K: np.ndarray, uv: np.ndarray) -> np.ndarray:
    return normalize(backproject(K, uv, 1.0))


def pixel_ray_world(pose: Pose, K: np.ndarray, uv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return (camera_origin_world, unit_direction_world) for a pixel."""
    d_cam = pixel_ray_cam(K, uv)
    d_w = pose.R @ d_cam
    n = np.linalg.norm(d_w)
    if n < 1e-12:
        return pose.t.copy(), d_w
    return pose.t.copy(), d_w / n


def pixel_to_plane(K: np.ndarray, pose: Pose, uv: np.ndarray, plane_z: float) -> np.ndarray | None:
    """Plane-induced mapping: image pixel → point on the horizontal plane z = plane_z.

    This is the homography induced by that plane (no IMU translation required
    beyond the camera pose used to form the ray).
    """
    origin, direction = pixel_ray_world(pose, K, uv)
    return ray_plane_intersection(origin, direction, plane_z)


def transform_points(R: np.ndarray, t: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    return pts @ R.T + t.reshape(1, 3)


def world_from_cam_point(pose: Pose, p_cam: np.ndarray) -> np.ndarray:
    return pose.apply(p_cam)


def triangulate_dlt(P1: np.ndarray, P2: np.ndarray, uv1: np.ndarray, uv2: np.ndarray) -> np.ndarray:
    """Linear triangulation from two projection matrices 3x4."""
    u1, v1 = uv1
    u2, v2 = uv2
    A = np.array(
        [
            u1 * P1[2] - P1[0],
            v1 * P1[2] - P1[1],
            u2 * P2[2] - P2[0],
            v2 * P2[2] - P2[1],
        ]
    )
    _, _, vh = np.linalg.svd(A)
    x = vh[-1]
    x = x / x[3]
    return x[:3]


def projection_matrix(K: np.ndarray, pose_world_from_cam: Pose) -> np.ndarray:
    """P = K [R|t] mapping world -> image, with x_cam = R_cw p_world + t_cw."""
    pose_cam_from_world = pose_world_from_cam.inverse()
    Rt = np.hstack([pose_cam_from_world.R, pose_cam_from_world.t.reshape(3, 1)])
    return K @ Rt


def ray_plane_intersection(
    origin: np.ndarray, direction: np.ndarray, plane_z: float
) -> np.ndarray | None:
    """Intersect a world ray with the horizontal plane z = plane_z."""
    o = np.asarray(origin, dtype=np.float64).reshape(3)
    d = np.asarray(direction, dtype=np.float64).reshape(3)
    if abs(d[2]) < 1e-9:
        return None
    lam = (plane_z - o[2]) / d[2]
    if lam <= 0:
        return None
    return o + lam * d


def interpolate_vector(t: np.ndarray, y: np.ndarray, tq: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    tq = np.asarray(tq, dtype=np.float64)
    out = np.empty((len(tq), y.shape[1]), dtype=np.float64)
    for i in range(y.shape[1]):
        out[:, i] = np.interp(tq, t, y[:, i])
    return out


def interpolate_quats(t: np.ndarray, qs: np.ndarray, tq: Iterable[float]) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    qs = np.asarray(qs, dtype=np.float64)
    tq = np.asarray(list(tq), dtype=np.float64)
    out = np.zeros((len(tq), 4))
    for i, ti in enumerate(tq):
        if ti <= t[0]:
            out[i] = qs[0]
            continue
        if ti >= t[-1]:
            out[i] = qs[-1]
            continue
        j = int(np.searchsorted(t, ti) - 1)
        u = (ti - t[j]) / max(t[j + 1] - t[j], 1e-12)
        out[i] = slerp(qs[j], qs[j + 1], u)
    return out


def constrain_table_height(positions: np.ndarray, h: float) -> np.ndarray:
    p = np.array(positions, dtype=np.float64, copy=True)
    p[:, 2] = h
    return p
