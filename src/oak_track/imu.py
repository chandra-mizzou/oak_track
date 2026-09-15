"""IMU attitude fusion and table-constrained strapdown odometry.

OAK-D Pro W ships with either BNO085/BNO086 (on-chip rotation vector) or BMI270
(raw accel + gyro only). Double-integrating accelerometer data drifts quickly;
this module still produces the Part A CSV the experiment asked for, with table
constraints (constant height, optional 1-D slide) to keep the result usable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from oak_track.geometry import (
    G,
    Pose,
    apply_rebase,
    quat_from_gravity,
    quat_from_rotation_matrix,
    quat_normalize,
    rebase_first_pose,
    rotation_matrix_from_quat,
)


def _quat_derivative_from_gyro(q: np.ndarray, gyro: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    gx, gy, gz = gyro
    return 0.5 * np.array(
        [
            -x * gx - y * gy - z * gz,
            w * gx + y * gz - z * gy,
            w * gy - x * gz + z * gx,
            w * gz + x * gy - y * gx,
        ]
    )


class Madgwick:
    """IMU orientation filter. Quaternion maps body RDF -> world (z-up)."""

    def __init__(self, beta: float = 0.08) -> None:
        self.beta = float(beta)
        self.q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def reset_from_accel(self, accel: np.ndarray) -> None:
        self.q = quat_from_gravity(accel)

    def update(self, gyro: np.ndarray, accel: np.ndarray, dt: float) -> np.ndarray:
        q = self.q
        gyro = np.asarray(gyro, dtype=np.float64).reshape(3)
        accel = np.asarray(accel, dtype=np.float64).reshape(3)
        an = np.linalg.norm(accel)
        if an < 1e-8 or dt <= 0:
            return q
        ax, ay, az = accel / an
        w, x, y, z = q
        f = np.array(
            [
                2 * (x * z - w * y) - ax,
                2 * (w * x + y * z) - ay,
                2 * (0.5 - x * x - y * y) - az,
            ]
        )
        j = np.array(
            [
                [-2 * y, 2 * z, -2 * w, 2 * x],
                [2 * x, 2 * w, 2 * z, 2 * y],
                [0.0, -4 * x, -4 * y, 0.0],
            ]
        )
        step = j.T @ f
        sn = np.linalg.norm(step)
        if sn > 1e-12:
            step = step / sn
        q_dot = _quat_derivative_from_gyro(q, gyro) - self.beta * step
        q = quat_normalize(q + q_dot * dt)
        self.q = q
        return q

    def update_gyro(self, gyro: np.ndarray, dt: float) -> np.ndarray:
        if dt <= 0:
            return self.q
        q_dot = _quat_derivative_from_gyro(self.q, np.asarray(gyro, dtype=np.float64).reshape(3))
        self.q = quat_normalize(self.q + q_dot * dt)
        return self.q


@dataclass
class IMUSample:
    t: float
    accel: np.ndarray  # specific force, m/s^2, IMU/RDF frame
    gyro: np.ndarray  # rad/s
    quat_body_to_world: Optional[np.ndarray] = None  # [w,x,y,z] if device fused
    linear_accel: Optional[np.ndarray] = None  # gravity-free, IMU frame


@dataclass
class IMUOdometryConfig:
    table_height: float
    optical_height: float = 0.03
    still_time_s: float = 1.0
    gravity: float = G
    constrain_z: bool = True
    constrain_slide_y: bool = True
    zero_velocity_still: bool = True
    accel_bias_alpha: float = 0.0  # leftover for compatibility
    madgwick_beta: float = 0.04
    lock_attitude_after_still: bool = True


@dataclass
class IMUState:
    t: np.ndarray
    quat: np.ndarray  # body -> world, [w,x,y,z]
    position: np.ndarray
    velocity: np.ndarray
    R_align: np.ndarray
    t_align: np.ndarray
    accel_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))
    gyro_bias: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def pose_at_index(self, i: int) -> Pose:
        R = rotation_matrix_from_quat(self.quat[i])
        return Pose(R, self.position[i])


def _split_still(samples: list[IMUSample], still_time_s: float) -> int:
    if not samples:
        return 0
    t0 = samples[0].t
    n = 0
    for s in samples:
        if s.t - t0 <= still_time_s:
            n += 1
        else:
            break
    return max(n, 1)


def integrate_imu(samples: list[IMUSample], cfg: IMUOdometryConfig) -> IMUState:
    """Integrate IMU samples. First output pose is rebased to (0, 0, h+offset)."""
    if len(samples) < 2:
        raise ValueError("Need at least two IMU samples")

    n_still = _split_still(samples, cfg.still_time_s)
    still = samples[:n_still]
    accel_mean = np.mean([s.accel for s in still], axis=0)
    gyro_bias = np.mean([s.gyro for s in still], axis=0)

    filt = Madgwick(beta=cfg.madgwick_beta)
    if still[0].quat_body_to_world is not None:
        q0 = quat_normalize(still[0].quat_body_to_world)
        filt.q = q0
    else:
        filt.reset_from_accel(accel_mean)

    # Specific force at rest should equal -g_world in world, i.e. +g * z_hat.
    # Residual after rotating mean accel into world is accelerometer bias.
    g_world = np.array([0.0, 0.0, -cfg.gravity])
    R0 = rotation_matrix_from_quat(filt.q)
    accel_bias_body = accel_mean - R0.T @ (-g_world)

    ts = np.array([s.t for s in samples], dtype=np.float64)
    qs = np.zeros((len(samples), 4))
    pos = np.zeros((len(samples), 3))
    vel = np.zeros((len(samples), 3))
    qs[0] = filt.q

    for i in range(1, len(samples)):
        dt = samples[i].t - samples[i - 1].t
        if dt <= 0:
            qs[i] = qs[i - 1]
            pos[i] = pos[i - 1]
            vel[i] = vel[i - 1]
            continue
        gyro = samples[i].gyro - gyro_bias
        accel = samples[i].accel - accel_bias_body
        moving = i >= n_still
        if (
            samples[i].quat_body_to_world is not None
            and not (moving and cfg.lock_attitude_after_still)
        ):
            q = quat_normalize(samples[i].quat_body_to_world)
            filt.q = q
        elif moving and cfg.lock_attitude_after_still:
            # Table slide: do not treat linear accel as a change in gravity.
            q = filt.update_gyro(gyro, dt)
        elif abs(np.linalg.norm(accel) - cfg.gravity) > 0.5:
            q = filt.update_gyro(gyro, dt)
        else:
            q = filt.update(gyro, accel, dt)
        qs[i] = q
        R = rotation_matrix_from_quat(q)
        if samples[i].linear_accel is not None:
            a_world = R @ (samples[i].linear_accel - accel_bias_body)
        else:
            a_world = R @ accel + g_world
        if i < n_still and cfg.zero_velocity_still:
            a_world = np.zeros(3)
            vel[i] = np.zeros(3)
            pos[i] = pos[i - 1]
        else:
            vel[i] = vel[i - 1] + a_world * dt
            pos[i] = pos[i - 1] + vel[i - 1] * dt + 0.5 * a_world * dt * dt
            if cfg.constrain_z:
                pos[i, 2] = 0.0
                vel[i, 2] = 0.0
            if cfg.constrain_slide_y:
                # Weakly kill sideways drift; vision later recovers Y object range.
                vel[i, 1] *= 0.5
                pos[i, 1] *= 0.2

    # Rebase so sample 0 optical center is (0, 0, h+offset) with nominal axes.
    R_first = rotation_matrix_from_quat(qs[0])
    R_align, t_align = rebase_first_pose(
        R_first, pos[0], cfg.table_height, cfg.optical_height
    )
    pos_a = np.zeros_like(pos)
    qs_a = np.zeros_like(qs)
    for i in range(len(samples)):
        Ri = rotation_matrix_from_quat(qs[i])
        Rn, tn = apply_rebase(R_align, t_align, Ri, pos[i])
        pos_a[i] = tn
        qs_a[i] = quat_from_rotation_matrix(Rn)
        if cfg.constrain_z:
            pos_a[i, 2] = cfg.table_height + cfg.optical_height

    return IMUState(
        t=ts,
        quat=qs_a,
        position=pos_a,
        velocity=vel,
        R_align=R_align,
        t_align=t_align,
        accel_bias=accel_bias_body,
        gyro_bias=gyro_bias,
    )


def reported_camera_xyz(state: IMUState, table_height: float) -> np.ndarray:
    """Part A convention: first frame (0,0,h), z held at table height."""
    xyz = state.position.copy()
    xyz[:, 2] = table_height
    xyz[0] = np.array([0.0, 0.0, table_height])
    return xyz


def poses_at_times(state: IMUState, times: np.ndarray) -> list[Pose]:
    from oak_track.geometry import interpolate_quats, interpolate_vector, rotation_matrix_from_quat

    tq = np.asarray(times, dtype=np.float64)
    pos = interpolate_vector(state.t, state.position, tq)
    qs = interpolate_quats(state.t, state.quat, tq)
    poses = []
    for i in range(len(tq)):
        poses.append(Pose(rotation_matrix_from_quat(qs[i]), pos[i]))
    return poses
