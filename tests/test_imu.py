import numpy as np

from oak_track.geometry import G, R_WORLD_FROM_CAM
from oak_track.imu import IMUOdometryConfig, IMUSample, Madgwick, integrate_imu, reported_camera_xyz
from oak_track.simulate import camera_motion


def test_madgwick_stationary():
    filt = Madgwick(beta=0.1)
    filt.reset_from_accel(np.array([0.0, 0.0, G]))
    q0 = filt.q.copy()
    for _ in range(200):
        filt.update(np.zeros(3), np.array([0.0, 0.0, G]), 0.005)
    np.testing.assert_allclose(filt.q, q0, atol=1e-4)


def test_rest_stays_at_origin():
    n = 400
    dt = 0.005
    samples = []
    a_body = R_WORLD_FROM_CAM.T @ np.array([0.0, 0.0, G])
    for i in range(n):
        samples.append(
            IMUSample(
                t=i * dt,
                accel=a_body,
                gyro=np.zeros(3),
            )
        )
    cfg = IMUOdometryConfig(table_height=0.75, optical_height=0.03, still_time_s=2.0)
    state = integrate_imu(samples, cfg)
    xyz = reported_camera_xyz(state, 0.75)
    np.testing.assert_allclose(xyz[0], [0.0, 0.0, 0.75])
    np.testing.assert_allclose(xyz[-1], [0.0, 0.0, 0.75], atol=1e-3)


def test_slide_recovers_distance():
    dt = 0.005
    still, slide, dist = 1.0, 2.0, 0.40
    t = np.arange(0.0, still + slide + 0.2, dt)
    x, ax = camera_motion(t, still, slide, dist)
    samples = []
    for i, ti in enumerate(t):
        a_world = np.array([ax[i], 0.0, 0.0])
        f_world = a_world - np.array([0.0, 0.0, -G])
        a_body = R_WORLD_FROM_CAM.T @ f_world
        samples.append(IMUSample(t=float(ti), accel=a_body, gyro=np.zeros(3)))
    cfg = IMUOdometryConfig(table_height=0.80, optical_height=0.0, still_time_s=still)
    state = integrate_imu(samples, cfg)
    xyz = reported_camera_xyz(state, 0.80)
    np.testing.assert_allclose(xyz[0], [0.0, 0.0, 0.80], atol=1e-6)
    assert abs(xyz[-1, 0] - dist) < 0.04
    assert abs(xyz[-1, 2] - 0.80) < 1e-9
