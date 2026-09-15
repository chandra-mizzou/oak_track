import numpy as np

from oak_track.geometry import Pose, R_WORLD_FROM_CAM, backproject, camera_matrix, project_point
from oak_track.localize import fuse_object, refine_slide_and_object


def _slide_scene(n=40, h=0.75, z_off=0.03, obj=None, slide=0.40):
    if obj is None:
        obj = np.array([0.05, 0.90, h])
    K = camera_matrix(420, 420, 319.5, 179.5)
    xs = np.linspace(0.0, slide, n)
    poses, uvs, depths = [], [], []
    for x in xs:
        pose = Pose(R_WORLD_FROM_CAM.copy(), np.array([x, 0.0, h + z_off]))
        p_cam = pose.R.T @ (obj - pose.t)
        uvs.append(project_point(K, p_cam))
        depths.append(p_cam[2])
        poses.append(pose)
    return poses, np.asarray(uvs), np.asarray(depths), K, obj


def test_refine_recovers_object():
    h, z_off = 0.75, 0.03
    poses, uvs, depths, K, obj = _slide_scene(n=50, h=h, z_off=z_off)
    # Perturb IMU/VO camera x so the optimiser has work to do.
    noisy = []
    rng = np.random.default_rng(1)
    for p in poses:
        t = p.t.copy()
        t[0] += rng.normal(0, 0.01)
        noisy.append(Pose(p.R, t))
    est = refine_slide_and_object(noisy, uvs, depths, K, h, z_off)
    np.testing.assert_allclose(est.xyz, obj, atol=0.015)
    assert est.n_used == 50
    assert est.rms_reproj_px < 2.0


def test_fuse_prefers_vo_when_imu_fails():
    h, z_off = 0.75, 0.03
    vo, uvs, depths, K, obj = _slide_scene(n=30, h=h, z_off=z_off, slide=0.35)
    imu = [Pose(p.R, np.array([0.0, 0.0, p.t[2]])) for p in vo]
    est, used = fuse_object(imu, vo, uvs, depths, K, h, z_off)
    np.testing.assert_allclose(est.xyz, obj, atol=0.02)
    assert used[0].t[0] == 0.0
    assert abs(used[-1].t[0] - 0.35) < 1e-6


def test_backproject_matches_depth():
    K = camera_matrix(400, 400, 320, 180)
    p = backproject(K, np.array([320.0, 180.0]), 0.9)
    np.testing.assert_allclose(p, [0.0, 0.0, 0.9])
