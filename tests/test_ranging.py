import numpy as np

from oak_track.geometry import Pose, R_WORLD_FROM_CAM, camera_matrix, pixel_to_plane, project_point
from oak_track.ranging import GroundPlaneConfig, localize_uv, pose_source_vo_or_imu


def test_pixel_to_plane_recovers_table_point():
    h, z_off = 0.75, 0.03
    K = camera_matrix(420, 420, 319.5, 179.5)
    pose = Pose(R_WORLD_FROM_CAM.copy(), np.array([0.0, 0.0, h + z_off]))
    obj = np.array([0.05, 0.90, h])
    uv = project_point(K, pose.R.T @ (obj - pose.t))
    p = pixel_to_plane(K, pose, uv, h)
    assert p is not None
    np.testing.assert_allclose(p, obj, atol=1e-6)


def test_stereo_behind_table_is_rejected():
    h, z_off = 0.75, 0.03
    K = camera_matrix(420, 420, 319.5, 179.5)
    pose = Pose(R_WORLD_FROM_CAM.copy(), np.array([0.0, 0.0, h + z_off]))
    obj = np.array([0.05, 0.90, h])
    p_cam = pose.R.T @ (obj - pose.t)
    uv = project_point(K, p_cam)
    cfg = GroundPlaneConfig.table(h)
    # Same pixel as the table object, but stereo reports a wall 2.13 m away.
    fix = localize_uv(pose, K, uv, depth_m=2.13, cfg=cfg)
    assert fix.source == "rejected_background"
    assert fix.xyz is None


def test_agreed_stereo_and_plane_are_fused():
    h, z_off = 0.75, 0.03
    K = camera_matrix(420, 420, 319.5, 179.5)
    pose = Pose(R_WORLD_FROM_CAM.copy(), np.array([0.0, 0.0, h + z_off]))
    obj = np.array([0.04, 0.92, h])
    p_cam = pose.R.T @ (obj - pose.t)
    uv = project_point(K, p_cam)
    cfg = GroundPlaneConfig.table(h)
    fix = localize_uv(pose, K, uv, depth_m=float(p_cam[2]), cfg=cfg)
    assert fix.source == "fused"
    np.testing.assert_allclose(fix.xyz, obj, atol=0.01)


def test_vo_preferred_when_camera_slid():
    h = 0.75
    vo = [Pose(R_WORLD_FROM_CAM.copy(), np.array([x, 0.0, h + 0.03])) for x in (0.0, 0.2, 0.4)]
    imu = [Pose(p.R, np.array([0.0, 0.0, p.t[2]])) for p in vo]
    used, src = pose_source_vo_or_imu(imu, vo)
    assert src == "vo"
    assert abs(used[-1].t[0] - 0.4) < 1e-9
