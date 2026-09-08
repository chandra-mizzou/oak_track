import numpy as np

from oak_track.geometry import (
    Pose,
    R_WORLD_FROM_CAM,
    backproject,
    camera_matrix,
    nominal_cam_to_world,
    project_point,
    quat_from_gravity,
    rotation_matrix_from_quat,
    world_from_cam_point,
)
from oak_track.io_utils import format_xyz, parse_xyz, write_frame_xyz_csv, read_frame_xyz_csv


def test_cam_forward_is_world_y():
    p_cam = np.array([0.0, 0.0, 1.0])
    p_w = R_WORLD_FROM_CAM @ p_cam
    np.testing.assert_allclose(p_w, [0.0, 1.0, 0.0])


def test_cam_down_is_world_minus_z():
    p_cam = np.array([0.0, 1.0, 0.0])
    p_w = R_WORLD_FROM_CAM @ p_cam
    np.testing.assert_allclose(p_w, [0.0, 0.0, -1.0])


def test_backproject_project_roundtrip():
    K = camera_matrix(400, 400, 320, 180)
    p = np.array([0.1, -0.05, 0.9])
    uv = project_point(K, p)
    p2 = backproject(K, uv, 0.9)
    np.testing.assert_allclose(p, p2, atol=1e-10)


def test_world_from_cam_at_origin():
    pose = nominal_cam_to_world(np.array([0.0, 0.0, 0.75]))
    p_cam = np.array([0.05, 0.02, 0.90])
    p_w = world_from_cam_point(pose, p_cam)
    expected = pose.R @ p_cam + pose.t
    np.testing.assert_allclose(p_w, expected)
    assert abs(p_w[2] - 0.73) < 1e-9  # 0.75 - 0.02 (cam y down)


def test_quat_from_gravity_identity():
    q = quat_from_gravity(np.array([0.0, 0.0, 9.81]))
    R = rotation_matrix_from_quat(q)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-6)


def test_csv_two_columns(tmp_path):
    path = tmp_path / "cam.csv"
    xyz = np.array([[0.0, 0.0, 0.75], [0.1, 0.0, 0.75]])
    write_frame_xyz_csv(path, [0, 1], xyz)
    text = path.read_text().splitlines()
    assert text[0] == "frame_number,xyz"
    assert "(0.000000, 0.000000, 0.750000)" in text[1]
    frames, got = read_frame_xyz_csv(path)
    np.testing.assert_array_equal(frames, [0, 1])
    np.testing.assert_allclose(got, xyz)
    np.testing.assert_allclose(parse_xyz(format_xyz([1, 2, 3])), [1, 2, 3])
