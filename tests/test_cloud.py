from pathlib import Path

import numpy as np

from oak_track.cloud import (
    fuse_dense_cloud,
    read_ply_xyzrgb,
    voxel_downsample,
    write_ply_xyzrgb,
)
from oak_track.geometry import Pose, R_WORLD_FROM_CAM, camera_matrix
from oak_track.io_utils import read_json
from oak_track.pipeline import process_run
from oak_track.simulate import simulate_run


def test_voxel_downsample_averages_duplicates():
    xyz = np.array([[0.001, 0.0, 0.0], [0.002, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    rgb = np.array([[10, 0, 0], [30, 0, 0], [0, 50, 0]], dtype=np.uint8)
    out_xyz, out_rgb = voxel_downsample(xyz, rgb, voxel_m=0.01)
    assert len(out_xyz) == 2
    xs = np.sort(out_xyz[:, 0])
    np.testing.assert_allclose(xs[0], 0.0015, atol=1e-12)
    np.testing.assert_allclose(xs[1], 1.0, atol=1e-12)
    first = int(np.argmin(out_xyz[:, 0]))
    assert out_rgb[first, 0] == 20


def test_ply_roundtrip(tmp_path: Path):
    xyz = np.array([[0.1, 0.2, 0.75], [-0.3, 1.1, 0.74]], dtype=np.float64)
    rgb = np.array([[255, 10, 0], [0, 128, 64]], dtype=np.uint8)
    path = tmp_path / "cloud.ply"
    write_ply_xyzrgb(path, xyz, rgb)
    xyz2, rgb2 = read_ply_xyzrgb(path)
    np.testing.assert_allclose(xyz2, xyz, atol=1e-6)
    np.testing.assert_array_equal(rgb2, rgb)


def test_backproject_plane_into_world():
    K = camera_matrix(200.0, 200.0, 20.0, 15.0)
    h, w = 31, 41
    depth = np.full((h, w), 1000, dtype=np.uint16)  # 1 m
    color = np.zeros((h, w, 3), dtype=np.uint8)
    color[:, :] = (0, 0, 200)  # BGR red
    pose = Pose(R_WORLD_FROM_CAM.copy(), np.array([0.0, 0.0, 0.78]))
    xyz, rgb, meta = fuse_dense_cloud(
        [color],
        [depth],
        [pose],
        K,
        pixel_stride=1,
        voxel_m=0.0,
        depth_min_m=0.2,
        depth_max_m=2.0,
    )
    assert meta["n_points"] == h * w
    # Optical axis: camera z_cam=1 → world +Y.
    mid = xyz[(h // 2) * w + (w // 2)]
    np.testing.assert_allclose(mid, [0.0, 1.0, 0.78], atol=0.02)
    assert np.all(rgb[:, 0] == 200)


def test_simulate_process_writes_cloud(tmp_path: Path):
    h = 0.75
    run = tmp_path / "run"
    simulate_run(
        run,
        table_height=h,
        optical_height=0.03,
        object_xy=(0.04, 0.92),
        slide_m=0.30,
        still_s=0.4,
        slide_s=1.0,
        fps=12,
        imu_hz=80,
        width=320,
        height=180,
        fx=240.0,
    )
    summary = process_run(run, table_height=h, optical_height=0.03, detector="hsv", write_preview=False)
    ply = run / "cloud.ply"
    assert ply.is_file()
    assert (run / "cloud.json").is_file()
    assert (run / "cloud_preview.png").is_file()
    xyz, rgb = read_ply_xyzrgb(ply)
    assert len(xyz) > 50
    assert summary["cloud_n_points"] == len(xyz)
    # Simulated depth is on the table plane; fused Z should cluster near h.
    assert abs(float(np.median(xyz[:, 2])) - h) < 0.08
    meta = read_json(run / "cloud.json")
    assert meta["n_frames_used"] >= 5
    assert rgb.shape == xyz.shape
