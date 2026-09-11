from pathlib import Path

import numpy as np
import pytest

from oak_track.io_utils import read_frame_xyz_csv, read_json, run_paths
from oak_track.pipeline import capture_then_process, process_run
from oak_track.simulate import simulate_run


def test_simulate_and_process(tmp_path: Path):
    h = 0.75
    obj = (0.04, 0.92)
    run = tmp_path / "run"
    simulate_run(
        run,
        table_height=h,
        optical_height=0.03,
        object_xy=obj,
        slide_m=0.35,
        still_s=0.6,
        slide_s=1.6,
        fps=15,
        imu_hz=100,
        width=480,
        height=270,
        fx=320.0,
    )
    summary = process_run(run, table_height=h, optical_height=0.03, detector="hsv", write_preview=False)
    fused = np.array(summary["object_fused_xyz"])
    assert fused[2] == pytest.approx(h)
    # Planimetric error on the table. Depth (Y) should be especially tight.
    assert abs(fused[1] - obj[1]) < 0.06
    assert abs(fused[0] - obj[0]) < 0.08
    assert summary["n_object_detections"] >= 10

    paths = run_paths(run)
    frames, cam = read_frame_xyz_csv(paths["camera_imu"])
    assert frames[0] == 0
    np.testing.assert_allclose(cam[0], [0.0, 0.0, h], atol=1e-6)
    np.testing.assert_allclose(cam[:, 2], h)
    text = paths["camera_imu"].read_text().splitlines()[0]
    assert text == "frame_number,xyz"
    _, obj_csv = read_frame_xyz_csv(paths["object"])
    np.testing.assert_allclose(obj_csv[:, 2], h)
    assert paths["cloud"].is_file()
    assert summary["cloud_n_points"] > 50
    gt = read_json(paths["ground_truth"])
    assert gt["object_xyz"][1] == pytest.approx(obj[1])


def test_capture_then_process_simulate(tmp_path: Path):
    h = 0.75
    out = tmp_path / "combined"
    summary = capture_then_process(
        out_dir=out,
        table_height=h,
        optical_height=0.03,
        detector="hsv",
        write_preview=False,
        simulate=True,
        object_xy=(0.04, 0.92),
        slide_m=0.30,
        still_s=0.5,
        slide_s=1.2,
    )
    assert (out / "frames.csv").is_file()
    assert (out / "camera_imu.csv").is_file()
    assert (out / "object.csv").is_file()
    assert (out / "cloud.ply").is_file()
    fused = np.array(summary["object_fused_xyz"])
    assert fused[2] == pytest.approx(h)
    assert summary["n_object_detections"] >= 8
