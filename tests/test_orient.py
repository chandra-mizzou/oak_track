import numpy as np

from oak_track.orient import (
    apply_invert_vec,
    detect_inverted_from_accel,
    rotate_k_180,
)


def test_detect_inverted_y_up():
    assert detect_inverted_from_accel(np.array([0.1, 9.7, 0.2])) is True
    assert detect_inverted_from_accel(np.array([0.1, -9.7, 0.2])) is False


def test_invert_accel_flips_xy():
    v = np.array([1.0, 2.0, 3.0])
    out = apply_invert_vec(v)
    np.testing.assert_allclose(out, [-1.0, -2.0, 3.0])


def test_rotate_k_180():
    K = np.array([[700.0, 0.0, 640.0], [0.0, 700.0, 360.0], [0.0, 0.0, 1.0]])
    Kn = rotate_k_180(K, 1280, 720)
    np.testing.assert_allclose(Kn[0, 2], 1280 - 1 - 640)
    np.testing.assert_allclose(Kn[1, 2], 720 - 1 - 360)
    np.testing.assert_allclose(Kn[0, 0], 700.0)
