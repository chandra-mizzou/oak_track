import numpy as np

from oak_track.vision import ObjectTracker, detect_hsv


def _canvas_with_red_boxes(boxes, size=(180, 320)):
    h, w = size
    img = np.full((h, w, 3), 40, dtype=np.uint8)
    img[:, :, 0] = 50
    img[:, :, 1] = 90
    img[:, :, 2] = 40
    for x, y, bw, bh in boxes:
        img[y : y + bh, x : x + bw] = (0, 0, 255)
    return img


def test_hsv_picks_largest_when_unlocked():
    img = _canvas_with_red_boxes([(20, 40, 20, 20), (200, 40, 60, 60)])
    det = detect_hsv(img, None)
    assert det is not None
    assert det.uv[0] > 180


def test_lock_first_ignores_later_larger_blob():
    small = (30, 50, 24, 24)
    large = (220, 50, 70, 70)
    tracker = ObjectTracker(mode="hsv", lock_first=True)
    d0 = tracker.detect(_canvas_with_red_boxes([small]), None)
    assert d0 is not None
    assert d0.uv[0] < 80
    d1 = tracker.detect(_canvas_with_red_boxes([small, large]), None)
    assert d1 is not None
    assert d1.uv[0] < 80


def test_no_lock_first_jumps_to_larger_blob():
    small = (30, 50, 24, 24)
    large = (220, 50, 70, 70)
    tracker = ObjectTracker(mode="hsv", lock_first=False)
    d0 = tracker.detect(_canvas_with_red_boxes([small]), None)
    assert d0 is not None
    d1 = tracker.detect(_canvas_with_red_boxes([small, large]), None)
    assert d1 is not None
    assert d1.uv[0] > 180


def test_lock_follows_same_object_when_it_shrinks():
    tracker = ObjectTracker(mode="hsv", lock_first=True)
    d0 = tracker.detect(_canvas_with_red_boxes([(40, 40, 50, 50)]), None)
    assert d0 is not None
    d1 = tracker.detect(_canvas_with_red_boxes([(48, 48, 18, 18), (240, 40, 80, 80)]), None)
    assert d1 is not None
    assert d1.uv[0] < 100
