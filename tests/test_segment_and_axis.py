import cv2
import numpy as np
import pytest

from gaugevision.measurement.axis import derotate, estimate_axis
from gaugevision.measurement.segment import segment_screw


def _make_rotated_bar_image(angle_deg: float, length: int = 200, width: int = 30, size: int = 300):
    """A bright rectangle on a dark background, rotated by angle_deg."""
    canvas = np.zeros((size, size), dtype=np.uint8)
    center = (size // 2, size // 2)
    rect = ((center[0], center[1]), (width, length), angle_deg)
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(canvas, [box], 255)
    return canvas


def test_segment_screw_finds_bright_foreground():
    image = _make_rotated_bar_image(angle_deg=0.0)
    result = segment_screw(image)
    assert result.mask.shape == image.shape
    assert np.count_nonzero(result.mask) > 0
    # Foreground should roughly match the drawn rectangle area (~30*200)
    assert 4500 < np.count_nonzero(result.mask) < 7500


def test_segment_screw_raises_on_empty_image():
    blank = np.zeros((100, 100), dtype=np.uint8)
    with pytest.raises(RuntimeError):
        segment_screw(blank)


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 60.0, -45.0])
def test_estimate_axis_matches_known_rotation(angle_deg):
    image = _make_rotated_bar_image(angle_deg=angle_deg)
    seg = segment_screw(image)
    axis = estimate_axis(seg)
    # PCA principal axis is ambiguous mod 180 degrees; normalize both to
    # the [0, 180) range before comparing to the drawn rectangle's long axis
    # (cv2's boxPoints angle convention means the long axis is angle+90).
    expected = (angle_deg + 90.0) % 180.0
    actual = axis.angle_deg % 180.0
    diff = min(abs(actual - expected), 180.0 - abs(actual - expected))
    assert diff < 5.0


def test_derotate_produces_vertical_axis():
    image = _make_rotated_bar_image(angle_deg=25.0)
    seg = segment_screw(image)
    axis = estimate_axis(seg)
    _, rotated_mask = derotate(image, seg.mask, axis)

    rotated_seg_like = type(seg)(mask=rotated_mask, bbox=(0, 0, 0, 0))
    new_axis = estimate_axis(rotated_seg_like)
    # After derotation the axis should be close to vertical (90 degrees)
    normalized = new_axis.angle_deg % 180.0
    assert min(abs(normalized - 90.0), abs(normalized - 270.0)) < 5.0
