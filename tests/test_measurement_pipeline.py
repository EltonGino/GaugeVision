import cv2
import numpy as np
import pytest

from gaugevision.calibration.metric_calibration import MetricCalibration
from gaugevision.measurement.pipeline import run_measurement_pipeline


def _make_synthetic_screw(
    shank_width: int = 60,
    shank_length: int = 260,
    thread_amplitude: int = 8,
    period_px: int = 20,
    head_width: int = 100,
    head_length: int = 50,
    size: tuple[int, int] = (400, 200),
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Procedurally draws a simple screw-like silhouette: a wider head plus a
    shank whose width oscillates periodically to simulate thread crests."""
    h, w = size
    canvas = np.zeros((h, w), dtype=np.uint8)
    cx = w // 2

    # Head (top)
    top = 10
    cv2.rectangle(
        canvas,
        (cx - head_width // 2, top),
        (cx + head_width // 2, top + head_length),
        255,
        thickness=-1,
    )

    # Threaded shank (below head): width oscillates with period_px
    shank_top = top + head_length
    for row in range(shank_length):
        y = shank_top + row
        if y >= h:
            break
        local_width = shank_width + thread_amplitude * np.sin(2 * np.pi * row / period_px)
        half = int(local_width / 2)
        cv2.line(canvas, (cx - half, y), (cx + half, y), 255, 1)

    if angle_deg != 0.0:
        rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        canvas = cv2.warpAffine(canvas, rot_mat, (w, h))

    return canvas


def test_measurement_pipeline_runs_end_to_end_on_synthetic_screw():
    image = _make_synthetic_screw()
    calibration = MetricCalibration.demo_reference(px_per_mm=32.4)

    result = run_measurement_pipeline(image, calibration=calibration)

    assert result.major_diameter_px > 0
    assert result.major_diameter_mm == pytest.approx(
        result.major_diameter_px / 32.4
    )
    assert result.calibration_source == "demo_reference"
    assert result.calibrated is False
    assert 0.0 <= result.confidence <= 1.0

    if result.pitch_px is not None:
        # Known synthetic pitch is 20px; peak-spacing should be in the ballpark.
        assert 10 < result.pitch_px < 30
        assert result.pitch_mm == pytest.approx(result.pitch_px / 32.4)


def test_measurement_pipeline_handles_rotated_synthetic_screw():
    image = _make_synthetic_screw(angle_deg=15.0)
    calibration = MetricCalibration.demo_reference(px_per_mm=32.4)

    result = run_measurement_pipeline(image, calibration=calibration)
    assert result.major_diameter_px > 0


def test_measurement_pipeline_raises_on_blank_image():
    blank = np.zeros((100, 100), dtype=np.uint8)
    calibration = MetricCalibration.demo_reference()
    with pytest.raises(RuntimeError):
        run_measurement_pipeline(blank, calibration=calibration)
