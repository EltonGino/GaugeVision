import numpy as np
import pytest

from gaugevision.calibration.metric_calibration import MetricCalibration
from gaugevision.measurement.pipeline import run_measurement_pipeline
from tests.synth import make_synthetic_screw as _make_synthetic_screw


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
