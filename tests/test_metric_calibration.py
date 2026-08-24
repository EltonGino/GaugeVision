import pytest

from gaugevision.calibration.metric_calibration import MetricCalibration


def test_pixels_to_mm_roundtrip():
    cal = MetricCalibration(px_per_mm=10.0, source="test")
    assert cal.pixels_to_mm(100.0) == pytest.approx(10.0)
    assert cal.mm_to_pixels(10.0) == pytest.approx(100.0)


def test_from_reference():
    cal = MetricCalibration.from_reference(
        reference_length_px=200.0, reference_length_mm=20.0, source="test_ref"
    )
    assert cal.px_per_mm == pytest.approx(10.0)
    assert cal.source == "test_ref"
    assert cal.validated is False


def test_demo_reference_is_unvalidated():
    cal = MetricCalibration.demo_reference()
    assert cal.source == "demo_reference"
    assert cal.validated is False
    assert cal.px_per_mm > 0


def test_rejects_non_positive_scale():
    with pytest.raises(ValueError):
        MetricCalibration(px_per_mm=0.0, source="bad")
    with pytest.raises(ValueError):
        MetricCalibration(px_per_mm=-5.0, source="bad")


def test_from_reference_rejects_non_positive_inputs():
    with pytest.raises(ValueError):
        MetricCalibration.from_reference(0.0, 10.0, source="bad")
    with pytest.raises(ValueError):
        MetricCalibration.from_reference(10.0, 0.0, source="bad")
