import numpy as np
import pytest

from gaugevision.calibration.metric_calibration import MetricCalibration
from gaugevision.data.synthetic_thread import (
    SyntheticThreadSpec,
    generate_synthetic_thread,
)
from gaugevision.measurement.pipeline import run_measurement_pipeline


def test_generate_synthetic_thread_produces_binary_image():
    spec = SyntheticThreadSpec(major_diameter_mm=6.0, minor_diameter_mm=5.0, pitch_mm=1.0)
    sample = generate_synthetic_thread(spec, px_per_mm=20.0, condition="clean")
    assert sample.image.dtype == np.uint8
    assert np.count_nonzero(sample.image) > 0
    assert sample.px_per_mm == 20.0
    assert sample.reference_bar_length_px == 200.0  # 10mm * 20px/mm


def test_generate_synthetic_thread_rejects_unknown_condition():
    spec = SyntheticThreadSpec(major_diameter_mm=6.0, minor_diameter_mm=5.0, pitch_mm=1.0)
    try:
        generate_synthetic_thread(spec, condition="bogus")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_synthetic_reference_calibration_recovers_true_scale():
    spec = SyntheticThreadSpec(major_diameter_mm=6.0, minor_diameter_mm=5.0, pitch_mm=1.0)
    sample = generate_synthetic_thread(spec, px_per_mm=20.0, condition="clean")
    calibration = MetricCalibration.synthetic_reference(
        sample.reference_bar_length_px, sample.reference_bar_length_mm
    )
    assert calibration.px_per_mm == pytest.approx(sample.px_per_mm, rel=1e-6)
    assert calibration.validated is True
    assert calibration.source == "synthetic_reference"


def test_measurement_pipeline_accurate_on_clean_synthetic_thread():
    spec = SyntheticThreadSpec(major_diameter_mm=6.0, minor_diameter_mm=5.0, pitch_mm=1.0)
    sample = generate_synthetic_thread(spec, px_per_mm=20.0, condition="clean")
    calibration = MetricCalibration.synthetic_reference(
        sample.reference_bar_length_px, sample.reference_bar_length_mm
    )
    result = run_measurement_pipeline(sample.image, calibration=calibration)

    assert abs(result.major_diameter_mm - spec.major_diameter_mm) < 0.2
    assert result.pitch_mm is not None
    assert abs(result.pitch_mm - spec.pitch_mm) < 0.1
    assert result.confidence > 0.8


def test_measurement_pipeline_reasonably_accurate_under_blur_and_rotation():
    spec = SyntheticThreadSpec(major_diameter_mm=6.0, minor_diameter_mm=5.0, pitch_mm=1.0)
    for condition in ("blur", "rotated"):
        sample = generate_synthetic_thread(spec, px_per_mm=20.0, condition=condition)
        calibration = MetricCalibration.synthetic_reference(
            sample.reference_bar_length_px, sample.reference_bar_length_mm
        )
        result = run_measurement_pipeline(sample.image, calibration=calibration)
        assert abs(result.major_diameter_mm - spec.major_diameter_mm) < 0.3, condition
