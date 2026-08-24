import numpy as np

from gaugevision.anomaly.types import AnomalyResult
from gaugevision.decision.fuse import fuse_verdict
from gaugevision.decision.iso965_table import lookup_by_designation, nearest_designation
from gaugevision.measurement.types import MeasurementResult

M6 = lookup_by_designation("M6")


def _measurement(diameter_mm: float, pitch_mm: float | None = 1.0, confidence: float = 0.9):
    return MeasurementResult(
        major_diameter_px=diameter_mm * 32.4,
        major_diameter_mm=diameter_mm,
        pitch_px=pitch_mm * 32.4 if pitch_mm else None,
        pitch_mm=pitch_mm,
        scale_px_per_mm=32.4,
        calibration_source="demo_reference",
        calibrated=False,
        confidence=confidence,
    )


def _anomaly(score: float, threshold: float = 0.5):
    return AnomalyResult(
        score=score, heatmap=np.zeros((4, 4), dtype=np.float32), threshold=threshold,
        is_anomalous=score > threshold,
    )


def test_nearest_designation_matches_exact_size():
    assert nearest_designation(6.0) == "M6"
    assert nearest_designation(3.0) == "M3"


def test_go_when_within_tolerance_and_low_anomaly_score():
    measurement = _measurement(diameter_mm=(M6.major_diameter_min_mm + M6.major_diameter_max_mm) / 2)
    verdict = fuse_verdict(_anomaly(0.1), measurement, inference_ms=10.0, thread_designation="M6")
    assert verdict.verdict == "GO"
    assert verdict.failed_checks == []


def test_no_go_on_high_anomaly_score():
    measurement = _measurement(diameter_mm=(M6.major_diameter_min_mm + M6.major_diameter_max_mm) / 2)
    verdict = fuse_verdict(_anomaly(0.9), measurement, inference_ms=10.0, thread_designation="M6")
    assert verdict.verdict == "NO_GO"
    assert "anomaly_score" in verdict.failed_checks


def test_no_go_on_diameter_out_of_tolerance():
    measurement = _measurement(diameter_mm=M6.major_diameter_max_mm + 0.5)
    verdict = fuse_verdict(_anomaly(0.1), measurement, inference_ms=10.0, thread_designation="M6")
    assert verdict.verdict == "NO_GO"
    assert "major_diameter" in verdict.failed_checks


def test_both_checks_can_fail_simultaneously():
    measurement = _measurement(diameter_mm=M6.major_diameter_max_mm + 0.5)
    verdict = fuse_verdict(_anomaly(0.9), measurement, inference_ms=10.0, thread_designation="M6")
    assert set(verdict.failed_checks) == {"anomaly_score", "major_diameter"}


def test_verdict_notes_uncalibrated_source():
    measurement = _measurement(diameter_mm=6.0)
    verdict = fuse_verdict(_anomaly(0.1), measurement, inference_ms=10.0, thread_designation="M6")
    assert any("demonstration reference" in r for r in verdict.reasoning)
    assert verdict.calibrated is False
