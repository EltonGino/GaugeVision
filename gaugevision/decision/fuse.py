"""Go/No-Go decision fusion — CLAUDE.md §4.5.

"ISO-informed dimensional validation," not "ISO 965 compliance verification":
measuring major diameter from an image is not a complete ISO 965 conformity
check (pitch diameter and full thread-profile geometry are also part of the
standard and aren't covered here).
"""

from __future__ import annotations

from pydantic import BaseModel

from gaugevision.anomaly.types import AnomalyResult
from gaugevision.decision.iso965_table import (
    ISO_965_TOLERANCE_TABLE,
    nearest_designation,
)
from gaugevision.measurement.types import MeasurementResult

ISO_DISCLAIMER = (
    "The tolerance table is used to demonstrate standards-aware decision logic "
    "and does not constitute certified dimensional inspection or complete "
    "ISO 965 compliance verification."
)


class InspectionVerdict(BaseModel):
    verdict: str  # "GO" | "NO_GO"
    anomaly_score: float
    anomaly_threshold: float
    measurements: dict[str, float | None]
    failed_checks: list[str]
    reasoning: list[str]
    calibration_source: str
    calibrated: bool
    measurement_confidence: float
    matched_thread_designation: str | None
    inference_ms: float


def fuse_verdict(
    anomaly: AnomalyResult,
    measurement: MeasurementResult,
    inference_ms: float,
    thread_designation: str | None = None,
) -> InspectionVerdict:
    """Combine anomaly detection and dimensional measurement into an
    explainable Go/No-Go verdict.

    Args:
        anomaly: fitted-model anomaly score/heatmap/threshold for this image.
        measurement: MeasurementResult from the measurement pipeline.
        inference_ms: total wall-clock inference time for this inspection.
        thread_designation: nominal thread size (e.g. "M6") to check against.
            If omitted, the nearest starter-table entry by measured major
            diameter is used — MVTec images carry no nominal-size label.
    """
    failed_checks: list[str] = []
    reasoning: list[str] = []

    if anomaly.score > anomaly.threshold:
        failed_checks.append("anomaly_score")
        reasoning.append(
            f"anomaly score {anomaly.score:.3f} exceeds threshold {anomaly.threshold:.3f}"
        )
    else:
        reasoning.append(
            f"anomaly score {anomaly.score:.3f} within threshold {anomaly.threshold:.3f}"
        )

    designation = thread_designation or nearest_designation(measurement.major_diameter_mm)
    entry = ISO_965_TOLERANCE_TABLE[designation]
    diameter = measurement.major_diameter_mm
    if not (entry.major_diameter_min_mm <= diameter <= entry.major_diameter_max_mm):
        failed_checks.append("major_diameter")
        reasoning.append(
            f"major diameter {diameter:.3f}mm outside {designation} class-"
            f"{entry.tolerance_class} limits "
            f"[{entry.major_diameter_min_mm:.3f}, {entry.major_diameter_max_mm:.3f}]mm"
        )
    else:
        reasoning.append(
            f"major diameter {diameter:.3f}mm within {designation} class-"
            f"{entry.tolerance_class} limits "
            f"[{entry.major_diameter_min_mm:.3f}, {entry.major_diameter_max_mm:.3f}]mm"
        )

    if not measurement.calibrated:
        reasoning.append(
            f"dimensional check used calibration source '{measurement.calibration_source}' "
            "(Phase-1 demonstration reference, not dimensionally validated) — "
            "treat the dimensional check as illustrative, not certified"
        )

    reasoning.append(ISO_DISCLAIMER)

    verdict = "NO_GO" if failed_checks else "GO"

    return InspectionVerdict(
        verdict=verdict,
        anomaly_score=anomaly.score,
        anomaly_threshold=anomaly.threshold,
        measurements={
            "major_diameter_mm": measurement.major_diameter_mm,
            "pitch_mm": measurement.pitch_mm,
        },
        failed_checks=failed_checks,
        reasoning=reasoning,
        calibration_source=measurement.calibration_source,
        calibrated=measurement.calibrated,
        measurement_confidence=measurement.confidence,
        matched_thread_designation=designation,
        inference_ms=inference_ms,
    )
