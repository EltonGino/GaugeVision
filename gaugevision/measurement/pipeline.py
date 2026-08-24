"""Measurement pipeline orchestration (CLAUDE.md §4.2).

Wires the fixed stage sequence:
image -> lens correction (optional, §4.1a) -> ROI/segmentation -> axis
estimation -> derotation -> thread-region isolation -> profile extraction ->
major-diameter estimation -> pitch estimation -> metric calibration (§4.1b)
-> MeasurementResult.
"""

from __future__ import annotations

import logging

import numpy as np

from gaugevision.calibration.metric_calibration import MetricCalibration
from gaugevision.measurement import axis as axis_mod
from gaugevision.measurement import diameter as diameter_mod
from gaugevision.measurement import profile as profile_mod
from gaugevision.measurement import segment as segment_mod
from gaugevision.measurement.pitch import PeakPitchEstimator, PitchEstimator
from gaugevision.measurement.types import MeasurementResult

logger = logging.getLogger(__name__)


def run_measurement_pipeline(
    image: np.ndarray,
    calibration: MetricCalibration,
    pitch_estimator: PitchEstimator | None = None,
    head_margin_fraction: float = 0.22,
) -> MeasurementResult:
    """Run the full classical-CV measurement pipeline on a single screw image.

    Args:
        image: grayscale or BGR image containing one screw.
        calibration: MetricCalibration used for every px->mm conversion.
            Phase 1 callers pass ``MetricCalibration.demo_reference()``.
        pitch_estimator: strategy to use; defaults to ``PeakPitchEstimator``
            (Phase-1 default per CLAUDE.md §4.2).
        head_margin_fraction: fraction of part length excluded from the
            widest (head) end when isolating the thread-bearing region.

    Returns:
        MeasurementResult with px and mm values, calibration status, and a
        confidence score.
    """
    pitch_estimator = pitch_estimator or PeakPitchEstimator()
    notes: list[str] = []

    segmentation = segment_mod.segment_screw(image)
    axis_estimate = axis_mod.estimate_axis(segmentation)
    _, rotated_mask = axis_mod.derotate(image, segmentation.mask, axis_estimate)

    width_profile = profile_mod.compute_width_profile(rotated_mask)
    thread_region = profile_mod.isolate_thread_region(
        width_profile, head_margin_fraction=head_margin_fraction
    )
    thread_profile = profile_mod.extract_thread_profile(width_profile, thread_region)

    diameter_estimate = diameter_mod.estimate_major_diameter(thread_profile)
    pitch_estimate = pitch_estimator.estimate(thread_profile)
    notes.extend(pitch_estimate.notes)

    if pitch_estimate.pitch_px is None:
        notes.append("pitch could not be reliably estimated for this image")

    major_diameter_mm = calibration.pixels_to_mm(diameter_estimate.major_diameter_px)
    pitch_mm = (
        calibration.pixels_to_mm(pitch_estimate.pitch_px)
        if pitch_estimate.pitch_px is not None
        else None
    )

    if not calibration.validated:
        notes.append(
            f"calibration source '{calibration.source}' is a Phase-1 demonstration "
            "reference, not a dimensionally validated scale — mm values are "
            "illustrative, see CLAUDE.md §3.2/§4.1b"
        )

    return MeasurementResult(
        major_diameter_px=diameter_estimate.major_diameter_px,
        major_diameter_mm=major_diameter_mm,
        pitch_px=pitch_estimate.pitch_px,
        pitch_mm=pitch_mm,
        scale_px_per_mm=calibration.px_per_mm,
        calibration_source=calibration.source,
        calibrated=calibration.validated,
        confidence=pitch_estimate.confidence,
        notes=notes,
    )
