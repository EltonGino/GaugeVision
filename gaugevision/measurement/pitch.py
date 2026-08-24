"""Pitch estimation — measurement pipeline steps 6-7 (CLAUDE.md §4.2).

Pitch estimation is implemented behind a common ``PitchEstimator`` interface
with two competing strategies, per CLAUDE.md:

- ``PeakPitchEstimator`` (peak-to-peak spacing of thread crests): the
  Phase-1 default, fully implemented and used by the pipeline.
- ``FFTPitchEstimator`` (periodicity via FFT of the 1D signal): the
  interface exists from Phase 1 so nothing needs restructuring later, but
  per CLAUDE.md's Phase-1 scope guardrail ("only one pitch-estimation
  strategy needs to be fully functional in Phase 1"), it is intentionally
  left unimplemented here. Phase 4 implements it and produces the
  quantitative Peak-vs-FFT comparison table against synthetic ground truth.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from scipy.signal import find_peaks

from gaugevision.measurement.types import PitchEstimate, ThreadProfile


class PitchEstimator(ABC):
    @abstractmethod
    def estimate(self, profile: ThreadProfile) -> PitchEstimate: ...


class PeakPitchEstimator(PitchEstimator):
    """Estimates pitch from the median spacing between thread-crest peaks.

    Confidence is derived from the coefficient of variation of peak spacings:
    consistent spacing (low CV) -> high confidence; irregular spacing (noisy
    segmentation, too few peaks) -> low confidence. This directly feeds
    ``MeasurementResult.confidence`` and the "flag when the estimate isn't
    reliable" requirement in CLAUDE.md §4.2 step 7.
    """

    def __init__(self, min_peak_distance_px: int = 3, prominence: float | None = None):
        self.min_peak_distance_px = min_peak_distance_px
        self.prominence = prominence

    def estimate(self, profile: ThreadProfile) -> PitchEstimate:
        signal = profile.signal
        notes: list[str] = []

        if signal.size < 5:
            return PitchEstimate(
                pitch_px=None,
                confidence=0.0,
                method="peak_spacing",
                notes=["signal too short to estimate pitch"],
            )

        prominence = self.prominence
        if prominence is None:
            prominence = 0.15 * (np.ptp(signal) if np.ptp(signal) > 0 else 1.0)

        peaks, _ = find_peaks(
            signal, distance=self.min_peak_distance_px, prominence=prominence
        )

        if len(peaks) < 3:
            return PitchEstimate(
                pitch_px=None,
                confidence=0.0,
                method="peak_spacing",
                notes=[f"found only {len(peaks)} thread-crest peaks; need >= 3"],
            )

        spacings = np.diff(peaks).astype(np.float64)
        pitch_px = float(np.median(spacings))

        mean_spacing = float(np.mean(spacings))
        std_spacing = float(np.std(spacings))
        cv = std_spacing / mean_spacing if mean_spacing > 0 else 1.0
        confidence = float(np.clip(1.0 - cv, 0.0, 1.0))

        if len(peaks) < 5:
            notes.append(f"only {len(peaks)} peaks found; pitch estimate may be unstable")
            confidence *= 0.7

        return PitchEstimate(
            pitch_px=pitch_px, confidence=confidence, method="peak_spacing", notes=notes
        )


class FFTPitchEstimator(PitchEstimator):
    """Phase-4 strategy: periodicity via FFT of the 1D thread-width signal.

    Not implemented in Phase 1 — see module docstring. The interface exists
    now so the Phase-4 comparison table (CLAUDE.md §4.2, §8) can be added
    without restructuring callers.
    """

    def estimate(self, profile: ThreadProfile) -> PitchEstimate:
        raise NotImplementedError(
            "FFTPitchEstimator is a Phase-4 deliverable (CLAUDE.md §4.2/§7); "
            "PeakPitchEstimator is the Phase-1 default."
        )
