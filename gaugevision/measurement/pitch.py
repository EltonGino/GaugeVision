"""Pitch estimation — measurement pipeline steps 6-7 (CLAUDE.md §4.2).

Pitch estimation is implemented behind a common ``PitchEstimator`` interface
with two competing strategies, per CLAUDE.md:

- ``PeakPitchEstimator`` (peak-to-peak spacing of thread crests): the
  Phase-1 default, fully implemented and used by the pipeline.
- ``FFTPitchEstimator`` (periodicity via FFT of the 1D signal): implemented
  in Phase 4 — see ``measurement/validate.py`` for the quantitative
  Peak-vs-FFT comparison against synthetic ground truth (CLAUDE.md §4.2,
  §8) that the default estimator choice is based on.
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
    """Estimates pitch from the dominant frequency of the 1D thread-width
    signal's FFT magnitude spectrum.

    Steps: detrend (subtract a moving-average baseline so the signal's
    slowly-varying taper doesn't dominate the DC/low-frequency bins), window
    (Hann, to reduce spectral leakage from the non-periodic boundary), FFT,
    then take the reciprocal of the dominant non-DC frequency as the pitch.

    Confidence is how far the dominant peak's magnitude stands out above the
    mean of the rest of the (non-DC) spectrum — a strongly periodic signal
    has one sharp spectral peak; a noisy/aperiodic one has a flat spectrum
    with no bin standing out. This directly feeds ``MeasurementResult.
    confidence`` and the "flag when the estimate isn't reliable" requirement
    in CLAUDE.md §4.2 step 7.
    """

    def __init__(self, min_signal_length: int = 16):
        self.min_signal_length = min_signal_length

    def estimate(self, profile: ThreadProfile) -> PitchEstimate:
        signal = profile.signal
        n = signal.size

        if n < self.min_signal_length:
            return PitchEstimate(
                pitch_px=None,
                confidence=0.0,
                method="fft",
                notes=[f"signal too short ({n} < {self.min_signal_length})"],
            )

        window_len = max(3, (n // 4) | 1)  # odd, ~quarter of signal length
        # Edge-replicate padding before convolving avoids the artificial
        # ramp mode="same" would otherwise create at the boundaries via
        # implicit zero-padding — that ramp is not real thread signal and
        # would leak spurious low-frequency energy into the spectrum below.
        pad = window_len // 2
        padded = np.pad(signal, pad, mode="edge")
        baseline = np.convolve(padded, np.ones(window_len) / window_len, mode="valid")
        detrended = signal - baseline
        windowed = detrended * np.hanning(n)

        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n)  # cycles per sample

        # Exclude DC and implausibly low frequencies (fewer than 2 full
        # thread cycles across the whole sampled region isn't a usable pitch
        # estimate for a part this short).
        valid = freqs >= 2.0 / n
        if not np.any(valid):
            return PitchEstimate(
                pitch_px=None, confidence=0.0, method="fft", notes=["no valid frequency bins"]
            )

        valid_spectrum = spectrum[valid]
        valid_freqs = freqs[valid]
        peak_idx = int(np.argmax(valid_spectrum))
        peak_freq = float(valid_freqs[peak_idx])

        if peak_freq <= 0:
            return PitchEstimate(
                pitch_px=None,
                confidence=0.0,
                method="fft",
                notes=["dominant frequency is zero"],
            )

        pitch_px = 1.0 / peak_freq

        peak_mag = float(valid_spectrum[peak_idx])
        mean_mag = float(np.mean(valid_spectrum))
        if mean_mag <= 0:
            confidence = 0.0
        else:
            ratio = peak_mag / mean_mag
            # ratio=1 (peak indistinguishable from noise floor) -> confidence 0;
            # ratio>=6 -> saturate near confidence 1.
            confidence = float(np.clip((ratio - 1.0) / 5.0, 0.0, 1.0))

        return PitchEstimate(pitch_px=pitch_px, confidence=confidence, method="fft", notes=[])
