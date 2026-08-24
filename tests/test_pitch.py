import numpy as np
import pytest

from gaugevision.measurement.pitch import FFTPitchEstimator, PeakPitchEstimator
from gaugevision.measurement.types import ThreadProfile


def _synthetic_profile(period_px: float, n_periods: int = 12, noise_std: float = 0.0, seed: int = 0):
    rng = np.random.default_rng(seed)
    n = int(period_px * n_periods)
    x = np.arange(n)
    signal = 20.0 + 5.0 * np.sin(2 * np.pi * x / period_px)
    if noise_std > 0:
        signal = signal + rng.normal(0, noise_std, size=n)
    return ThreadProfile(signal=signal, axis_positions_px=x)


def test_peak_pitch_estimator_clean_signal():
    profile = _synthetic_profile(period_px=12.0)
    estimate = PeakPitchEstimator().estimate(profile)
    assert estimate.pitch_px is not None
    assert estimate.pitch_px == pytest.approx(12.0, abs=1.0)
    assert estimate.confidence > 0.8


def test_peak_pitch_estimator_noisy_signal_lower_confidence():
    clean = _synthetic_profile(period_px=12.0)
    noisy = _synthetic_profile(period_px=12.0, noise_std=3.0, seed=1)
    clean_est = PeakPitchEstimator().estimate(clean)
    noisy_est = PeakPitchEstimator().estimate(noisy)
    assert noisy_est.confidence <= clean_est.confidence


def test_peak_pitch_estimator_too_short_signal():
    profile = ThreadProfile(signal=np.array([1.0, 2.0, 1.0]), axis_positions_px=np.arange(3))
    estimate = PeakPitchEstimator().estimate(profile)
    assert estimate.pitch_px is None
    assert estimate.confidence == 0.0


def test_peak_pitch_estimator_flat_signal_no_peaks():
    profile = ThreadProfile(signal=np.full(50, 10.0), axis_positions_px=np.arange(50))
    estimate = PeakPitchEstimator().estimate(profile)
    assert estimate.pitch_px is None


def test_fft_pitch_estimator_not_implemented_in_phase1():
    profile = _synthetic_profile(period_px=12.0)
    with pytest.raises(NotImplementedError):
        FFTPitchEstimator().estimate(profile)
