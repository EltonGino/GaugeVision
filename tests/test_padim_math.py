"""Deterministic unit tests for PaDiM's shared math (no backbone/network
required) — the same functions both the PyTorch and ONNX Runtime inference
paths call, so correctness here backs both backends at once."""

import numpy as np
import pytest

from gaugevision.anomaly.padim import PaDiMStats, mahalanobis_score, preprocess


def _identity_stats(n_patches: int = 4, d: int = 3, grid_size: tuple[int, int] = (2, 2)) -> PaDiMStats:
    mean = np.zeros((n_patches, d))
    cov_inv = np.stack([np.eye(d)] * n_patches)
    feature_indices = np.arange(d)
    return PaDiMStats(
        mean=mean,
        cov_inv=cov_inv,
        feature_indices=feature_indices,
        grid_size=grid_size,
        input_size=(224, 224),
    )


def test_mahalanobis_score_zero_at_mean():
    stats = _identity_stats()
    embedding = np.zeros((4, 3))  # exactly at the fitted mean
    score, dist_map = mahalanobis_score(embedding, stats)
    assert score == pytest.approx(0.0, abs=1e-9)
    assert dist_map.shape == (2, 2)
    assert np.allclose(dist_map, 0.0)


def test_mahalanobis_score_identity_covariance_is_euclidean_distance():
    stats = _identity_stats()
    embedding = np.zeros((4, 3))
    embedding[0] = [3.0, 4.0, 0.0]  # distance 5 from mean under identity cov
    score, dist_map = mahalanobis_score(embedding, stats)
    assert score == pytest.approx(5.0)
    assert dist_map.flat[0] == pytest.approx(5.0)


def test_mahalanobis_score_uses_feature_indices_subset():
    stats = _identity_stats(n_patches=1, d=2, grid_size=(1, 1))
    stats.feature_indices = np.array([1, 3])  # select from a wider embedding
    embedding = np.array([[0.0, 3.0, 0.0, 4.0]])  # cols 1,3 -> (3,4), dist=5
    score, _ = mahalanobis_score(embedding, stats)
    assert score == pytest.approx(5.0)


def test_preprocess_grayscale_produces_normalized_3channel_tensor():
    image = np.full((50, 60), 128, dtype=np.uint8)
    tensor = preprocess(image, input_size=(224, 224))
    assert tensor.shape == (1, 3, 224, 224)
    assert tensor.dtype.is_floating_point


def test_preprocess_bgr_and_grayscale_same_size_agree_after_conversion():
    gray = np.full((50, 50), 200, dtype=np.uint8)
    bgr = np.stack([gray] * 3, axis=-1)
    t_gray = preprocess(gray, input_size=(64, 64))
    t_bgr = preprocess(bgr, input_size=(64, 64))
    assert np.allclose(t_gray.numpy(), t_bgr.numpy(), atol=1e-5)
