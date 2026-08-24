import numpy as np
import pytest

from gaugevision.anomaly.patchcore import _greedy_coreset


def test_greedy_coreset_selects_requested_size():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(500, 8))
    coreset = _greedy_coreset(features, coreset_size=50, seed=0)
    assert coreset.shape == (50, 8)


def test_greedy_coreset_caps_at_available_points():
    rng = np.random.default_rng(0)
    features = rng.normal(size=(10, 4))
    coreset = _greedy_coreset(features, coreset_size=100, seed=0)
    assert coreset.shape == (10, 4)


def test_greedy_coreset_covers_distinct_clusters():
    """A coreset selected from two well-separated clusters should include
    points from both clusters, not collapse onto one (the whole point of
    farthest-point sampling over random subsampling)."""
    rng = np.random.default_rng(0)
    cluster_a = rng.normal(loc=0.0, scale=0.1, size=(100, 4))
    cluster_b = rng.normal(loc=20.0, scale=0.1, size=(100, 4))
    features = np.vstack([cluster_a, cluster_b])

    coreset = _greedy_coreset(features, coreset_size=10, seed=0)
    near_a = (np.linalg.norm(coreset - 0.0, axis=1) < 5.0).sum()
    near_b = (np.linalg.norm(coreset - 20.0, axis=1) < 5.0).sum()
    assert near_a > 0
    assert near_b > 0


@pytest.fixture
def small_patchcore_model():
    import cv2

    from gaugevision.anomaly.patchcore import PatchCore

    imgs = []
    for _ in range(3):
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.circle(img, (50, 50), 30, 255, -1)
        imgs.append(img)

    model = PatchCore(n_features=16, coreset_size=30, candidate_pool_size=100)
    model.fit(imgs)
    return model, imgs


def test_patchcore_scores_normal_lower_than_anomalous(small_patchcore_model):
    import cv2

    model, imgs = small_patchcore_model
    normal_score = model.predict(imgs[0], threshold=0.0).score

    anomalous = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(anomalous, (10, 10), (90, 90), 255, -1)
    anomalous_score = model.predict(anomalous, threshold=0.0).score

    assert anomalous_score > normal_score


def test_patchcore_save_and_load_roundtrip(small_patchcore_model, tmp_path):
    model, imgs = small_patchcore_model
    model.set_threshold(1.23)
    path = tmp_path / "patchcore_test.npz"
    model.save(str(path))

    from gaugevision.anomaly.patchcore import PatchCore

    loaded = PatchCore.load(str(path))
    original_score = model.predict(imgs[0], threshold=0.0).score
    loaded_score = loaded.predict(imgs[0], threshold=0.0).score
    assert original_score == pytest.approx(loaded_score, abs=1e-6)
    assert loaded.threshold_ == pytest.approx(1.23)


def test_patchcore_predict_before_fit_raises():
    from gaugevision.anomaly.patchcore import PatchCore

    model = PatchCore()
    with pytest.raises(RuntimeError):
        model.predict(np.zeros((50, 50), dtype=np.uint8), threshold=0.0)
