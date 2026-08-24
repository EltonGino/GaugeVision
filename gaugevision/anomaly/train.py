"""Fit and persist the Phase-1 PaDiM baseline (CLAUDE.md §4.3, §7 Phase 1).

Run as: python -m gaugevision.anomaly.train

Threshold selection deliberately uses only the training (normal-only) score
distribution — a percentile of training scores — rather than the labeled
test set, so the test-set AUROC reported here stays a clean held-out
evaluation rather than one where the threshold was tuned against it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from gaugevision.anomaly.evaluate import evaluate_image_level_auroc
from gaugevision.anomaly.padim import PaDiM
from gaugevision.data.mvtec_loader import (
    load_screw_category,
    test_samples,
    train_normal_samples,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/padim_screw.npz")
THRESHOLD_PERCENTILE = 99.0


def _read_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"failed to read image: {path}")
    return img


def fit_and_save(
    model_path: Path = DEFAULT_MODEL_PATH,
    n_features: int = 100,
    seed: int = 0,
) -> dict:
    samples = load_screw_category()
    train = train_normal_samples(samples)
    test = test_samples(samples)
    logger.info("Fitting PaDiM on %d normal training images", len(train))

    train_images = [_read_gray(s.image_path) for s in train]
    model = PaDiM(n_features=n_features, seed=seed)
    model.fit(train_images)

    train_scores = np.array(
        [model.predict(img, threshold=0.0).score for img in train_images]
    )
    threshold = float(np.percentile(train_scores, THRESHOLD_PERCENTILE))
    model.set_threshold(threshold)
    logger.info(
        "Threshold (train p%.1f of normal-only scores): %.4f",
        THRESHOLD_PERCENTILE,
        threshold,
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(model_path))
    logger.info("Saved model to %s", model_path)

    test_images = [_read_gray(s.image_path) for s in test]
    test_labels = [1 if s.is_defective else 0 for s in test]
    eval_result = evaluate_image_level_auroc(model, test_images, test_labels)
    logger.info(
        "Held-out test set image-level AUROC: %.4f (n_normal=%d, n_anomalous=%d)",
        eval_result.image_auroc,
        eval_result.n_normal,
        eval_result.n_anomalous,
    )

    return {
        "model_path": str(model_path),
        "threshold": threshold,
        "test_image_auroc": eval_result.image_auroc,
        "n_train": len(train),
        "n_test": len(test),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = fit_and_save()
    print(result)
