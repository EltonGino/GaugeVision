"""Anomaly-detection evaluation — CLAUDE.md §4.3.

Deliberately separate from inference logic (``padim.py``, ``patchcore.py``):
this module only consumes a fitted model's scores against ground-truth
labels, and works identically for either model (anything with a ``predict(
image, threshold=...) -> AnomalyResult`` method). Image-level AUROC is the
metric benchmarked here — CLAUDE.md is explicit that image-level and
pixel-level AUROC are different numbers that are easy to conflate across
papers, so any comparison against published PaDiM/PatchCore results must cite
the specific table, method, backbone, and metric it was read from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from gaugevision.anomaly.types import AnomalyResult


class AnomalyModel(Protocol):
    def predict(self, image: np.ndarray, threshold: float | None = None) -> AnomalyResult: ...


@dataclass(frozen=True)
class EvaluationResult:
    image_auroc: float
    n_normal: int
    n_anomalous: int
    recommended_threshold: float
    scores: np.ndarray
    labels: np.ndarray


def evaluate_image_level_auroc(
    model: AnomalyModel, images: list[np.ndarray], labels: list[int]
) -> EvaluationResult:
    """Compute image-level AUROC for a fitted anomaly model on labeled images.

    Args:
        images: test images (mix of normal and defective).
        labels: 0 = normal (good), 1 = anomalous (defective), same order as
            ``images``. Matches MVTec AD's convention.
    """
    if len(images) != len(labels):
        raise ValueError("images and labels must be the same length")
    if len(set(labels)) < 2:
        raise ValueError(
            "AUROC requires both normal and anomalous examples in the test set"
        )

    scores = np.array(
        [model.predict(img, threshold=0.0).score for img in images], dtype=np.float64
    )
    labels_arr = np.array(labels, dtype=np.int64)

    auroc = float(roc_auc_score(labels_arr, scores))

    fpr, tpr, thresholds = roc_curve(labels_arr, scores)
    youden_j = tpr - fpr
    best_idx = int(np.argmax(youden_j))
    recommended_threshold = float(thresholds[best_idx])

    return EvaluationResult(
        image_auroc=auroc,
        n_normal=int(np.sum(labels_arr == 0)),
        n_anomalous=int(np.sum(labels_arr == 1)),
        recommended_threshold=recommended_threshold,
        scores=scores,
        labels=labels_arr,
    )
