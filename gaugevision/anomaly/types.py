"""Stable output contract for the anomaly-detection module (CLAUDE.md §4.3)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnomalyResult:
    """Per-image anomaly-detection output.

    ``heatmap`` is float32, same (H, W) as the input image, normalized to the
    model's raw Mahalanobis-distance scale (not 0-1) so thresholds are
    comparable across images without per-image rescaling.
    """

    score: float
    heatmap: np.ndarray
    threshold: float
    is_anomalous: bool
