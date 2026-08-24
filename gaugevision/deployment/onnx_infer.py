"""ONNX Runtime CPU inference wrapper — CLAUDE.md §4.4 (Phase 2).

Runs the exported PaDiM backbone (``models/padim_backbone.onnx``) through
ONNX Runtime's CPU execution provider, then applies the exact same NumPy
Mahalanobis scoring (``gaugevision.anomaly.padim.mahalanobis_score``) used by
the PyTorch path. Preprocessing is also shared (``gaugevision.anomaly.padim.
preprocess``) so an output-equivalence check isolates the comparison to the
backbone forward pass itself, not incidental preprocessing differences.
"""

from __future__ import annotations

import numpy as np
import onnxruntime as ort

from gaugevision.anomaly.padim import load_stats, mahalanobis_score, preprocess
from gaugevision.anomaly.types import AnomalyResult


class ONNXPaDiM:
    """PaDiM inference via ONNX Runtime (CPU execution provider).

    Loads a previously exported backbone (``export_onnx.export_backbone``)
    and previously fitted statistics (``PaDiM.save``) — this class only runs
    inference, it cannot fit new statistics (fitting requires the PyTorch
    backbone's gradient-free forward pass over the training set, which stays
    on the PyTorch path in ``anomaly/train.py``).
    """

    def __init__(
        self,
        onnx_path: str,
        stats_path: str,
        providers: list[str] | None = None,
    ) -> None:
        self.session = ort.InferenceSession(
            onnx_path, providers=providers or ["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.stats, self.threshold_ = load_stats(stats_path)

    def predict(self, image: np.ndarray, threshold: float | None = None) -> AnomalyResult:
        x = preprocess(image, self.stats.input_size).numpy()
        (embedding,) = self.session.run(None, {self.input_name: x})  # (1, C, H, W)

        _, c, h, w = embedding.shape
        emb = embedding.reshape(c, h * w).T  # (n_patches, total_dim)

        grid_size = (h, w)
        if grid_size != self.stats.grid_size:
            raise RuntimeError(
                f"grid size mismatch: fitted on {self.stats.grid_size}, got {grid_size}"
            )

        score, dist_map = mahalanobis_score(emb, self.stats)
        heatmap = _resize_map_cv2(dist_map, image.shape[:2])

        threshold = self.threshold_ if threshold is None else threshold
        return AnomalyResult(
            score=score,
            heatmap=heatmap.astype(np.float32),
            threshold=float(threshold) if threshold is not None else float("nan"),
            is_anomalous=bool(threshold is not None and score > threshold),
        )


def _resize_map_cv2(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """cv2-based heatmap resize — keeps this inference path free of a torch
    dependency for anything beyond shared preprocessing."""
    import cv2

    h, w = size
    return cv2.resize(arr.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
