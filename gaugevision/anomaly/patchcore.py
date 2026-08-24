"""PatchCore anomaly detection — CLAUDE.md §4.3 (Phase-4 stretch).

Coreset-reduced memory bank of patch features + nearest-neighbor scoring
(Roth et al., 2022). Training-free like PaDiM: no backpropagation, only a
frozen backbone's forward pass over normal training images, followed by
greedy coreset subsampling of the resulting patch-feature memory bank.

Reuses the exact same ``PaDiMBackbone`` feature extractor as the PaDiM
baseline (``anomaly/padim.py``) rather than a separate backbone path — this
isolates the RESULTS.md comparison to "distribution modeling (PaDiM's
per-patch Gaussian) vs. memory-bank + nearest-neighbor (PatchCore)", not to
incidental backbone differences.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from gaugevision.anomaly.padim import PaDiMBackbone, preprocess
from gaugevision.anomaly.types import AnomalyResult

logger = logging.getLogger(__name__)


def _greedy_coreset(features: np.ndarray, coreset_size: int, seed: int) -> np.ndarray:
    """Greedy k-center (farthest-point) coreset selection.

    Iteratively picks the feature vector farthest (in L2) from the currently
    selected set, which is the standard subsampling strategy PatchCore uses
    to keep the memory bank small while preserving coverage of the normal
    feature distribution — a random subsample would under-represent rare
    but still-normal patch appearances.
    """
    n = features.shape[0]
    coreset_size = min(coreset_size, n)
    rng = np.random.default_rng(seed)

    selected_indices = [int(rng.integers(0, n))]
    diff0 = features - features[selected_indices[0]]
    min_dist_sq = np.einsum("ij,ij->i", diff0, diff0)

    for _ in range(coreset_size - 1):
        next_idx = int(np.argmax(min_dist_sq))
        selected_indices.append(next_idx)
        diff = features - features[next_idx]
        dist_sq = np.einsum("ij,ij->i", diff, diff)
        min_dist_sq = np.minimum(min_dist_sq, dist_sq)

    return features[selected_indices]


@dataclass
class PatchCoreMemoryBank:
    features: np.ndarray  # (coreset_size, d)
    feature_indices: np.ndarray  # (d,) indices into the full backbone embedding dim
    grid_size: tuple[int, int]
    input_size: tuple[int, int]


class PatchCore:
    """PatchCore anomaly detector: memory bank of normal patch features,
    scored at inference via nearest-neighbor distance."""

    def __init__(
        self,
        n_features: int = 100,
        coreset_size: int = 2000,
        candidate_pool_size: int = 20000,
        input_size: tuple[int, int] = (224, 224),
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.n_features = n_features
        self.coreset_size = coreset_size
        self.candidate_pool_size = candidate_pool_size
        self.input_size = input_size
        self.seed = seed
        self.device = torch.device(device)
        self.backbone = PaDiMBackbone().to(self.device)
        self.memory_bank: PatchCoreMemoryBank | None = None

    @torch.no_grad()
    def _extract_embedding(self, image: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
        x = preprocess(image, self.input_size).to(self.device)
        embedding = self.backbone(x)  # (1, C, H, W)
        _, c, h, w = embedding.shape
        emb = embedding.reshape(1, c, h * w).permute(0, 2, 1).squeeze(0)
        return emb.numpy(), (h, w)

    def fit(self, train_images: list[np.ndarray]) -> None:
        """Build the coreset memory bank from defect-free training images."""
        if not train_images:
            raise ValueError("PatchCore.fit requires at least one training image")

        all_embeddings: list[np.ndarray] = []
        grid_size = None
        for img in train_images:
            emb, grid_size = self._extract_embedding(img)
            all_embeddings.append(emb)

        stacked = np.concatenate(all_embeddings, axis=0)  # (n_images*n_patches, total_dim)
        total_dim = stacked.shape[1]

        rng = np.random.default_rng(self.seed)
        n_selected = min(self.n_features, total_dim)
        feature_indices = np.sort(rng.choice(total_dim, size=n_selected, replace=False))
        stacked = stacked[:, feature_indices]

        n_patches_total = stacked.shape[0]
        if n_patches_total > self.candidate_pool_size:
            pool_idx = rng.choice(n_patches_total, size=self.candidate_pool_size, replace=False)
            candidate_pool = stacked[pool_idx]
        else:
            candidate_pool = stacked

        logger.info(
            "PatchCore: %d total patches -> %d candidate pool -> selecting %d-point coreset",
            n_patches_total,
            candidate_pool.shape[0],
            self.coreset_size,
        )
        coreset = _greedy_coreset(candidate_pool, self.coreset_size, self.seed)

        self.memory_bank = PatchCoreMemoryBank(
            features=coreset,
            feature_indices=feature_indices,
            grid_size=grid_size,
            input_size=self.input_size,
        )
        logger.info("PatchCore fit complete: memory bank size %d", coreset.shape[0])

    def predict(self, image: np.ndarray, threshold: float | None = None) -> AnomalyResult:
        if self.memory_bank is None:
            raise RuntimeError("PatchCore.predict called before fit()")

        emb, grid_size = self._extract_embedding(image)
        if grid_size != self.memory_bank.grid_size:
            raise RuntimeError(
                f"grid size mismatch: fitted on {self.memory_bank.grid_size}, got {grid_size}"
            )
        emb = emb[:, self.memory_bank.feature_indices]  # (n_patches, d)

        # Per-patch 1-NN L2 distance to the memory bank.
        bank = self.memory_bank.features  # (coreset_size, d)
        dists_sq = (
            (emb**2).sum(axis=1, keepdims=True)
            - 2 * emb @ bank.T
            + (bank**2).sum(axis=1)[None, :]
        )
        min_dist = np.sqrt(np.clip(dists_sq.min(axis=1), 0, None))

        h, w = grid_size
        dist_map = min_dist.reshape(h, w)

        orig_h, orig_w = image.shape[:2]
        heatmap = _resize_map(dist_map, (orig_h, orig_w))

        score = float(dist_map.max())
        threshold = self.threshold_ if threshold is None else threshold
        return AnomalyResult(
            score=score,
            heatmap=heatmap.astype(np.float32),
            threshold=float(threshold) if threshold is not None else float("nan"),
            is_anomalous=bool(threshold is not None and score > threshold),
        )

    def set_threshold(self, threshold: float) -> None:
        self.threshold_ = float(threshold)

    def save(self, path: str) -> None:
        if self.memory_bank is None:
            raise RuntimeError("PatchCore.save called before fit()")
        np.savez_compressed(
            path,
            features=self.memory_bank.features,
            feature_indices=self.memory_bank.feature_indices,
            grid_size=np.array(self.memory_bank.grid_size),
            input_size=np.array(self.memory_bank.input_size),
            threshold=np.array([getattr(self, "threshold_", np.nan)]),
            n_features=np.array([self.n_features]),
            coreset_size=np.array([self.coreset_size]),
            candidate_pool_size=np.array([self.candidate_pool_size]),
            seed=np.array([self.seed]),
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> PatchCore:
        data = np.load(path)
        model = cls(
            n_features=int(data["n_features"][0]),
            coreset_size=int(data["coreset_size"][0]),
            candidate_pool_size=int(data["candidate_pool_size"][0]),
            input_size=tuple(int(v) for v in data["input_size"]),
            seed=int(data["seed"][0]),
            device=device,
        )
        model.memory_bank = PatchCoreMemoryBank(
            features=data["features"],
            feature_indices=data["feature_indices"],
            grid_size=tuple(int(v) for v in data["grid_size"]),
            input_size=tuple(int(v) for v in data["input_size"]),
        )
        threshold = float(data["threshold"][0])
        if not np.isnan(threshold):
            model.threshold_ = threshold
        return model


def _resize_map(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()
