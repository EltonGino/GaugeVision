"""PaDiM anomaly detection — CLAUDE.md §4.3 (Phase-1 baseline).

Patch Distribution Modeling (Defard et al., 2020): a frozen ImageNet-pretrained
CNN backbone extracts multi-scale patch features; a per-patch-location
multivariate Gaussian is fit over normal (defect-free) training images only,
in the standard one-class / unsupervised-anomaly-detection paradigm used
throughout industrial inspection. At inference, the Mahalanobis distance
between a test image's patch features and its location's fitted Gaussian
gives a pixel-level anomaly heatmap; the max of that heatmap is the
image-level anomaly score.

Training-free beyond fitting Gaussian statistics: no backpropagation, no
labeled defects required — this mirrors why the anomaly-detection industry
standard is "model normal, flag deviation" (defective samples are rare and
hard to source).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from gaugevision.anomaly.types import AnomalyResult

logger = logging.getLogger(__name__)

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class _ResNet18FeatureExtractor(nn.Module):
    """Frozen ResNet18 backbone exposing layer1/layer2/layer3 activations."""

    def __init__(self) -> None:
        super().__init__()
        backbone = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)

        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.stem(x)
        f1 = self.layer1(x)
        f2 = self.layer2(f1)
        f3 = self.layer3(f2)
        return f1, f2, f3


def _concat_multiscale_features(
    f1: torch.Tensor, f2: torch.Tensor, f3: torch.Tensor
) -> torch.Tensor:
    """Resize f1/f2/f3 to a common spatial resolution and concat channel-wise.

    Uses layer2's resolution (28x28 for a 224x224 input) rather than layer1's
    (56x56): a defensible CPU-feasibility tradeoff for Phase 1 (per-patch
    covariance fitting is O(n_patches), so 4x fewer patches is a ~4x speedup)
    at the cost of coarser heatmap localization. CLAUDE.md §4.3 asks for a
    CPU-friendly Phase-1 baseline over maximum localization fidelity.
    """
    target_size = f2.shape[-2:]
    f1_down = F.interpolate(f1, size=target_size, mode="bilinear", align_corners=False)
    f3_up = F.interpolate(f3, size=target_size, mode="bilinear", align_corners=False)
    return torch.cat([f1_down, f2, f3_up], dim=1)


class PaDiMBackbone(nn.Module):
    """Feature-extraction half of PaDiM as a single exportable module.

    Combines ``_ResNet18FeatureExtractor`` and ``_concat_multiscale_features``
    into one ``forward`` so the same graph used at PyTorch inference time is
    exactly what gets traced for ONNX export (CLAUDE.md §4.4) — there is no
    separate "export path" that could silently diverge from the inference
    path. Only this module is exported to ONNX; PaDiM's Gaussian-distribution
    Mahalanobis scoring (``mahalanobis_score`` below) stays in NumPy, per the
    CLAUDE.md §4.4 scope note that PaDiM's scoring logic may not translate
    cleanly through the ONNX exporter and a backbone/scoring split is an
    acceptable architecture.
    """

    def __init__(self) -> None:
        super().__init__()
        self.extractor = _ResNet18FeatureExtractor()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f1, f2, f3 = self.extractor(x)
        return _concat_multiscale_features(f1, f2, f3)


def preprocess(
    image: np.ndarray, input_size: tuple[int, int] = (224, 224)
) -> torch.Tensor:
    """Grayscale/BGR uint8 image -> normalized 3-channel model input tensor.

    Shared by both the PyTorch and ONNX Runtime inference paths so an
    output-equivalence benchmark (CLAUDE.md §4.4) isolates the comparison to
    "does the exported graph match the original graph," not to incidental
    preprocessing differences between the two paths.
    """
    if image.ndim == 2:
        rgb = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 1:
        rgb = np.repeat(image, 3, axis=2)
    else:
        rgb = image[:, :, ::-1]  # assume BGR (OpenCV) -> RGB

    tensor = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0)
    tensor = F.interpolate(tensor, size=input_size, mode="bilinear", align_corners=False)
    tensor = (tensor - _IMAGENET_MEAN) / _IMAGENET_STD
    return tensor


def mahalanobis_score(
    embedding: np.ndarray, stats: PaDiMStats
) -> tuple[float, np.ndarray]:
    """Per-patch Mahalanobis distance against fitted PaDiM statistics.

    Args:
        embedding: (n_patches, total_dim) raw concatenated backbone features
            for one image, pre feature-selection (same shape either backend's
            backbone forward pass produces).
        stats: fitted PaDiMStats (mean/cov_inv/feature_indices).

    Returns:
        (image_level_score, per_patch_distance_map (h, w)).
    """
    selected = embedding[:, stats.feature_indices]  # (n_patches, d)
    diff = selected - stats.mean
    dist_sq = np.einsum("pi,pij,pj->p", diff, stats.cov_inv, diff)
    dist = np.sqrt(np.clip(dist_sq, 0, None))

    h, w = stats.grid_size
    dist_map = dist.reshape(h, w)
    return float(dist_map.max()), dist_map


@dataclass
class PaDiMStats:
    """Fitted per-patch-location Gaussian parameters."""

    mean: np.ndarray  # (n_patches, d)
    cov_inv: np.ndarray  # (n_patches, d, d)
    feature_indices: np.ndarray  # (d,) indices into the full concatenated feature dim
    grid_size: tuple[int, int]  # (H_patch, W_patch)
    input_size: tuple[int, int]  # (H, W) the model expects


def load_stats(path: str) -> tuple[PaDiMStats, float | None]:
    """Load fitted PaDiMStats + threshold from a PaDiM.save() checkpoint.

    Shared by ``PaDiM.load`` (PyTorch path) and
    ``deployment.onnx_infer.ONNXPaDiM`` (ONNX Runtime path) — neither backend
    needs a backbone to read these statistics, only a fitted model checkpoint.
    """
    data = np.load(path)
    stats = PaDiMStats(
        mean=data["mean"],
        cov_inv=data["cov_inv"],
        feature_indices=data["feature_indices"],
        grid_size=tuple(int(v) for v in data["grid_size"]),
        input_size=tuple(int(v) for v in data["input_size"]),
    )
    threshold = float(data["threshold"][0])
    return stats, (None if np.isnan(threshold) else threshold)


class PaDiM:
    """PaDiM anomaly detector: fit on normal images, score/localize on any image."""

    def __init__(
        self,
        n_features: int = 100,
        input_size: tuple[int, int] = (224, 224),
        seed: int = 0,
        device: str = "cpu",
    ) -> None:
        self.n_features = n_features
        self.input_size = input_size
        self.seed = seed
        self.device = torch.device(device)
        self.backbone = PaDiMBackbone().to(self.device)
        self.stats: PaDiMStats | None = None

    @torch.no_grad()
    def _extract_embedding(self, image: np.ndarray) -> torch.Tensor:
        """Returns (n_patches, total_dim) feature embedding for one image."""
        x = preprocess(image, self.input_size).to(self.device)
        embedding = self.backbone(x)  # (1, C, H, W)
        _, c, h, w = embedding.shape
        return embedding.reshape(1, c, h * w).permute(0, 2, 1).squeeze(0), (h, w)

    def fit(self, train_images: list[np.ndarray]) -> None:
        """Fit per-patch Gaussians over a set of defect-free training images."""
        if not train_images:
            raise ValueError("PaDiM.fit requires at least one training image")

        embeddings: list[torch.Tensor] = []
        grid_size = None
        for img in train_images:
            emb, grid_size = self._extract_embedding(img)
            embeddings.append(emb)

        stacked = torch.stack(embeddings, dim=0)  # (N, n_patches, total_dim)
        total_dim = stacked.shape[-1]

        rng = np.random.default_rng(self.seed)
        n_selected = min(self.n_features, total_dim)
        feature_indices = np.sort(
            rng.choice(total_dim, size=n_selected, replace=False)
        )
        stacked = stacked[:, :, feature_indices]  # (N, n_patches, d)

        n_images, n_patches, d = stacked.shape
        data = stacked.numpy()

        mean = data.mean(axis=0)  # (n_patches, d)
        cov_inv = np.empty((n_patches, d, d), dtype=np.float64)
        eps_identity = 0.01 * np.eye(d)
        for p in range(n_patches):
            centered = data[:, p, :] - mean[p]
            cov = (centered.T @ centered) / max(n_images - 1, 1) + eps_identity
            cov_inv[p] = np.linalg.inv(cov)

        self.stats = PaDiMStats(
            mean=mean,
            cov_inv=cov_inv,
            feature_indices=feature_indices,
            grid_size=grid_size,
            input_size=self.input_size,
        )
        logger.info(
            "PaDiM fit complete: %d train images, %d patches, %d/%d features",
            n_images,
            n_patches,
            n_selected,
            total_dim,
        )

    def predict(self, image: np.ndarray, threshold: float | None = None) -> AnomalyResult:
        """Score and localize anomalies in a single image."""
        if self.stats is None:
            raise RuntimeError("PaDiM.predict called before fit()")

        emb, grid_size = self._extract_embedding(image)
        if grid_size != self.stats.grid_size:
            raise RuntimeError(
                f"grid size mismatch: fitted on {self.stats.grid_size}, got {grid_size}"
            )
        score, dist_map = mahalanobis_score(emb.numpy(), self.stats)

        orig_h, orig_w = image.shape[:2]
        heatmap = _resize_map(dist_map, (orig_h, orig_w))

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
        if self.stats is None:
            raise RuntimeError("PaDiM.save called before fit()")
        np.savez_compressed(
            path,
            mean=self.stats.mean,
            cov_inv=self.stats.cov_inv,
            feature_indices=self.stats.feature_indices,
            grid_size=np.array(self.stats.grid_size),
            input_size=np.array(self.stats.input_size),
            threshold=np.array([getattr(self, "threshold_", np.nan)]),
            n_features=np.array([self.n_features]),
            seed=np.array([self.seed]),
        )

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> PaDiM:
        data = np.load(path)
        model = cls(
            n_features=int(data["n_features"][0]),
            input_size=tuple(int(v) for v in data["input_size"]),
            seed=int(data["seed"][0]),
            device=device,
        )
        model.stats, threshold = load_stats(path)
        if threshold is not None:
            model.threshold_ = threshold
        return model


def _resize_map(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    tensor = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(tensor, size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()
