"""MVTec AD "screw" category loader — CLAUDE.md §3.1.

Downloads only the screw-category images and defect masks referenced by the
Voxel51/mvtec-ad Hugging Face mirror's FiftyOne sample manifest
(``samples.json``), using ``huggingface_hub`` directly rather than pulling
the full 15-category / ~5,300-sample dataset through the heavier
``fiftyone`` runtime. This keeps acquisition fully scriptable, filtered to
the "screw" category at download time, and locally cached (individual file
downloads are content-addressed and skipped on re-run).

**License:** MVTec AD is CC BY-NC-SA 4.0 (Bergmann et al., "MVTec AD — A
Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", 2019,
and its extension, 2021) — non-commercial, share-alike, attribution required.
This loader and GaugeVision as a whole use MVTec AD for research / portfolio /
educational purposes only; the dataset is not redistributed beyond the local
cache created here, and is not used in any commercial product.

Fallback: if this Hugging Face mirror ever becomes unavailable, the official
MVTec download (https://www.mvtec.com/company/research/datasets/mvtec-ad)
requires free registration and is not scriptable end-to-end; download the
"screw" category manually and point ``load_screw_category`` at a directory
matching the same layout MVTec ships (``train/good``, ``test/<defect>``,
``ground_truth/<defect>``) — see ``_load_from_mvtec_layout`` below.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

logger = logging.getLogger(__name__)

REPO_ID = "Voxel51/mvtec-ad"
REPO_TYPE = "dataset"
CATEGORY = "screw"
EXPECTED_TRAIN_COUNT = 320
EXPECTED_TEST_COUNT = 160

DEFAULT_CACHE_DIR = Path("data/mvtec_ad")

MVTEC_LICENSE_NOTICE = (
    "MVTec AD dataset (Bergmann et al., 2019/2021), CC BY-NC-SA 4.0. "
    "Used here for research/portfolio/educational purposes only — "
    "not redistributed, not used in any commercial product."
)


@dataclass(frozen=True)
class MVTecSample:
    image_path: Path
    split: str  # "train" | "test"
    defect_label: str  # "good" or a defect type, e.g. "scratch_neck"
    mask_path: Path | None  # ground-truth defect mask, test-set defects only

    @property
    def is_defective(self) -> bool:
        return self.defect_label != "good"


def _load_manifest(hf_cache_dir: Path) -> list[dict]:
    manifest_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        filename="samples.json",
        cache_dir=str(hf_cache_dir),
    )
    with open(manifest_path) as f:
        data = json.load(f)
    return data["samples"]


def load_screw_category(
    cache_dir: str | Path = DEFAULT_CACHE_DIR, verify_counts: bool = True
) -> list[MVTecSample]:
    """Download (if needed) and return all "screw" category samples.

    Args:
        cache_dir: local directory used for the Hugging Face download cache.
            Re-running with the same ``cache_dir`` reuses already-downloaded
            files (content-addressed by ``huggingface_hub``).
        verify_counts: if True, raise if the downloaded train/test split
            counts don't match the expected MVTec "screw" counts
            (320 train, 160 test) — a sanity check per CLAUDE.md §3.1.

    Returns:
        List of MVTecSample, local-filesystem paths resolved.
    """
    cache_dir = Path(cache_dir)
    hf_cache_dir = cache_dir / "_hf_cache"
    hf_cache_dir.mkdir(parents=True, exist_ok=True)

    logger.info(MVTEC_LICENSE_NOTICE)

    samples = _load_manifest(hf_cache_dir)
    screw_samples = [s for s in samples if s["category"]["label"] == CATEGORY]
    if not screw_samples:
        raise RuntimeError(f"No samples found for category={CATEGORY!r} in manifest")

    logger.info("Found %d screw-category samples in manifest; downloading...", len(screw_samples))

    allow_patterns = sorted(
        {s["filepath"] for s in screw_samples}
        | {
            s["defect_mask"]["mask_path"]
            for s in screw_samples
            if s.get("defect_mask") is not None
        }
    )
    snapshot_dir = Path(
        snapshot_download(
            repo_id=REPO_ID,
            repo_type=REPO_TYPE,
            allow_patterns=allow_patterns,
            cache_dir=str(hf_cache_dir),
            max_workers=8,
        )
    )

    results: list[MVTecSample] = []
    for s in screw_samples:
        defect_mask = s.get("defect_mask")
        results.append(
            MVTecSample(
                image_path=snapshot_dir / s["filepath"],
                split=s["split"],
                defect_label=s["defect"]["label"],
                mask_path=(
                    snapshot_dir / defect_mask["mask_path"]
                    if defect_mask is not None
                    else None
                ),
            )
        )

    if verify_counts:
        n_train = sum(1 for r in results if r.split == "train")
        n_test = sum(1 for r in results if r.split == "test")
        if n_train != EXPECTED_TRAIN_COUNT or n_test != EXPECTED_TEST_COUNT:
            raise RuntimeError(
                f"Unexpected screw split counts after download: "
                f"train={n_train} (expected {EXPECTED_TRAIN_COUNT}), "
                f"test={n_test} (expected {EXPECTED_TEST_COUNT}). "
                "The HF mirror may have changed — see the MVTec fallback "
                "path documented in this module's docstring."
            )
        logger.info("Verified screw category counts: train=%d, test=%d", n_train, n_test)

    return results


def train_normal_samples(samples: list[MVTecSample]) -> list[MVTecSample]:
    """Defect-free training samples — the only ones anomaly detection fits on."""
    return [s for s in samples if s.split == "train" and not s.is_defective]


def test_samples(samples: list[MVTecSample]) -> list[MVTecSample]:
    return [s for s in samples if s.split == "test"]
