"""Export fitted PaDiM statistics to a flat binary format for the C++ edge
demo — CLAUDE.md §4.8 (Phase 6).

The C++ demo (``edge/cpp/``) loads the exported ONNX backbone via the ONNX
Runtime C++ API and needs the fitted per-patch Gaussian statistics (mean,
inverse covariance, feature indices) to compute the same Mahalanobis
anomaly score the Python path does — but those are saved as a NumPy .npz
(a ZIP of .npy files), and adding a NumPy/zip-reading dependency to a
"minimal" C++ demo is more machinery than CLAUDE.md's Phase 6 scope calls
for. This writes a small custom flat binary format instead: a fixed-size
header of array dimensions, followed by raw float32/int32 arrays in a fixed
order — trivial to read with ~20 lines of C++ and no external format
dependency.

Run as: python -m gaugevision.deployment.export_cpp_stats
"""

from __future__ import annotations

import logging
import struct
from pathlib import Path

import numpy as np

from gaugevision.anomaly.padim import load_stats
from gaugevision.logging_config import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_STATS_PATH = Path("models/padim_stats_cpp.bin")
MAGIC = b"GVPD"  # GaugeVision PaDiM
FORMAT_VERSION = 1


def export_stats_for_cpp(
    model_path: str = "models/padim_screw.npz",
    output_path: Path = DEFAULT_STATS_PATH,
) -> dict:
    stats, threshold = load_stats(model_path)
    n_patches, d = stats.mean.shape
    grid_h, grid_w = stats.grid_size

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<i", FORMAT_VERSION))
        f.write(struct.pack("<iiii", n_patches, d, grid_h, grid_w))
        f.write(struct.pack("<f", threshold if threshold is not None else float("nan")))

        f.write(stats.feature_indices.astype(np.int32).tobytes())  # (d,)
        f.write(stats.mean.astype(np.float32).tobytes())  # (n_patches, d)
        f.write(stats.cov_inv.astype(np.float32).tobytes())  # (n_patches, d, d)

    logger.info(
        "Exported C++ stats: n_patches=%d d=%d grid=%dx%d -> %s (%d bytes)",
        n_patches,
        d,
        grid_h,
        grid_w,
        output_path,
        output_path.stat().st_size,
    )
    return {
        "output_path": str(output_path),
        "n_patches": n_patches,
        "d": d,
        "grid_size": (grid_h, grid_w),
        "size_bytes": output_path.stat().st_size,
    }


if __name__ == "__main__":
    configure_logging()
    result = export_stats_for_cpp()
    print(result)
