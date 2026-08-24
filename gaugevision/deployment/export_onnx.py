"""PyTorch -> ONNX export — CLAUDE.md §4.4 (Phase 2).

Only the PaDiM backbone (frozen ResNet18 feature extraction + multiscale
concat, ``PaDiMBackbone``) is exported to ONNX. PaDiM's Gaussian-distribution
Mahalanobis scoring stays in NumPy (``gaugevision.anomaly.padim.
mahalanobis_score``) and runs identically on top of either backend's
extracted features — per CLAUDE.md §4.4's explicit scope note that PaDiM's
scoring logic may not translate cleanly through the ONNX exporter, and a
backbone-in-ONNX + scoring-in-NumPy split is an acceptable architecture.
Document which parts run where; don't claim a full-pipeline ONNX benchmark.

A PyTorch backbone checkpoint (``models/padim_backbone.pt``, the state_dict)
is also saved alongside the ONNX export purely so the "model size" benchmark
bullet in CLAUDE.md §4.4 has a real PyTorch artifact to compare against — the
Phase-1 training flow (``anomaly/train.py``) never persists backbone weights
on its own, since they're loaded fresh from torchvision's pretrained
ImageNet checkpoint each run.

Run as: python -m gaugevision.deployment.export_onnx
"""

from __future__ import annotations

import logging
from pathlib import Path

import onnx
import torch

from gaugevision.anomaly.padim import PaDiMBackbone
from gaugevision.logging_config import configure_logging

logger = logging.getLogger(__name__)

DEFAULT_ONNX_PATH = Path("models/padim_backbone.onnx")
DEFAULT_PT_PATH = Path("models/padim_backbone.pt")
DEFAULT_INPUT_SIZE = (224, 224)
ONNX_OPSET = 17


def export_backbone(
    onnx_path: Path = DEFAULT_ONNX_PATH,
    pt_path: Path = DEFAULT_PT_PATH,
    input_size: tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> dict:
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    pt_path.parent.mkdir(parents=True, exist_ok=True)

    backbone = PaDiMBackbone()
    backbone.eval()

    torch.save(backbone.state_dict(), pt_path)
    logger.info("Saved PyTorch backbone state_dict to %s", pt_path)

    dummy_input = torch.zeros(1, 3, *input_size, dtype=torch.float32)

    torch.onnx.export(
        backbone,
        (dummy_input,),
        str(onnx_path),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=ONNX_OPSET,
        dynamo=False,
    )
    logger.info("Exported ONNX backbone to %s (opset %d)", onnx_path, ONNX_OPSET)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX model passed onnx.checker validation")

    return {
        "onnx_path": str(onnx_path),
        "pt_path": str(pt_path),
        "onnx_size_bytes": onnx_path.stat().st_size,
        "pt_size_bytes": pt_path.stat().st_size,
        "opset": ONNX_OPSET,
        "input_size": input_size,
    }


if __name__ == "__main__":
    configure_logging()
    result = export_backbone()
    print(result)
