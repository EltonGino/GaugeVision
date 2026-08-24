"""PyTorch vs ONNX Runtime CPU inference benchmark — CLAUDE.md §4.4 (Phase 2).

Produces every metric CLAUDE.md §4.4 asks for, all from the actually
implemented code — never a number pulled from memory:

- Output equivalence (PyTorch vs ONNX Runtime scores, on real MVTec images)
- Model size (PyTorch backbone state_dict vs ONNX file)
- Cold-start time
- Mean / p50 / p95 inference latency
- Throughput (images/sec)
- CPU inference (both backends forced to CPU / CPUExecutionProvider)
- Peak process memory (via stdlib ``resource``, no extra profiling tooling)

Latency/memory are measured by running each backend in its own subprocess
(``_bench_worker.py``) so peak RSS is attributable to one backend at a time,
and so neither backend's import/init affects the other's cold-start timing.

Run as: python -m gaugevision.deployment.benchmark
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from gaugevision.logging_config import configure_logging

logger = logging.getLogger(__name__)

N_ITERS = 50
N_WARMUP = 5
N_EQUIVALENCE_IMAGES = 20
SCORE_ATOL = 1e-2
SCORE_RTOL = 1e-3


def _run_worker(backend: str, n_iters: int = N_ITERS, n_warmup: int = N_WARMUP) -> dict:
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "gaugevision.deployment._bench_worker",
            backend,
            str(n_iters),
            str(n_warmup),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _latency_stats(latencies_ms: list[float]) -> dict:
    arr = np.array(latencies_ms)
    mean = float(arr.mean())
    return {
        "mean_ms": mean,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "throughput_images_per_sec": 1000.0 / mean if mean > 0 else float("inf"),
    }


def _check_output_equivalence(n_images: int = N_EQUIVALENCE_IMAGES) -> dict:
    from gaugevision.anomaly.padim import PaDiM
    from gaugevision.data.mvtec_loader import load_screw_category, test_samples
    from gaugevision.deployment.onnx_infer import ONNXPaDiM

    samples = test_samples(load_screw_category())[:n_images]
    images = [cv2.imread(str(s.image_path), cv2.IMREAD_GRAYSCALE) for s in samples]

    pt_model = PaDiM.load("models/padim_screw.npz")
    onnx_model = ONNXPaDiM("models/padim_backbone.onnx", "models/padim_screw.npz")

    pt_scores = np.array([pt_model.predict(img).score for img in images])
    onnx_scores = np.array([onnx_model.predict(img).score for img in images])

    abs_diff = np.abs(pt_scores - onnx_scores)
    rel_diff = abs_diff / np.maximum(np.abs(pt_scores), 1e-8)

    passed = bool(np.allclose(pt_scores, onnx_scores, atol=SCORE_ATOL, rtol=SCORE_RTOL))

    return {
        "n_images": n_images,
        "max_abs_diff": float(abs_diff.max()),
        "mean_abs_diff": float(abs_diff.mean()),
        "max_rel_diff": float(rel_diff.max()),
        "tolerance_atol": SCORE_ATOL,
        "tolerance_rtol": SCORE_RTOL,
        "passed": passed,
    }


def run_benchmark() -> dict:
    onnx_path = Path("models/padim_backbone.onnx")
    pt_path = Path("models/padim_backbone.pt")
    if not onnx_path.exists() or not pt_path.exists():
        raise RuntimeError(
            "Run `python -m gaugevision.deployment.export_onnx` first "
            "to produce models/padim_backbone.{onnx,pt}"
        )

    logger.info("Checking output equivalence on %d real MVTec test images...", N_EQUIVALENCE_IMAGES)
    equivalence = _check_output_equivalence()
    logger.info(
        "Equivalence: max_abs_diff=%.2e, max_rel_diff=%.2e, passed=%s",
        equivalence["max_abs_diff"],
        equivalence["max_rel_diff"],
        equivalence["passed"],
    )

    logger.info("Benchmarking PyTorch CPU (%d iters, %d warmup)...", N_ITERS, N_WARMUP)
    pt_result = _run_worker("pytorch")
    pt_stats = _latency_stats(pt_result["latencies_ms"])

    logger.info("Benchmarking ONNX Runtime CPU (%d iters, %d warmup)...", N_ITERS, N_WARMUP)
    onnx_result = _run_worker("onnx")
    onnx_stats = _latency_stats(onnx_result["latencies_ms"])

    speedup = pt_stats["mean_ms"] / onnx_stats["mean_ms"] if onnx_stats["mean_ms"] > 0 else float("inf")

    result = {
        "output_equivalence": equivalence,
        "model_size": {
            "pytorch_backbone_state_dict_bytes": pt_path.stat().st_size,
            "onnx_backbone_bytes": onnx_path.stat().st_size,
        },
        "pytorch_cpu": {
            "cold_start_ms": pt_result["cold_start_ms"],
            "max_rss_mb": pt_result["max_rss_mb"],
            **pt_stats,
        },
        "onnxruntime_cpu": {
            "cold_start_ms": onnx_result["cold_start_ms"],
            "max_rss_mb": onnx_result["max_rss_mb"],
            **onnx_stats,
        },
        "speedup_pytorch_over_onnx": speedup,
        "n_iters": N_ITERS,
        "n_warmup": N_WARMUP,
    }
    return result


if __name__ == "__main__":
    configure_logging()
    result = run_benchmark()
    print(json.dumps(result, indent=2))
