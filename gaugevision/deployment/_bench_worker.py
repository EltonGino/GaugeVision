"""Internal single-backend latency/memory worker for benchmark.py.

Run in its own subprocess (one per backend) so peak RSS is attributable to
that backend alone, rather than a cumulative number across both PyTorch and
ONNX Runtime loaded in the same long-lived process. Not part of the public
API — invoked via ``python -m gaugevision.deployment._bench_worker``.
"""

from __future__ import annotations

import json
import resource
import sys
import time

import numpy as np


def _max_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # ru_maxrss units differ by platform: bytes on macOS (Darwin), KB on Linux.
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024


def _make_dummy_image(size: int = 400) -> np.ndarray:
    rng = np.random.default_rng(0)
    return (rng.random((size, size)) * 255).astype(np.uint8)


def run_pytorch(n_iters: int, n_warmup: int) -> dict:
    t0 = time.perf_counter()
    from gaugevision.anomaly.padim import PaDiM

    model = PaDiM.load("models/padim_screw.npz")
    cold_start_ms = (time.perf_counter() - t0) * 1000.0

    image = _make_dummy_image()
    for _ in range(n_warmup):
        model.predict(image)

    latencies_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        model.predict(image)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "backend": "pytorch",
        "cold_start_ms": cold_start_ms,
        "latencies_ms": latencies_ms,
        "max_rss_mb": _max_rss_mb(),
    }


def run_onnx(n_iters: int, n_warmup: int) -> dict:
    t0 = time.perf_counter()
    from gaugevision.deployment.onnx_infer import ONNXPaDiM

    model = ONNXPaDiM("models/padim_backbone.onnx", "models/padim_screw.npz")
    cold_start_ms = (time.perf_counter() - t0) * 1000.0

    image = _make_dummy_image()
    for _ in range(n_warmup):
        model.predict(image)

    latencies_ms = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        model.predict(image)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "backend": "onnxruntime_cpu",
        "cold_start_ms": cold_start_ms,
        "latencies_ms": latencies_ms,
        "max_rss_mb": _max_rss_mb(),
    }


if __name__ == "__main__":
    backend, n_iters, n_warmup = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    result = run_pytorch(n_iters, n_warmup) if backend == "pytorch" else run_onnx(n_iters, n_warmup)
    print(json.dumps(result))
