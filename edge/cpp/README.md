# GaugeVision — C++ edge inference demo

A minimal C++/ONNX Runtime edge inference implementation, demonstrating
deployment outside the Python runtime (CLAUDE.md §4.8, Phase 6). Not a port
of the project — no training, no measurement pipeline, no API — just an
image in, an anomaly score + latency out, using the exact same Mahalanobis
scoring math as the Python `PaDiM` class.

## What it does

1. Loads the PaDiM backbone exported to ONNX in Phase 2
   (`models/padim_backbone.onnx`) via the ONNX Runtime C++ API.
2. Loads fitted PaDiM statistics (mean/inverse-covariance/feature indices)
   from a small custom flat binary file — not the same `.npz` format the
   Python side uses, since parsing NumPy's zip-of-`.npy` format in C++
   would be more machinery than this "minimal" demo calls for. Produced by:
   ```bash
   python -m gaugevision.deployment.export_cpp_stats
   ```
3. Reads an image via OpenCV, preprocesses it (grayscale → 3-channel,
   resize to 224×224, ImageNet normalization — mirroring
   `gaugevision.anomaly.padim.preprocess`, though bilinear resize in OpenCV
   isn't bit-identical to PyTorch's `F.interpolate`, so scores are close to
   but not guaranteed identical to the Python path's for the same image).
4. Runs the ONNX Runtime session, computes the per-patch Mahalanobis
   distance against the fitted statistics, and reports the image-level
   score (max over patches) plus end-to-end latency (preprocess + inference
   + scoring).

## Build

Requires ONNX Runtime and OpenCV C++ development packages, and CMake ≥3.16.

```bash
brew install onnxruntime opencv cmake   # macOS
cmake -S edge/cpp -B edge/cpp/build
cmake --build edge/cpp/build
```

If ONNX Runtime isn't auto-discovered (e.g. the official prebuilt tarball
from [github.com/microsoft/onnxruntime/releases](https://github.com/microsoft/onnxruntime/releases)
instead of a package manager), pass its location explicitly:

```bash
cmake -S edge/cpp -B edge/cpp/build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
```

## Run

```bash
# From the repo root, with a trained model and exported backbone already present:
python -m gaugevision.deployment.export_onnx        # if not already run (Phase 2)
python -m gaugevision.deployment.export_cpp_stats    # writes models/padim_stats_cpp.bin

./edge/cpp/build/gaugevision_edge_demo \
    models/padim_backbone.onnx \
    models/padim_stats_cpp.bin \
    path/to/image.png [more images...]
```

Output: one line per image, e.g.
```
path/to/image.png                       score=14.5524  latency_ms=42.31
```
