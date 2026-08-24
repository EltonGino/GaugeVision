# GaugeVision

An industrial visual-inspection pipeline for threaded parts (screws) that fuses
classical dimensional metrology, deep-learning anomaly detection, and served
inference into an auditable Go/No-Go decision.

**This is a demonstration pipeline built on a public dataset and synthetic
validation data — not a claim of working with real engine-block or production
manufacturing data.** It's built to demonstrate the technical skill cluster
(camera/metric calibration, classical CV measurement, CNN-based anomaly
detection, optimized inference, API/container serving) behind that kind of
industrial-inspection system, using an openly available screw dataset instead
of pretending to have access to proprietary hardware or data.

## Current status: Phase 3 — video input

The full architecture (see below) is wired end-to-end on stable interfaces:
`image → measurement pipeline → anomaly detection → decision fusion → FastAPI
→ Docker → Gradio` (Phase 1), a benchmarked PyTorch-vs-ONNX-Runtime CPU
inference path (Phase 2, see `docs/RESULTS.md`), and file-based video input
with an annotated output video and temporal aggregation (Phase 3, §4.7
below). Validation depth (synthetic-ground-truth measurement error, a second
pitch-estimation strategy, PatchCore) is still layered in over the remaining
phases described in `CLAUDE.md`.

## Architecture

```
                 ┌─────────────────────────────────────────┐
                 │              Measurement (§4.2)          │
image ──────────▶│ segment → axis → derotate → thread       │───┐
                 │ region → profile → diameter → pitch      │   │
                 │      (px → mm via MetricCalibration)      │   │
                 └─────────────────────────────────────────┘   │
                                                                  ▼
                 ┌─────────────────────────────────────────┐  ┌──────────────┐
                 │         Anomaly detection (§4.3)          │  │  Decision    │
image ──────────▶│  PaDiM: frozen ResNet18 + per-patch        │─▶│  fusion      │──▶ Verdict
                 │  Gaussian, fit on normal training images   │  │  (§4.5)      │
                 └─────────────────────────────────────────┘  └──────────────┘

Gradio UI ──HTTP──▶ FastAPI (/inspect/image, /health, /model/info) ──▶ pipeline above
```

**Video input** (CLAUDE.md §4.7, Phase 3) is not a separate algorithm — it's
the pipeline above run per sampled frame, plus temporal aggregation:
`video file → OpenCV frame sampling (~2 samples/sec by default) → the same
measurement + anomaly + decision pipeline per frame → worst-case aggregation
(NO_GO if any sampled frame fails a check) → annotated output video`. Served
via `POST /inspect/video`, which reuses `run_measurement_pipeline`, `PaDiM.
predict`, and `fuse_verdict` directly rather than a parallel implementation.
**This is file-based video input, not live/streaming ingestion** — there is
no RTSP or real-time camera path here (that's a stretch goal, CLAUDE.md §9).

Two calibration concepts are kept deliberately separate (CLAUDE.md §4.1):

- **Lens/geometric calibration** (`calibration/lens_calibration.py`): standard
  OpenCV chessboard calibration. There's no physical camera behind this
  project, so it runs against a procedurally rendered synthetic checkerboard
  set — a **capability demonstration**, not a calibration of any real camera.
- **Metric (px→mm) calibration** (`calibration/metric_calibration.py`):
  a *separate* problem — recovering real-world scale from a reference object.
  Every pixel↔mm conversion in the measurement pipeline routes through the
  `MetricCalibration` class; no bare pixel/constant division is scattered
  through measurement code.

## Known Phase-1 limitations (read before trusting a number)

- **Measurements are not dimensionally validated.** MVTec AD provides no
  physical scale reference, so Phase 1 uses a configured demonstration
  `MetricCalibration` (`px_per_mm=32.4`, `source="demo_reference"`). Every
  `MeasurementResult` has `calibrated=False` and carries this fact in its
  `notes`. Phase 4 replaces this with a scale validated against a synthetic
  thread generator with known ground truth.
- **Only one pitch estimator is implemented** (`PeakPitchEstimator`, spacing
  between thread-crest peaks). `FFTPitchEstimator` exists as an interface stub
  and raises `NotImplementedError` — it's a Phase 4 deliverable, along with
  the quantitative Peak-vs-FFT comparison table.
- **Pitch estimation does not reliably fire on real MVTec images** (0/15
  sampled test images produced a pitch estimate; `confidence=0.0`,
  `pitch_px=None`). Diagnosis: the per-row silhouette-width signal
  (`profile.compute_width_profile`) is dominated by the screw's overall
  head/taper envelope, not the fine thread-crest oscillation — real thread
  ridges are a few px of amplitude against a silhouette whose width already
  varies by tens of px along the shank from taper alone, so the peak
  detector correctly reports "no reliable periodicity" rather than
  fabricating a number. It works as intended on the unit-test/synthetic
  suite (`tests/test_pitch.py`, `tests/test_measurement_pipeline.py`),
  where thread amplitude isn't confounded with taper. Major-diameter
  estimation and anomaly detection are unaffected — this only removes
  pitch/pitch_mm from the output. Robust thread-crest profile extraction
  (e.g. edge-based rather than fill-width-based) is Phase-4-scope work per
  CLAUDE.md §11's explicit anticipation that this sub-problem could be
  harder than it looks.
- **Head/shank separation is a simple heuristic** (exclude a configurable
  margin from the widest row, assumed to be the head), not a robust
  axis/derotation system. This is flagged in `CLAUDE.md` §11 as a
  sub-problem that may be harder than it looks on real MVTec images; Phase 1
  ships the simplest defensible version rather than over-building it.
- **The ISO 965 tolerance table is a starter reference**, class 6g only, for
  M1.6–M12, cross-referenced against public engineering tables (not
  independently re-derived from the ISO 965-1 standard text). See
  `decision/iso965_table.py` for sourcing notes. The decision layer performs
  "ISO-informed dimensional validation," **not** certified ISO 965 compliance
  verification — pitch diameter and full thread-profile geometry aren't
  measured here.
- **Anomaly heatmap resolution is coarser than the original PaDiM paper**
  (28×28 patch grid instead of 56×56) — a documented CPU-feasibility
  tradeoff (see `anomaly/padim.py`).

## Dataset

**MVTec AD, "screw" category** (Bergmann et al., *MVTec AD — A Comprehensive
Real-World Dataset for Unsupervised Anomaly Detection*, 2019, and its 2021
extension). **License: CC BY-NC-SA 4.0 — non-commercial, share-alike,
attribution required.** Used here for research/portfolio/educational purposes
only; not redistributed beyond the local cache this loader creates, and not
used in any commercial product.

`gaugevision/data/mvtec_loader.py` downloads only the screw-category files
(320 defect-free train images, 160 test images spanning good + five defect
types) from the `Voxel51/mvtec-ad` Hugging Face mirror, using
`huggingface_hub` directly against that dataset's FiftyOne sample manifest
(rather than pulling the full 15-category dataset through the heavier
`fiftyone` runtime). It verifies the expected 320/160 train/test split counts
after download. If this mirror ever becomes unavailable, the official MVTec
download (registration required, not scriptable) is documented as a fallback
in the loader's docstring.

Training (anomaly detection) uses only defect-free images — the standard
one-class / unsupervised-anomaly-detection paradigm, since real defective
samples are rare and hard to source, mirroring how production inspection
lines actually work.

## Setup

Requires **Python 3.11** (PyTorch does not yet support newer CPython
releases at time of writing).

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 1. Download the dataset

```bash
python -c "from gaugevision.data.mvtec_loader import load_screw_category; load_screw_category()"
```

### 2. Fit the PaDiM anomaly model

```bash
python -m gaugevision.anomaly.train
```

Writes `models/padim_screw.npz` and prints the fitted threshold and held-out
test-set image-level AUROC.

### 3. Run the API

```bash
uvicorn gaugevision.api.main:app --reload
```

- `GET /health`
- `GET /model/info`
- `POST /inspect/image` (multipart file upload)
- `POST /inspect/video` (multipart file upload — video input, not streaming;
  see §4.7 below)

### 4. Run the Gradio demo

```bash
python -m gaugevision.app.demo
```

Open http://localhost:7860. The UI has an Image tab and a Video tab; both
call the FastAPI service over HTTP — set `GAUGEVISION_API_URL` if the API
isn't on `localhost:8000`.

### 5. ONNX export + inference benchmark (Phase 2)

```bash
python -m gaugevision.deployment.export_onnx
python -m gaugevision.deployment.benchmark
```

Exports the PaDiM backbone to `models/padim_backbone.onnx`, then benchmarks
PyTorch vs ONNX Runtime CPU inference (output equivalence, model size,
cold-start, latency percentiles, throughput, peak memory) — see
`docs/RESULTS.md` for the numbers this produced. Only the backbone is
exported; scoring stays in NumPy — see `anomaly/padim.py` and
`deployment/export_onnx.py` docstrings for why.

## Docker

```bash
docker compose up --build
```

Starts the API on `:8000` and the Gradio UI on `:7860`. The model checkpoint
(`models/padim_screw.npz`) must exist before building/running — run step 2
above first, or mount it at `/app/models/padim_screw.npz`.

## Tests

```bash
pytest tests/ -q
```

Focused on deterministic contracts and processing stages: calibration math,
segmentation/axis estimation on synthetic silhouettes, pitch estimation
against synthetic periodic signals, the full measurement pipeline on a
synthetic screw, decision-fusion logic against the ISO tolerance table,
PaDiM's shared Mahalanobis-scoring/preprocessing math (the same functions
both the PyTorch and ONNX Runtime inference paths call), and the video
pipeline (frame sampling interval, temporal aggregation, annotated-video
writing) against a synthesized video — MVTec AD has no video assets, so this
mirrors the synthetic-checkerboard precedent already used for lens
calibration (§4.1a). The ONNX export/benchmark itself is exercised manually
(see step 5 above) rather than in the fast CI suite, since it needs a fitted
model checkpoint and downloads pretrained ImageNet weights.

## What's next (Phase 4+)

See `CLAUDE.md` §7 for the full phase plan: a synthetic-thread validation
harness with the Peak-vs-FFT comparison table and a validated metric
calibration source (Phase 4), CI + polish (Phase 5), and a
minimal C++/ONNX Runtime edge-inference demo (Phase 6).
