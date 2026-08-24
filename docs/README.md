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

## Current status: Phase 4 — depth pass

The full architecture is wired end-to-end: `image → measurement pipeline →
anomaly detection → decision fusion → FastAPI → Docker → Gradio` (Phase 1),
a benchmarked PyTorch-vs-ONNX-Runtime CPU inference path (Phase 2), and
file-based video input with an annotated output video and temporal
aggregation (Phase 3). Phase 4 adds the validation depth CLAUDE.md scoped for
this stage: a synthetic thread generator with known ground truth,
`FFTPitchEstimator` implemented and quantitatively compared against
`PeakPitchEstimator`, a PatchCore anomaly-detection stretch model benchmarked
against PaDiM, and a validated calibration source used within that
validation harness. **All real numbers are in `docs/RESULTS.md`** — several
Phase 4 findings genuinely changed decisions rather than just confirming
Phase 1/2/3 choices (see Known Limitations below).

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

## Known limitations (read before trusting a number)

- **Real MVTec-image measurements are still not dimensionally validated,
  and this is intentional, not an oversight.** MVTec AD provides no physical
  scale reference in its photos, so `/inspect/image` and `/inspect/video`
  use a configured demonstration `MetricCalibration`
  (`px_per_mm=32.4`, `source="demo_reference"`, `calibrated=False`) — every
  `MeasurementResult` carries this fact in its `notes`. Phase 4 added a
  second, **validated** calibration source
  (`MetricCalibration.synthetic_reference`), but it is deliberately used
  only within the measurement-validation harness
  (`measurement/validate.py`), against synthetic images that actually
  contain a rendered reference object of known size — never applied to real
  MVTec images, which have no such object to calibrate against. Applying a
  synthetic scale to real MVTec pixels would fabricate a number with no
  relationship to those photos' actual scale; see
  `calibration/metric_calibration.py`'s docstring for the full reasoning.
- **Two pitch estimators are implemented and quantitatively compared**
  (`PeakPitchEstimator`, `FFTPitchEstimator`) — see `docs/RESULTS.md` for
  the full comparison table. **The surprising result: `FFTPitchEstimator`
  wins on synthetic ground truth (100% coverage, small errors) but loses on
  real MVTec images**, where it always reports a confident-looking estimate
  from what is actually segmentation noise (values ranging 32-178px across
  visually similar screws, all at `confidence=1.00`). `PeakPitchEstimator`
  correctly reports "no reliable estimate" on those same images instead of
  guessing. **`PeakPitchEstimator` remains the pipeline's deployed
  default** because its failure mode (honest "I don't know") is safer for a
  QC decision system than FFT's (confident and wrong) — this is a case
  where evidence from the real target dataset overrides evidence from
  synthetic data alone.
- **Pitch estimation still does not reliably fire on real MVTec images**
  (0/15 sampled test images produced a `PeakPitchEstimator` pitch estimate).
  Root cause, confirmed rather than just diagnosed in Phase 4: the per-row
  silhouette-width signal (`profile.compute_width_profile`) is dominated by
  the screw's head/taper envelope and segmentation noise, not fine
  thread-crest oscillation — this holds for *both* pitch estimators (see
  above), so the fix belongs in upstream profile extraction (e.g.
  edge-based rather than fill-width-based thread-crest isolation), not in
  either estimator's own algorithm. Confirmed working correctly on synthetic
  ground truth where the signal genuinely contains periodicity (see
  `docs/RESULTS.md`'s validation table) — both estimators are sound, the
  input signal on real images just doesn't carry the information they need.
  Major-diameter estimation and anomaly detection are unaffected.
- **A real bug in `isolate_thread_region` was found and fixed during Phase
  4 validation**: its head-exclusion margin was computed relative to the
  whole derotated frame length rather than the part's own foreground
  extent, which silently inflated the measured major diameter to the head's
  width whenever there was background margin around the part. Invisible in
  Phase 1's own unit tests (screw nearly filled its test canvas); exposed by
  the more realistic canvas layout in the Phase 4 synthetic-thread
  validation harness. See `measurement/profile.py` and `docs/RESULTS.md`
  for details — this is exactly the kind of thing a synthetic validation
  harness with independently-known ground truth is supposed to catch.
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
- **PatchCore (Phase 4 stretch) underperformed PaDiM** in this project's
  configuration (0.7208 vs 0.7569 image-level AUROC on the MVTec "screw"
  test set) — expected given it deliberately reuses PaDiM's ResNet18
  feature representation rather than PatchCore's own WideResNet50 +
  local-aggregation setup from the original paper, for a controlled
  same-features comparison. PaDiM remains the model behind the live API.
  See `docs/RESULTS.md` for the full reasoning and the published-paper
  comparison numbers for both models.

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

### 6. Measurement validation against synthetic ground truth (Phase 4)

```python
from gaugevision.measurement.validate import run_validation_sweep, summarize, format_pitch_comparison_table
records = run_validation_sweep()
print(format_pitch_comparison_table(summarize(records)))
```

Runs the measurement pipeline against `data/synthetic_thread.py` samples
with known ground-truth dimensions across a sweep of thread sizes and
clean/blur/rotated conditions, and produces the `PeakPitchEstimator` vs
`FFTPitchEstimator` comparison table — see `docs/RESULTS.md` for the full
results and the (non-obvious) default-estimator decision this evidence
drove.

### 7. Fit the PatchCore anomaly model (Phase 4 stretch)

```bash
python -m gaugevision.anomaly.train_patchcore
```

Writes `models/patchcore_screw.npz` and prints the held-out test-set
image-level AUROC, benchmarked against PaDiM in `docs/RESULTS.md`. Not
wired into the live API — PaDiM remains the deployed model (see Known
Limitations below for why).

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
against synthetic periodic signals (both estimators), the full measurement
pipeline on synthetic screws (including the physically-dimensioned Phase 4
generator with a rendered reference object), decision-fusion logic against
the ISO tolerance table, PaDiM's and PatchCore's shared math (greedy
coreset selection, Mahalanobis scoring, preprocessing — the same functions
both the PyTorch and ONNX Runtime inference paths call), and the video
pipeline (frame sampling interval, temporal aggregation, annotated-video
writing) against a synthesized video — MVTec AD has no video or dimensional
ground-truth assets, so all of this mirrors the synthetic-checkerboard
precedent already used for lens calibration (§4.1a). The ONNX
export/benchmark and the full PaDiM/PatchCore training runs are exercised
manually (see the numbered setup steps above) rather than in the fast CI
suite, since they need dataset downloads, pretrained ImageNet weights, and
in PatchCore's case several minutes of coreset selection.

## What's next (Phase 5+)

See `CLAUDE.md` §7 for the full phase plan: CI + structured logging + final
polish (Phase 5), and a minimal C++/ONNX Runtime edge-inference demo
(Phase 6).
