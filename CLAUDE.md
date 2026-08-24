# GaugeVision — CLAUDE.md

## 0. Purpose of this document
This is the architecture and build brief for **Claude Code**. Claude (chat) has done the planning; Claude Code owns all implementation — repo scaffolding, actual code, tests, and iteration. Do not re-litigate the architecture below without flagging the tradeoff first; do feel free to push back if something is impractical once you're in the code.

**Revision note:** this replaces an earlier "full flagship, no MVP" version of this doc. Elton is in an active interview process with CESAR right now — the plan below gets a working, deployed vertical slice out as early as possible, built on final interfaces rather than throwaway code, with validation depth added in layers after that. Nothing below was cut, only reordered/reframed.

## 1. Project overview

**Working name:** GaugeVision
**One-liner:** An industrial visual-inspection pipeline for threaded parts (screws) that fuses classical dimensional metrology, deep-learning anomaly detection, and optimized/served inference into an auditable Go/No-Go decision — mirroring the kind of system built for automated thread/bore inspection in manufacturing (engine-block-style QC), and explicitly covering the integration/deployment/optimization side the CESAR posting emphasizes, not just the modeling side.

**Why this project exists:** built as a portfolio piece to demonstrate readiness for computer-vision roles in industrial inspection — camera/metric calibration, classical CV measurement, CNN-based anomaly detection, optimized inference (ONNX), served via API + container, plus a small C++ edge-inference demo. This is a *demonstration* pipeline (public dataset + synthetic validation), not a claim of working with real engine-block data — say so explicitly and early in the README. Using screws and public data rather than pretending to have engine-block access is a feature, not a limitation: it shows the same technical principles honestly.

**Build order: final architecture early, depth in layers.** Two different things are being sequenced here, not one: (a) architecture/interfaces should be right from commit one — that's cheap and prevents rework; (b) implementation *depth* (validation rigor, estimator comparisons, robustness) is what actually gets layered in over phases. Get `image → pipeline → verdict → API → Docker` working end-to-end on top of stable interfaces before deepening any one module. See §7. An interview could happen any day — a working, well-architected repo beats a deeper but incomplete one, and a repo that has to be restructured later is worse than either.

## 2. Success criteria

- A working pipeline that takes a screw image (and, after phase 3, a video) and returns: (a) an anomaly score + heatmap, (b) a `MeasurementResult` (major diameter, pitch, calibration status, confidence — see §4.2), (c) a fused Go/No-Go verdict with reasoning (which check failed, if any), (d) inference latency.
- Served behind a FastAPI layer, containerized with Docker, with a Gradio UI on top calling the API (not calling the pipeline directly) — demonstrates the API/serving/container skill cluster, not just the modeling skill cluster.
- PyTorch → ONNX → ONNX Runtime conversion with a **verified, reproducible** benchmark: output equivalence between PyTorch and ONNX, model size, cold-start time, mean/p50/p95 latency, throughput, CPU inference (memory if easy to measure without extra tooling).
- Anomaly detection benchmarked against published PaDiM/PatchCore MVTec-AD "screw" results — **using the same metric and evaluation protocol as the source**, not a number pulled from memory. Don't write a benchmark figure into RESULTS.md until the source table has been checked directly (image-level AUROC and pixel-level AUROC are very different numbers and are easy to conflate across papers).
- Measurement module validated against **known ground truth** (synthetic threads), with quantified error (mean absolute error in mm/px) and a documented comparison between the two pitch-estimation strategies — this validation depth is phase-4 work, but the interfaces it validates exist from phase 1 (§4.2).
- Video input supported as a second entry point (file-based, not real-time streaming) with an annotated output video (heatmap, measurements, FPS, verdict).
- A small C++/ONNX Runtime edge-inference demo (image in → score + latency out) as a deployment-outside-Python proof point.
- Everything runs on free tooling — no paid API keys, no paid compute required for the core pipeline (see §5 and §10).

## 3. Data

### 3.1 Primary dataset — MVTec AD, "screw" category (public)
- Standard industrial anomaly-detection benchmark. The "screw" category: ~320 defect-free training images, ~160 test images (mix of normal + defective: e.g. scratch neck, scratch head, thread damage, manipulated front). Pixel-level defect masks are included for the test-set anomalies, so localization can be evaluated too, not just image-level classification.
- **License: CC BY-NC-SA 4.0 — non-commercial, share-alike, attribution required.** README must state this explicitly: dataset used for research/portfolio/educational purposes only, not redistributed, not used in any commercial product. Cite MVTec (Bergmann et al., MVTec AD, 2019/2021) in the README.
- **Acquisition/loading mechanism (must be scripted, not manual):**
  - Primary path: pull the "screw" category only from the Hugging Face mirror `Voxel51/mvtec-ad` (confirmed live at planning time) via the `datasets` or `fiftyone` library — scriptable, no registration wall, keeps `data/mvtec_loader.py` fully automated. The mirror redistributes the same CC BY-NC-SA 4.0-licensed data; keep the attribution/license note regardless of which mirror is used.
  - Fallback path: the official MVTec download (mvtec.com research/datasets page) requires a free registration form and is not scriptable end-to-end — document it in the README as the fallback if the HF mirror ever becomes unavailable, but don't build the pipeline's automation around it.
  - `mvtec_loader.py` should filter to the "screw" category at download time (don't pull all 15 categories), cache locally, and verify the expected train/test counts (~320 train, ~160 test) after download as a sanity check.
- Train split (defect-free only) is used for the anomaly-detection module in the standard one-class / unsupervised-anomaly-detection paradigm — this is *industry-standard practice* for this problem (defective samples are rare and hard to source, so you model "normal" and flag deviations), and it's worth calling out explicitly in the writeup since it mirrors how real inspection lines work.

### 3.2 Secondary data — synthetic thread generator (for measurement ground truth)
MVTec has no dimensional ground truth (no known real-world mm scale, no annotated pitch/diameter). Since the "public dataset" choice covers the anomaly-detection half well but not the measurement half, the measurement module needs a source of *known* dimensions to validate against. Solution: a small synthetic thread-image generator (OpenCV/PIL, procedurally drawn thread profiles at known pixel-per-mm scale, known pitch, known major/minor diameter, with controllable noise/blur/rotation). This is standard practice for validating a measurement algorithm before pointing it at real photos — the synthetic set is not a substitute dataset, it's a calibration/unit-test harness for the measurement math. Real MVTec screw images are still used for the qualitative/demo pass of the measurement module.

**This is phase-4 depth work (§7) — it does not block the phase-1 vertical slice.** Phase 1 already uses the final `MetricCalibration` abstraction (§4.1b) with a configured demonstration reference scale (e.g. `MetricCalibration(px_per_mm=32.4, source="demo_reference")`); phase 4 replaces the demonstration calibration source with a validated synthetic/known-reference calibration and quantifies measurement error. No measurement logic should depend on hard-coded px/mm constants outside the calibration component — the interface is stable from commit one, only the calibration *source* and validation depth change between phase 1 and phase 4.

## 4. System architecture

Eight components (§4.1–§4.8), organized into a core inspection pipeline — calibration, measurement, and anomaly detection (§4.1–4.3), plus decision fusion (§4.5) — and a deployment/optimization/serving layer: ONNX inference (§4.4), the FastAPI + Docker serving layer (§4.6), video input (§4.7), and the C++ edge demo (§4.8). That second group is what the CESAR posting emphasizes explicitly (ONNX/ONNX Runtime appear twice in the posting, plus optimization, latency, Cloud/Edge, C/C++).

### 4.1 Calibration — two distinct sub-steps, not one
Keep these conceptually and code-wise separate; `cv2.calibrateCamera` alone does **not** convert pixels to millimeters.

- **4.1a Lens/geometric calibration:** standard OpenCV chessboard calibration (`cv2.calibrateCamera`) → intrinsics + distortion coefficients → undistortion. Since there's no physical camera, use OpenCV's own sample calibration image sets (or a rendered synthetic checkerboard set). Document clearly that this is a **capability demonstration**, not a calibration of a specific real camera.
- **4.1b Metric/scale calibration:** a *separate* step — recovering px→mm scale from a planar reference object of known real-world size in the scene (or a known-dimension target / homography), independent of the lens-distortion step. This is the piece that actually lets diameter/pitch be reported in mm. Make the README explicit that these are two different calibration problems that are commonly conflated.
- **Stable interface from Phase 1:** define a `MetricCalibration` class (`from_reference(...)`, `pixels_to_mm(...)`, `mm_to_pixels(...)`) and route every measurement conversion through it — never a bare pixel/constant division inline in measurement code. Phase 1 constructs it from a configured demonstration reference (`MetricCalibration(px_per_mm=32.4, source="demo_reference")`); phase 4 constructs it from the validated synthetic/known-reference calibration instead. Same interface, different source — no rewrite needed later.

### 4.2 Dimensional measurement module
Classical CV, no deep learning. **Define the full pipeline and the output contract in Phase 1** — implement the simplest correct version behind each stage; Phase 4 deepens the implementation and validates it, it doesn't restructure it.

**Pipeline (fixed from Phase 1):**
`Image → lens correction (§4.1a) → ROI/segmentation → axis estimation → thread-region isolation → thread-profile extraction → major-diameter estimation → pitch estimation → metric calibration (§4.1b) → MeasurementResult`

**Stable output contract, defined in Phase 1:**
```python
MeasurementResult(
    major_diameter_px=...,
    major_diameter_mm=...,
    pitch_px=...,
    pitch_mm=...,
    scale_px_per_mm=...,
    calibrated: bool,
    confidence: float,
)
```

**Segmentation & diameter (straightforward):** threshold + morphological cleanup (Otsu/adaptive works well against MVTec's controlled backgrounds) → contour extraction → major diameter via min-enclosing-circle or ellipse fit on the shank/head region.

**Pitch estimation is the harder sub-problem — treat it as one.** The real difficulty isn't "run an FFT on an image," it's obtaining a reliable periodic signal from the thread geometry despite pose, noise, and segmentation error. Break it down explicitly:
1. Estimate the screw's longitudinal axis.
2. Normalize/derotate the piece to that axis.
3. Isolate the region that actually contains thread (vs. head/shank).
4. Extract the upper/lower thread-crest profile along the axis.
5. Convert that profile into a 1D signal.
6. Estimate pitch from the signal.
7. Flag when the estimate isn't reliable (feeds `confidence` in `MeasurementResult`).

Implement pitch estimation behind a common interface with two competing strategies, compared quantitatively rather than picked by convenience:
```python
class PitchEstimator:
    def estimate(self, profile) -> PitchEstimate: ...

class PeakPitchEstimator(PitchEstimator): ...   # peak-to-peak spacing of thread crests
class FFTPitchEstimator(PitchEstimator): ...    # periodicity via FFT of the 1D signal
```
Phase 4 compares both against synthetic ground truth across conditions and reports a table in RESULTS.md, e.g.:

| Method | Clean MAE | Blur MAE | Rotated MAE |
|---|---|---|---|
| Peak spacing | … | … | … |
| FFT | … | … | … |

Pick the default estimator from that evidence, not from convenience — and keep the other estimator in the repo behind the same interface as a documented comparison, not deleted.

**Convert pixel measurements → mm using the §4.1b `MetricCalibration`** (phase 1: demonstration reference source; phase 4: validated source) — never a bare pixel/constant division inline.

**Phase 4:** validate against synthetic ground truth: report mean absolute error and % error for diameter and pitch across a range of synthetic thread sizes and clean/blur/rotated conditions (the table above), and confirm the `confidence` flag actually correlates with higher error when it fires.

*Scope guardrail — this matters as much as the interface design does: shipping the full architecture above does not mean both pitch estimators, or a robust axis/derotation step, need to be production-solid in Phase 1. Phase 1 ships one working `PitchEstimator` implementation and a correct-if-simple pass through every other stage; Phase 4 adds the second estimator, the robustness work, and the comparison table. The interface exists from commit one so nothing gets rewritten — but don't let defining it turn into over-building axis estimation or derotation before the vertical slice (§7 Phase 1) is out the door. That sub-problem is genuinely harder than it looks; budget for it accordingly rather than assuming it's a quick pass.*

### 4.3 Anomaly / defect detection module
- **Baseline: PaDiM** (patch distribution modeling over a frozen pretrained CNN backbone — ResNet18 or WideResNet50 from torchvision, ImageNet weights, no training required beyond fitting per-patch Gaussian statistics on the normal-only training set). Cheap, CPU-friendly, well-documented, good first target — this is the phase-1 model.
- **Stretch: PatchCore** (coreset-reduced memory bank of patch features + nearest-neighbor scoring) — typically stronger than PaDiM, still training-free on top of a frozen backbone, still CPU-feasible for a dataset this size. Phase-4 addition if time allows after PaDiM is working, served, and benchmarked.
- Output per image: anomaly score (image-level) + anomaly heatmap (pixel-level localization), thresholded against a score computed from the normal training distribution.
- **Evaluation — be precise about the metric before publishing anything:** benchmark image-level AUROC against MVTec's labeled test set. Before writing a number into RESULTS.md, find the specific published table for the specific method (PaDiM vs PatchCore), the specific backbone, and confirm whether it's reporting image-level or pixel-level AUROC — these differ substantially and are easy to conflate. Until that's confirmed, the README should say "compared against published PaDiM/PatchCore results using the same metric and evaluation protocol" rather than quote a number.

### 4.4 Deployment & optimized inference (core, not stretch)
This directly answers the posting's repeated emphasis on ONNX/ONNX Runtime, inference optimization, latency, and Cloud/Edge.
- Export the PaDiM (and later PatchCore) backbone + scoring logic to **ONNX**, run via **ONNX Runtime** (CPU execution provider — no GPU assumed).
- **Scope note for whoever implements this:** exporting the CNN backbone (feature extraction) to ONNX is straightforward; exporting PaDiM's Gaussian-distribution scoring logic is a different problem and may not translate cleanly through the ONNX exporter. If full end-to-end ONNX export starts requiring unreasonable workarounds, a backbone-in-ONNX-Runtime + scoring-in-NumPy/Python split is an acceptable architecture — the goal is demonstrating inference-optimization engineering, not winning a fight with the exporter. Either way, document which parts run where, and benchmark the actual architecture you ship (don't claim a full-pipeline ONNX benchmark if only the backbone was exported).
- Required benchmark, reported in RESULTS.md with real numbers once implemented:
  - Output equivalence: PyTorch vs ONNX Runtime predictions match within a defined numerical tolerance.
  - Model size (PyTorch checkpoint vs ONNX file).
  - Cold-start time.
  - Mean / p50 / p95 inference latency.
  - Throughput (images/sec).
  - CPU inference confirmed (this is the target environment — no paid GPU).
  - Memory consumption, if measurable without adding heavyweight profiling tooling.
- Target write-up shape for the README: `PyTorch: XX ms/image` / `ONNX Runtime CPU: YY ms/image` / `Speedup: Z×` — concrete numbers, not a claim without them.

### 4.5 Go/No-Go decision fusion — "ISO-informed dimensional validation," not ISO certification
- Combine (a) anomaly score above/below threshold and (b) measured diameter within a starter ISO 965 tolerance table for the nominal thread size.
- Ship a small starter lookup table of ISO 965 major/pitch diameter limits (max/min, mm) for common metric sizes M1–M12, sourced from public ISO 965 tolerance data — a starting reference table to extend, not exhaustive coverage (ISO 965 defines multiple tolerance classes/qualities; the starter table should assume one representative class and say so explicitly rather than silently pick one).
- **Naming/framing matters here.** Call this "ISO-informed dimensional validation" in code and docs, not "ISO 965 compliance verification" — measuring major diameter and pitch from an image is not a complete ISO 965 conformity check (pitch diameter and full thread-profile geometry are also part of the standard and aren't covered here). Put this sentence in the README near the tolerance table: *"The tolerance table is used to demonstrate standards-aware decision logic and does not constitute certified dimensional inspection or complete ISO 965 compliance verification."*
- Final verdict logic: **No-Go** if either check fails, with the specific failing check surfaced in the output (explainable Go/No-Go), e.g.:
```json
{
  "verdict": "NO_GO",
  "anomaly_score": 0.87,
  "anomaly_threshold": 0.61,
  "measurements": { "major_diameter_mm": 7.92, "pitch_mm": 1.25 },
  "failed_checks": ["anomaly_score"],
  "inference_ms": 31.7
}
```

### 4.6 Serving layer — FastAPI + Docker (core, not stretch)
- Architecture: `Gradio UI → FastAPI → Inspection Pipeline`. Gradio is the interface only — it calls the API, it does not call the pipeline directly.
- Endpoints: `POST /inspect/image`, `POST /inspect/video` (phase 3), `GET /health`, `GET /model/info`.
- Response schema: as shown in §4.5 above.
- `Dockerfile` + `docker-compose.yml` for the API service. This is the concrete Python/FastAPI/Docker/REST-API/model-serving evidence for the portfolio.

### 4.7 Video input (phase 3, core but not phase-1-blocking; file-based, not streaming)
The posting mentions video processing / real-time CV and borescope-style acquisition as relevant experience — worth covering, without inventing or implying a full RTSP/industrial streaming system.
- Pipeline: `Video file → OpenCV frame extraction (uses FFmpeg under the hood, no separate GStreamer pipeline needed) → frame sampling → inspection pipeline (4.1–4.5) → temporal aggregation → Go/No-Go`.
- Output: an annotated video with anomaly heatmap, measurements, FPS, verdict, and failing-check reason overlaid per sampled frame.
- Served via `POST /inspect/video` on the same FastAPI app.
- Naming: call this "video input" / "video processing pipeline" everywhere (code, README, endpoint docs) — not "streaming," which this isn't. True streaming ingestion is a stretch goal (§9).

### 4.8 C++ edge-inference demo (phase 6, small and scoped)
Not a port of the project — a minimal proof that inference works outside the Python runtime, directly addressing the posting's C/C++ nice-to-have.
```
edge/
  cpp/
    main.cpp
    inference.cpp
    inference.hpp
    CMakeLists.txt
```
Loads the ONNX model via the ONNX Runtime C++ API, reads an image via OpenCV, preprocesses, runs inference, prints score + latency. README line: *"A minimal C++/ONNX Runtime edge inference implementation is included to demonstrate deployment outside the Python runtime."*

## 5. Tech stack (free-tier only, per standing constraint)

- Python 3.11, OpenCV, NumPy, scikit-image
- PyTorch + torchvision (pretrained ResNet18/WideResNet50 backbones — free public weights)
- ONNX + ONNX Runtime (CPU)
- FastAPI + Uvicorn; Docker + docker-compose
- Gradio (free) as the UI layer calling the API
- C++17 + CMake + ONNX Runtime C++ API + OpenCV (C++ bindings) for the edge demo
- GitHub Actions (free tier for public repos) for basic CI — lint + tests on push
- Google Colab free tier as a fallback if local CPU is too slow for backbone feature extraction at scale; local-first otherwise
- GitHub for hosting; Hugging Face Spaces (free tier) as an optional live-demo link
- No paid APIs, no paid compute, no licensed datasets requiring purchase

## 6. Repo structure (proposed — Claude Code may adjust)

```
gaugevision/
  data/
    mvtec_loader.py        # download/verify/load MVTec AD screw split
    synthetic_thread.py    # procedural thread image + ground-truth generator (phase 4)
  calibration/
    lens_calibration.py    # 4.1a: chessboard calibration, undistortion
    metric_calibration.py  # 4.1b: MetricCalibration class (from_reference/pixels_to_mm/mm_to_pixels)
  measurement/
    types.py                 # MeasurementResult contract (phase 1, stable from commit one)
    segment.py                # ROI / silhouette segmentation
    axis.py                    # longitudinal axis estimation + derotation
    profile.py                  # thread-region isolation + profile extraction -> 1D signal
    diameter.py                  # major diameter estimation
    pitch.py                      # PitchEstimator interface: PeakPitchEstimator, FFTPitchEstimator
    validate.py                    # error metrics + estimator comparison vs synthetic ground truth (phase 4)
  anomaly/
    padim.py                 # PaDiM implementation/wrapper (phase 1)
    patchcore.py              # stretch (phase 4)
    evaluate.py                # AUROC vs MVTec labels, metric/protocol explicit
  deployment/
    export_onnx.py            # PyTorch -> ONNX export
    onnx_infer.py              # ONNX Runtime inference wrapper
    benchmark.py                # latency/throughput/size/cold-start benchmark -> RESULTS.md
  decision/
    iso965_table.py            # tolerance lookup table + logic
    fuse.py                      # combine anomaly + measurement -> verdict
  video/
    frame_extract.py             # phase 3: OpenCV-based frame sampling
    annotate.py                   # phase 3: overlay heatmap/measurements/verdict on frames
  api/
    main.py                        # FastAPI app: /inspect/image, /inspect/video, /health, /model/info
    Dockerfile
    docker-compose.yml
  app/
    demo.py                         # Gradio UI, calls the API
  edge/
    cpp/                             # phase 6: minimal C++/ONNX Runtime demo
      main.cpp
      inference.cpp
      inference.hpp
      CMakeLists.txt
  .github/
    workflows/
      ci.yml                          # phase 5: lint + tests
  notebooks/                          # EDA, experiment scratch work
  tests/
  docs/
    README.md                         # setup, license/attribution, ISO disclaimer, results
    RESULTS.md                        # benchmark numbers, error tables, screenshots
```

## 7. Build phases — final architecture early, depth in layers

**Phase 1 — working vertical slice, built on the final architecture (priority: get this on GitHub fast).**
Minimal data loader (screw category) → the full measurement pipeline from §4.2 (`MeasurementResult`, `MetricCalibration` with a demonstration reference source, **one** working `PitchEstimator` implementation — the second follows in phase 4) → PaDiM anomaly detection → decision fusion (§4.5, basic thresholds) → FastAPI (`/inspect/image`, `/health`) → Dockerfile → minimal Gradio UI hitting the API. Goal: a functional, demoable, containerized pipeline built on interfaces that won't need to be torn up later. Synthetic-ground-truth *validation* and PatchCore are still phase-4 — but the interfaces they plug into already exist from Phase 1.

**Phase 2 — ONNX export + optimized inference benchmark.**
Export PaDiM backbone to ONNX, wire up ONNX Runtime, run the full benchmark checklist from §4.4, write real numbers into RESULTS.md.

**Phase 3 — video input.**
Frame extraction, temporal aggregation, annotated output video, `/inspect/video` endpoint.

**Phase 4 — depth pass.**
Validated metric/scale calibration source (§4.1b) replacing the phase-1 demonstration reference; synthetic thread generator + measurement validation with quantified error, including the `PeakPitchEstimator` vs `FFTPitchEstimator` comparison table (§4.2); ISO-informed tolerance table + disclaimer language finalized; PatchCore added and benchmarked against PaDiM if time allows.

**Phase 5 — polish (prioritize this over the C++ demo if time runs short).**
GitHub Actions CI (lint + tests), basic structured logging (Python `logging` with a JSON formatter — no heavyweight observability stack needed), final README/RESULTS write-up, demo recording. Rationale: if an interview happens before everything is done, finding a tested, documented, CI-passing repo without the C++ demo is a better outcome than the reverse.

**Phase 6 — C++ edge-inference demo.**
Minimal ONNX Runtime C++ + OpenCV demo per §4.8.

## 8. Evaluation & validation (be honest about numbers)
- Report actual AUROC achieved, using the metric/protocol confirmed against the source table (see §4.3) — never publish a plausible-sounding number that wasn't checked against the actual paper/table for that exact method and metric.
- Report actual measurement error (mm and %) on synthetic ground truth (phase 4), across at least a small sweep of thread sizes and clean/blur/rotated conditions — including the `PeakPitchEstimator` vs `FFTPitchEstimator` comparison table from §4.2, with the default estimator chosen from that evidence.
- Report actual ONNX vs PyTorch benchmark numbers (phase 2) — this is likely the single most interview-relevant number in the whole project given the posting's emphasis on inference optimization.
- If any module underperforms, document why (screw's rotation invariance is a known weakness for patch-based anomaly methods; thin/low-contrast thread edges are a known weakness for classical pitch estimation) — an honest limitations section is worth more in an interview than a hidden failure mode.

## 9. Stretch goals (only after phases 1–6 are solid)
- TensorRT / real Jetson hardware benchmarking, if hardware becomes available.
- Kubernetes deployment manifest, MLflow experiment tracking.
- Real-time streaming ingestion (RTSP/GStreamer) beyond the phase-3 file-based video input.
- Extend the ISO 965 tolerance table to more thread sizes/tolerance classes.
- Add a second MVTec category (e.g. "metal_nut") to show the pipeline generalizes beyond screws specifically.

## 10. Constraints & conventions
- Free tools only — flag anything that would require a paid tier before adding it.
- Cite MVTec AD's CC BY-NC-SA 4.0 license prominently in the README; no commercial framing anywhere in the repo.
- Keep the "synthetic ground truth is for validating the measurement math, not a substitute dataset" distinction clear in the README.
- Keep the "ISO-informed, not ISO-certified" distinction clear near the tolerance table (§4.5).
- Keep the "video input, not streaming" distinction clear in the README and API docs (§4.7).
- No measurement logic should depend on hard-coded px/mm constants outside the `MetricCalibration` component (§4.1b) — route every conversion through it.
- Never publish a benchmark number (AUROC, latency, etc.) that hasn't been produced by the actual implemented code or checked directly against its cited source.
- Prefer honest, benchmarked numbers over polished claims throughout docs.

## 11. Open questions for Claude Code to flag back to Elton if they come up
- Whether to pursue PatchCore (phase 4) at all if PaDiM already benchmarks reasonably and time is tight.
- Whether a second public dataset with real dimensional ground truth is worth sourcing instead of the synthetic generator, if one turns up during implementation.
- Whether the Hugging Face Spaces demo is worth the setup time vs. a local-run demo + screen recording for the portfolio.
- Whether the C++ edge demo (phase 6) is worth expanding beyond a minimal single-image scorer if time allows.
- Whether axis estimation/derotation (§4.2 step 1–2) turns out harder than expected on real MVTec screw images (rotation-in-plane is common in that category) — if so, flag back rather than silently spending phase-1 time on robustness that belongs in phase 4.
