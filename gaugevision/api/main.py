"""FastAPI serving layer — CLAUDE.md §4.6.

Architecture: Gradio UI -> FastAPI -> Inspection Pipeline. This module is the
only place that talks to the pipeline directly; the Gradio app (``app/demo.py``)
must call these HTTP endpoints, never the pipeline modules.
"""

from __future__ import annotations

import base64
import logging
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from gaugevision import config
from gaugevision.anomaly.padim import PaDiM
from gaugevision.api.schemas import (
    FrameVerdict,
    HealthResponse,
    InspectImageResponse,
    InspectVideoResponse,
    ModelInfoResponse,
)
from gaugevision.calibration.metric_calibration import MetricCalibration
from gaugevision.decision.fuse import fuse_verdict
from gaugevision.logging_config import configure_logging
from gaugevision.measurement.pipeline import run_measurement_pipeline
from gaugevision.video.pipeline import run_video_inspection

configure_logging()
logger = logging.getLogger(__name__)

_state: dict = {"model": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading PaDiM model from %s", config.ANOMALY_MODEL_PATH)
    try:
        _state["model"] = PaDiM.load(str(config.ANOMALY_MODEL_PATH))
        logger.info("Model loaded. Threshold=%s", getattr(_state["model"], "threshold_", None))
    except FileNotFoundError:
        logger.warning(
            "No trained model found at %s — /inspect/image will 503 until "
            "`python -m gaugevision.anomaly.train` is run.",
            config.ANOMALY_MODEL_PATH,
        )
        _state["model"] = None
    yield
    _state.clear()


app = FastAPI(title="GaugeVision API", version="0.1.0", lifespan=lifespan)


def _get_model() -> PaDiM:
    model = _state.get("model")
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Anomaly model not loaded. Run "
                "`python -m gaugevision.anomaly.train` first."
            ),
        )
    return model


def _decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not decode image file")
    return image


def _encode_heatmap_png_base64(image: np.ndarray, heatmap: np.ndarray) -> str:
    """Overlay a JET-colormapped heatmap on the grayscale image, PNG+base64."""
    norm = heatmap - heatmap.min()
    max_val = norm.max()
    if max_val > 0:
        norm = norm / max_val
    norm_u8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)

    base_bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base_bgr, 0.5, color, 0.5, 0)

    ok, buf = cv2.imencode(".png", overlay)
    if not ok:
        raise RuntimeError("failed to encode heatmap overlay")
    return base64.b64encode(buf.tobytes()).decode("ascii")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", model_loaded=_state.get("model") is not None)


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    model = _state.get("model")
    calibration = MetricCalibration.demo_reference(config.DEMO_CALIBRATION_PX_PER_MM)
    return ModelInfoResponse(
        anomaly_model="PaDiM",
        anomaly_backbone=config.ANOMALY_BACKBONE,
        anomaly_n_features=config.ANOMALY_N_FEATURES,
        anomaly_threshold=getattr(model, "threshold_", None) if model else None,
        pitch_estimator="PeakPitchEstimator",
        calibration_source=calibration.source,
        calibration_validated=calibration.validated,
        phase="Phase 3 — video input",
    )


@app.post("/inspect/image", response_model=InspectImageResponse)
async def inspect_image(file: UploadFile = File(...)) -> InspectImageResponse:  # noqa: B008
    model = _get_model()
    data = await file.read()
    image = _decode_upload(data)

    start = time.perf_counter()

    calibration = MetricCalibration.demo_reference(config.DEMO_CALIBRATION_PX_PER_MM)
    try:
        measurement = run_measurement_pipeline(
            image,
            calibration=calibration,
            head_margin_fraction=config.THREAD_HEAD_MARGIN_FRACTION,
        )
    except RuntimeError as e:
        raise HTTPException(status_code=422, detail=f"measurement pipeline failed: {e}")

    anomaly = model.predict(image)

    inference_ms = (time.perf_counter() - start) * 1000.0

    verdict = fuse_verdict(anomaly=anomaly, measurement=measurement, inference_ms=inference_ms)
    heatmap_b64 = _encode_heatmap_png_base64(image, anomaly.heatmap)

    logger.info(
        "image inspection complete",
        extra={
            "verdict": verdict.verdict,
            "failed_checks": verdict.failed_checks,
            "anomaly_score": verdict.anomaly_score,
            "inference_ms": inference_ms,
        },
    )

    return InspectImageResponse(verdict=verdict, anomaly_heatmap_png_base64=heatmap_b64)


@app.post("/inspect/video", response_model=InspectVideoResponse)
async def inspect_video(file: UploadFile = File(...)) -> InspectVideoResponse:  # noqa: B008
    """Video input (file-based), not streaming — CLAUDE.md §4.7.

    Samples frames from the uploaded video file, runs the same per-frame
    measurement/anomaly/decision pipeline as ``/inspect/image``, and returns
    a temporally-aggregated verdict plus an annotated output video
    (heatmap, measurements, FPS, verdict, failing-check reason per sampled
    frame) as base64-encoded MP4.
    """
    model = _get_model()
    data = await file.read()

    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / f"input{suffix}"
        output_path = Path(tmpdir) / "annotated.mp4"
        input_path.write_bytes(data)

        calibration = MetricCalibration.demo_reference(config.DEMO_CALIBRATION_PX_PER_MM)
        try:
            result = run_video_inspection(
                str(input_path),
                model=model,
                calibration=calibration,
                output_path=str(output_path),
                head_margin_fraction=config.THREAD_HEAD_MARGIN_FRACTION,
            )
        except RuntimeError as e:
            raise HTTPException(status_code=422, detail=f"video inspection failed: {e}")

        video_b64 = base64.b64encode(output_path.read_bytes()).decode("ascii")

    logger.info(
        "video inspection complete",
        extra={
            "overall_verdict": result.overall_verdict,
            "overall_failed_checks": result.overall_failed_checks,
            "n_frames_sampled": result.n_frames_sampled,
            "inspection_throughput_fps": result.inspection_throughput_fps,
        },
    )

    return InspectVideoResponse(
        overall_verdict=result.overall_verdict,
        overall_failed_checks=result.overall_failed_checks,
        frame_results=[
            FrameVerdict(
                frame_index=fr.frame_index,
                timestamp_sec=fr.timestamp_sec,
                verdict=fr.verdict,
            )
            for fr in result.frame_results
        ],
        n_frames_sampled=result.n_frames_sampled,
        source_fps=result.source_fps,
        sample_interval_frames=result.sample_interval_frames,
        inspection_throughput_fps=result.inspection_throughput_fps,
        notes=result.notes,
        annotated_video_base64=video_b64,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled error processing %s", request.url)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})
