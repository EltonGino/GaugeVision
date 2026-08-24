"""FastAPI serving layer — CLAUDE.md §4.6.

Architecture: Gradio UI -> FastAPI -> Inspection Pipeline. This module is the
only place that talks to the pipeline directly; the Gradio app (``app/demo.py``)
must call these HTTP endpoints, never the pipeline modules.
"""

from __future__ import annotations

import base64
import logging
import time
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from gaugevision import config
from gaugevision.anomaly.padim import PaDiM
from gaugevision.api.schemas import (
    HealthResponse,
    InspectImageResponse,
    ModelInfoResponse,
)
from gaugevision.calibration.metric_calibration import MetricCalibration
from gaugevision.decision.fuse import fuse_verdict
from gaugevision.measurement.pipeline import run_measurement_pipeline

logging.basicConfig(level=logging.INFO)
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
        phase="Phase 1 — vertical slice",
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

    return InspectImageResponse(verdict=verdict, anomaly_heatmap_png_base64=heatmap_b64)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled error processing %s", request.url)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})
