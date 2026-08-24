"""Pydantic request/response schemas for the FastAPI service (CLAUDE.md §4.6)."""

from __future__ import annotations

from pydantic import BaseModel

from gaugevision.decision.fuse import InspectionVerdict


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    anomaly_model: str
    anomaly_backbone: str
    anomaly_n_features: int
    anomaly_threshold: float | None
    pitch_estimator: str
    calibration_source: str
    calibration_validated: bool
    phase: str


class InspectImageResponse(BaseModel):
    verdict: InspectionVerdict
    anomaly_heatmap_png_base64: str


class FrameVerdict(BaseModel):
    frame_index: int
    timestamp_sec: float
    verdict: InspectionVerdict


class InspectVideoResponse(BaseModel):
    overall_verdict: str
    overall_failed_checks: list[str]
    frame_results: list[FrameVerdict]
    n_frames_sampled: int
    source_fps: float
    sample_interval_frames: int
    inspection_throughput_fps: float
    notes: list[str]
    annotated_video_base64: str
    annotated_video_content_type: str = "video/mp4"
