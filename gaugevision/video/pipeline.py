"""Video inspection pipeline — CLAUDE.md §4.7 (Phase 3).

`Video file → OpenCV frame extraction → frame sampling → inspection
pipeline (§4.1–4.5) → temporal aggregation → Go/No-Go`.

Reuses the exact same per-frame measurement/anomaly/decision functions the
``/inspect/image`` endpoint calls (``run_measurement_pipeline``, `PaDiM`.
predict``, ``fuse_verdict``) — a video inspection is not a separate
algorithm, it's the image pipeline run per sampled frame plus temporal
aggregation across frames. This keeps the single-image and video code paths
from silently diverging.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2

from gaugevision.anomaly.padim import PaDiM
from gaugevision.calibration.metric_calibration import MetricCalibration
from gaugevision.decision.fuse import InspectionVerdict, fuse_verdict
from gaugevision.measurement.pipeline import run_measurement_pipeline
from gaugevision.video.annotate import annotate_frame
from gaugevision.video.frame_extract import extract_frames

logger = logging.getLogger(__name__)

VIDEO_FOURCC = "mp4v"


@dataclass(frozen=True)
class FrameInspection:
    frame_index: int
    timestamp_sec: float
    verdict: InspectionVerdict


@dataclass(frozen=True)
class VideoInspectionResult:
    overall_verdict: str  # "GO" | "NO_GO"
    overall_failed_checks: list[str]
    frame_results: list[FrameInspection]
    n_frames_sampled: int
    source_fps: float
    sample_interval_frames: int
    inspection_throughput_fps: float  # measured: sampled frames processed / wall time
    output_video_path: str | None = None
    notes: list[str] = field(default_factory=list)


def _aggregate(frame_results: list[FrameInspection]) -> tuple[str, list[str]]:
    """Worst-case aggregation: NO_GO if any sampled frame failed a check —
    the conservative default for a QC pass over a video (a defect visible
    in even one sampled frame should not be averaged away)."""
    failed_checks: set[str] = set()
    for fr in frame_results:
        failed_checks.update(fr.verdict.failed_checks)
    overall = "NO_GO" if failed_checks else "GO"
    return overall, sorted(failed_checks)


def run_video_inspection(
    video_path: str,
    model: PaDiM,
    calibration: MetricCalibration,
    sample_every_n_frames: int | None = None,
    output_path: str | None = None,
    head_margin_fraction: float = 0.22,
) -> VideoInspectionResult:
    """Run the full inspection pipeline over sampled frames of a video file.

    Args:
        video_path: local path to a video file.
        model: fitted PaDiM anomaly detector.
        calibration: MetricCalibration for px->mm conversion.
        sample_every_n_frames: sampling interval; auto-derived from source
            FPS if omitted (see ``frame_extract.default_sample_interval``).
        output_path: if given, an annotated video is written here (heatmap,
            measurements, FPS, verdict, failing-check reason per sampled
            frame — CLAUDE.md §4.7). If omitted, only the JSON-serializable
            summary is returned.
        head_margin_fraction: forwarded to the measurement pipeline.
    """
    info, sampled, interval = extract_frames(video_path, sample_every_n_frames)
    playback_fps = info.fps / interval if info.fps > 0 else 2.0

    writer = None
    if output_path is not None:
        fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
        writer = cv2.VideoWriter(output_path, fourcc, playback_fps, (info.width, info.height))
        if not writer.isOpened():
            raise RuntimeError(f"could not open video writer for: {output_path}")

    frame_results: list[FrameInspection] = []
    notes: list[str] = []
    start = time.perf_counter()

    try:
        for sf in sampled:
            frame_start = time.perf_counter()
            gray = cv2.cvtColor(sf.frame, cv2.COLOR_BGR2GRAY)

            try:
                measurement = run_measurement_pipeline(
                    gray, calibration=calibration, head_margin_fraction=head_margin_fraction
                )
            except RuntimeError as e:
                notes.append(f"frame {sf.frame_index}: measurement failed ({e})")
                measurement = None

            anomaly = model.predict(gray)
            inference_ms = (time.perf_counter() - frame_start) * 1000.0
            verdict = fuse_verdict(
                anomaly=anomaly, measurement=measurement, inference_ms=inference_ms
            )
            frame_results.append(
                FrameInspection(
                    frame_index=sf.frame_index,
                    timestamp_sec=sf.timestamp_sec,
                    verdict=verdict,
                )
            )

            if writer is not None:
                annotated = annotate_frame(sf.frame, anomaly, measurement, verdict, playback_fps)
                writer.write(annotated)
    finally:
        if writer is not None:
            writer.release()

    elapsed = time.perf_counter() - start
    throughput = len(sampled) / elapsed if elapsed > 0 else float("inf")

    overall_verdict, overall_failed_checks = _aggregate(frame_results)

    logger.info(
        "Video inspection: %d frames sampled (interval=%d), overall=%s, "
        "throughput=%.1f frames/sec",
        len(sampled),
        interval,
        overall_verdict,
        throughput,
    )

    return VideoInspectionResult(
        overall_verdict=overall_verdict,
        overall_failed_checks=overall_failed_checks,
        frame_results=frame_results,
        n_frames_sampled=len(sampled),
        source_fps=info.fps,
        sample_interval_frames=interval,
        inspection_throughput_fps=throughput,
        output_video_path=output_path,
        notes=notes,
    )
