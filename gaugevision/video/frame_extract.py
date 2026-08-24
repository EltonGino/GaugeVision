"""OpenCV-based frame sampling — CLAUDE.md §4.7 (Phase 3).

Video input, not streaming: this reads a video *file* end-to-end via
``cv2.VideoCapture`` (which uses FFmpeg under the hood — no separate
GStreamer pipeline needed) and samples frames at a configurable interval.
There is no live/RTSP ingestion here; see CLAUDE.md §9 for that stretch goal.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    fps: float
    frame_count: int
    width: int
    height: int


@dataclass(frozen=True)
class SampledFrame:
    frame: np.ndarray  # BGR, as read by cv2.VideoCapture
    frame_index: int  # index in the source video (0-based)
    timestamp_sec: float


def probe_video(video_path: str) -> VideoInfo:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video file: {video_path}")
    try:
        return VideoInfo(
            fps=cap.get(cv2.CAP_PROP_FPS) or 0.0,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def default_sample_interval(fps: float, target_samples_per_sec: float = 2.0) -> int:
    """Pick a frame-sampling interval that yields ~target_samples_per_sec
    inspected frames — CPU PaDiM inference is tens of ms/frame (see
    docs/RESULTS.md), so inspecting every frame of a real-time-shot video is
    unnecessary and slow; a couple of samples per second is enough to catch
    a defect without processing every frame."""
    if fps <= 0:
        return 10
    return max(1, round(fps / target_samples_per_sec))


def extract_frames(
    video_path: str, sample_every_n_frames: int | None = None
) -> tuple[VideoInfo, list[SampledFrame], int]:
    """Sample every Nth frame from a video file.

    Args:
        video_path: path to a video file readable by OpenCV/FFmpeg.
        sample_every_n_frames: sampling interval; if None, derived from the
            source FPS via ``default_sample_interval``.

    Returns:
        (VideoInfo, sampled frames in playback order, interval actually used).
    """
    info = probe_video(video_path)
    interval = sample_every_n_frames or default_sample_interval(info.fps)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open video file: {video_path}")

    sampled: list[SampledFrame] = []
    try:
        frame_index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_index % interval == 0:
                timestamp_sec = frame_index / info.fps if info.fps > 0 else 0.0
                sampled.append(
                    SampledFrame(
                        frame=frame, frame_index=frame_index, timestamp_sec=timestamp_sec
                    )
                )
            frame_index += 1
    finally:
        cap.release()

    if not sampled:
        raise RuntimeError(f"no frames could be read from video file: {video_path}")

    return info, sampled, interval
