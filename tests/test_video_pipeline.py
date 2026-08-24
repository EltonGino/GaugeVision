"""Video-input pipeline tests (CLAUDE.md §4.7) against a synthesized video.

MVTec AD has no video assets, so — mirroring the synthetic-checkerboard
precedent already used for lens calibration (CLAUDE.md §4.1a) — these tests
build a short synthetic video from the same procedural screw silhouette used
in test_measurement_pipeline.py, as a capability demonstration/unit-test
harness rather than a claim about real footage.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from gaugevision.video.frame_extract import default_sample_interval, extract_frames
from tests.synth import make_synthetic_screw


def _write_synthetic_video(path: Path, n_frames: int = 20, fps: float = 10.0) -> None:
    size = (400, 200)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, fps, (size[1], size[0]))
    assert writer.isOpened()
    try:
        for i in range(n_frames):
            gray = make_synthetic_screw(size=size, angle_deg=float(i))
            bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            writer.write(bgr)
    finally:
        writer.release()


@pytest.fixture
def synthetic_video(tmp_path) -> Path:
    path = tmp_path / "synthetic_screw.mp4"
    _write_synthetic_video(path)
    return path


def test_default_sample_interval_scales_with_fps():
    assert default_sample_interval(30.0, target_samples_per_sec=2.0) == 15
    assert default_sample_interval(0.0) == 10  # fallback for unreadable fps


def test_extract_frames_samples_expected_count(synthetic_video):
    info, sampled, interval = extract_frames(str(synthetic_video), sample_every_n_frames=5)
    assert info.frame_count == 20
    assert interval == 5
    assert len(sampled) == 4  # frames 0, 5, 10, 15
    assert [f.frame_index for f in sampled] == [0, 5, 10, 15]


def test_extract_frames_raises_on_missing_file(tmp_path):
    with pytest.raises(RuntimeError):
        extract_frames(str(tmp_path / "does_not_exist.mp4"))


def test_run_video_inspection_end_to_end(synthetic_video, tmp_path, monkeypatch):
    """Full pipeline against a synthetic video, using a fitted-looking fake
    PaDiM model (no real training data required) to keep this test fast and
    network-free — anomaly detection itself is exercised separately in
    tests/test_padim_math.py; this test is about the video orchestration."""
    from gaugevision.anomaly.types import AnomalyResult
    from gaugevision.calibration.metric_calibration import MetricCalibration
    from gaugevision.video import pipeline as video_pipeline

    class _FakePaDiM:
        def predict(self, image, threshold=None):
            return AnomalyResult(
                score=0.1,
                heatmap=np.zeros(image.shape[:2], dtype=np.float32),
                threshold=0.5,
                is_anomalous=False,
            )

    output_path = tmp_path / "annotated.mp4"
    calibration = MetricCalibration.demo_reference()

    result = video_pipeline.run_video_inspection(
        str(synthetic_video),
        model=_FakePaDiM(),
        calibration=calibration,
        sample_every_n_frames=5,
        output_path=str(output_path),
    )

    assert result.n_frames_sampled == 4
    assert result.overall_verdict in ("GO", "NO_GO")
    assert len(result.frame_results) == 4
    assert result.source_fps == pytest.approx(10.0, abs=0.5)
    assert result.sample_interval_frames == 5
    assert result.inspection_throughput_fps > 0
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_run_video_inspection_without_output_path_skips_video_write(synthetic_video):
    from gaugevision.anomaly.types import AnomalyResult
    from gaugevision.calibration.metric_calibration import MetricCalibration
    from gaugevision.video import pipeline as video_pipeline

    class _FakePaDiM:
        def predict(self, image, threshold=None):
            return AnomalyResult(
                score=0.9,
                heatmap=np.zeros(image.shape[:2], dtype=np.float32),
                threshold=0.5,
                is_anomalous=True,
            )

    result = video_pipeline.run_video_inspection(
        str(synthetic_video),
        model=_FakePaDiM(),
        calibration=MetricCalibration.demo_reference(),
        sample_every_n_frames=5,
        output_path=None,
    )
    assert result.output_video_path is None
    assert result.overall_verdict == "NO_GO"
    assert "anomaly_score" in result.overall_failed_checks
