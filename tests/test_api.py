"""API contract tests using FastAPI's TestClient.

These exercise routing, request/response schemas, and error handling without
requiring a trained model on disk — the unloaded-model path (503) is tested
explicitly since CI won't have `models/padim_screw.npz` available.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from gaugevision.api.main import app


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_model_info_endpoint(client):
    response = client.get("/model/info")
    assert response.status_code == 200
    body = response.json()
    assert body["anomaly_model"] == "PaDiM"
    assert body["calibration_source"] == "demo_reference"
    assert body["calibration_validated"] is False


def test_inspect_image_without_model_returns_503_or_succeeds(client, tmp_path):
    """If no model is loaded (e.g. in CI without models/padim_screw.npz),
    the endpoint should fail gracefully with 503, not crash."""
    image = np.zeros((100, 100), dtype=np.uint8)
    cv2.circle(image, (50, 50), 30, 255, -1)
    ok, buf = cv2.imencode(".png", image)
    assert ok

    response = client.post(
        "/inspect/image", files={"file": ("test.png", buf.tobytes(), "image/png")}
    )
    assert response.status_code in (503, 200, 422)


def test_inspect_image_rejects_invalid_file(client):
    response = client.post(
        "/inspect/image", files={"file": ("bad.png", b"not an image", "image/png")}
    )
    assert response.status_code in (400, 503)


def _make_small_video_bytes(n_frames: int = 6, size: tuple[int, int] = (60, 60)) -> bytes:
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(path), fourcc, 5.0, size)
        for _ in range(n_frames):
            frame = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            cv2.circle(frame, (size[0] // 2, size[1] // 2), 20, (255, 255, 255), -1)
            writer.write(frame)
        writer.release()
        return path.read_bytes()


def test_inspect_video_without_model_returns_503_or_succeeds(client):
    """Mirrors the image endpoint's unloaded-model contract test — 503 when
    no model is loaded (CI), otherwise a valid response or a 422 if the
    measurement pipeline can't process this synthetic clip's frames."""
    video_bytes = _make_small_video_bytes()
    response = client.post(
        "/inspect/video", files={"file": ("test.mp4", video_bytes, "video/mp4")}
    )
    assert response.status_code in (503, 200, 422)


def test_inspect_video_rejects_invalid_file(client):
    response = client.post(
        "/inspect/video", files={"file": ("bad.mp4", b"not a video", "video/mp4")}
    )
    assert response.status_code in (422, 503)
