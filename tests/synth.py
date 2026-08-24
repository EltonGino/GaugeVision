"""Shared synthetic-image test helpers (not a test module itself — no
``test_`` prefix, so pytest won't collect it)."""

from __future__ import annotations

import cv2
import numpy as np


def make_synthetic_screw(
    shank_width: int = 60,
    shank_length: int = 260,
    thread_amplitude: int = 8,
    period_px: int = 20,
    head_width: int = 100,
    head_length: int = 50,
    size: tuple[int, int] = (400, 200),
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Procedurally draws a simple screw-like silhouette: a wider head plus a
    shank whose width oscillates periodically to simulate thread crests."""
    h, w = size
    canvas = np.zeros((h, w), dtype=np.uint8)
    cx = w // 2

    # Head (top)
    top = 10
    cv2.rectangle(
        canvas,
        (cx - head_width // 2, top),
        (cx + head_width // 2, top + head_length),
        255,
        thickness=-1,
    )

    # Threaded shank (below head): width oscillates with period_px
    shank_top = top + head_length
    for row in range(shank_length):
        y = shank_top + row
        if y >= h:
            break
        local_width = shank_width + thread_amplitude * np.sin(2 * np.pi * row / period_px)
        half = int(local_width / 2)
        cv2.line(canvas, (cx - half, y), (cx + half, y), 255, 1)

    if angle_deg != 0.0:
        rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle_deg, 1.0)
        canvas = cv2.warpAffine(canvas, rot_mat, (w, h))

    return canvas
