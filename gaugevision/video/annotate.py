"""Per-frame annotation overlay — CLAUDE.md §4.7 (Phase 3).

Overlays the anomaly heatmap, measurements, FPS, verdict, and failing-check
reason onto a single BGR frame, matching the fields CLAUDE.md §4.7 asks for
in the output video.
"""

from __future__ import annotations

import cv2
import numpy as np

from gaugevision.anomaly.types import AnomalyResult
from gaugevision.decision.fuse import InspectionVerdict
from gaugevision.measurement.types import MeasurementResult

_GO_COLOR = (0, 200, 0)  # BGR
_NO_GO_COLOR = (0, 0, 220)
_TEXT_COLOR = (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _blend_heatmap(frame_bgr: np.ndarray, heatmap: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    norm = heatmap - heatmap.min()
    max_val = norm.max()
    if max_val > 0:
        norm = norm / max_val
    norm_u8 = (norm * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame_bgr, 1 - alpha, color, alpha, 0)


def annotate_frame(
    frame_bgr: np.ndarray,
    anomaly: AnomalyResult,
    measurement: MeasurementResult | None,
    verdict: InspectionVerdict,
    display_fps: float,
) -> np.ndarray:
    """Return a copy of ``frame_bgr`` with the heatmap blended in and text
    overlays for verdict, measurements, and FPS. ``measurement`` may be None
    if the measurement pipeline failed on this frame — annotated as "n/a"
    rather than a fabricated value."""
    annotated = _blend_heatmap(frame_bgr, anomaly.heatmap)

    diameter_mm = measurement.major_diameter_mm if measurement else None
    pitch_mm = measurement.pitch_mm if measurement else None

    verdict_color = _GO_COLOR if verdict.verdict == "GO" else _NO_GO_COLOR
    lines = [
        (f"{verdict.verdict}", verdict_color, 0.9, 2),
        (f"anomaly {anomaly.score:.2f} / {anomaly.threshold:.2f}", _TEXT_COLOR, 0.55, 1),
        (
            "diameter " + (f"{diameter_mm:.2f}mm" if diameter_mm is not None else "n/a"),
            _TEXT_COLOR,
            0.55,
            1,
        ),
        (
            "pitch " + (f"{pitch_mm:.2f}mm" if pitch_mm else "n/a"),
            _TEXT_COLOR,
            0.55,
            1,
        ),
        (f"{display_fps:.1f} fps", _TEXT_COLOR, 0.55, 1),
    ]
    if verdict.failed_checks:
        lines.append((f"failed: {', '.join(verdict.failed_checks)}", _NO_GO_COLOR, 0.55, 1))

    y = 28
    for text, color, scale, thickness in lines:
        cv2.putText(
            annotated, text, (10, y), _FONT, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA
        )
        cv2.putText(annotated, text, (10, y), _FONT, scale, color, thickness, cv2.LINE_AA)
        y += int(28 * max(scale, 0.55))

    return annotated
