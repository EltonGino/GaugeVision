"""Synthetic thread-image generator — CLAUDE.md §3.2, §4.2 (Phase 4).

MVTec AD has no dimensional ground truth (no known real-world mm scale, no
annotated pitch/diameter — CLAUDE.md §3.2), so the measurement module needs a
source of *known* dimensions to validate against. This generator procedurally
draws a screw-like thread silhouette at a specified real-world size (mm),
plus a reference bar of known physical length rendered into the same scene —
mirroring how a real calibration photo would include a reference object.

This is a calibration/unit-test harness for the measurement math, not a
substitute dataset (CLAUDE.md §3.2) — real MVTec screw images are still used
for the qualitative/demo pass of the measurement module (see the API/Gradio
UI running against real MVTec images).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SyntheticThreadSpec:
    """Ground-truth physical dimensions used to render one sample."""

    major_diameter_mm: float
    minor_diameter_mm: float
    pitch_mm: float
    shank_length_mm: float = 20.0
    head_diameter_mm: float = 12.0
    head_length_mm: float = 4.0


@dataclass(frozen=True)
class SyntheticThreadSample:
    image: np.ndarray  # grayscale
    spec: SyntheticThreadSpec
    px_per_mm: float  # ground-truth rendering scale
    reference_bar_length_px: float
    reference_bar_length_mm: float
    condition: str  # "clean" | "blur" | "rotated"


def generate_synthetic_thread(
    spec: SyntheticThreadSpec,
    px_per_mm: float = 20.0,
    condition: str = "clean",
    rotation_deg: float = 20.0,
    blur_kernel: int = 7,
    reference_bar_length_mm: float = 10.0,
    canvas_size: tuple[int, int] = (500, 700),
) -> SyntheticThreadSample:
    """Render one synthetic thread sample with known ground truth.

    Args:
        spec: physical dimensions (mm) of the screw to render.
        px_per_mm: ground-truth scale used to render the scene — this is
            what a validated ``MetricCalibration`` should recover via the
            rendered reference bar.
        condition: "clean" (no perturbation), "blur" (Gaussian blur), or
            "rotated" (in-plane rotation of the whole scene).
        rotation_deg: rotation applied for condition="rotated".
        blur_kernel: Gaussian kernel size for condition="blur".
        reference_bar_length_mm: physical length of the rendered reference
            bar — the "known reference object" a real calibration photo
            would include.
        canvas_size: (height, width) of the output image.

    Returns:
        SyntheticThreadSample with the rendered image and full ground truth.
    """
    if condition not in ("clean", "blur", "rotated"):
        raise ValueError(f"unknown condition: {condition!r}")

    h, w = canvas_size
    canvas = np.zeros((h, w), dtype=np.uint8)

    major_px = spec.major_diameter_mm * px_per_mm
    minor_px = spec.minor_diameter_mm * px_per_mm
    pitch_px = spec.pitch_mm * px_per_mm
    shank_length_px = spec.shank_length_mm * px_per_mm
    head_width_px = spec.head_diameter_mm * px_per_mm
    head_length_px = spec.head_length_mm * px_per_mm
    reference_bar_length_px = reference_bar_length_mm * px_per_mm

    # Screw confined to the left half of the canvas, reference bar drawn
    # entirely in the right half — a fixed, generous column gap (not just a
    # few background pixels) guarantees the two never touch/merge into one
    # contour during segmentation, regardless of thread amplitude or blur.
    screw_column_width = w * 0.5
    cx = screw_column_width / 2
    max_half_width = max(head_width_px, major_px) / 2
    if cx + max_half_width >= screw_column_width:
        raise ValueError(
            f"canvas_size {canvas_size} too narrow for spec {spec} at "
            f"px_per_mm={px_per_mm}: increase canvas width or reduce diameters"
        )

    base_width = (major_px + minor_px) / 2.0
    amplitude = (major_px - minor_px) / 2.0

    top = int(0.08 * h)
    cv2.rectangle(
        canvas,
        (int(cx - head_width_px / 2), top),
        (int(cx + head_width_px / 2), int(top + head_length_px)),
        255,
        thickness=-1,
    )

    shank_top = top + head_length_px
    n_rows = int(shank_length_px)
    for row in range(n_rows):
        y = int(shank_top + row)
        if y >= h:
            break
        local_width = base_width + amplitude * np.sin(2 * np.pi * row / pitch_px)
        half = local_width / 2.0
        cv2.line(canvas, (int(cx - half), y), (int(cx + half), y), 255, 1)

    bar_x0 = int(screw_column_width + 0.1 * w)
    bar_x1 = int(bar_x0 + reference_bar_length_px)
    if bar_x1 >= w:
        raise ValueError(
            f"canvas_size {canvas_size} too narrow to fit a "
            f"{reference_bar_length_mm}mm reference bar at px_per_mm={px_per_mm}"
        )
    bar_y = int(h / 2)
    cv2.line(canvas, (bar_x0, bar_y), (bar_x1, bar_y), 255, thickness=3)

    if condition == "rotated":
        rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), rotation_deg, 1.0)
        canvas = cv2.warpAffine(canvas, rot_mat, (w, h))
    elif condition == "blur":
        canvas = cv2.GaussianBlur(canvas, (blur_kernel, blur_kernel), 0)

    return SyntheticThreadSample(
        image=canvas,
        spec=spec,
        px_per_mm=px_per_mm,
        reference_bar_length_px=reference_bar_length_px,
        reference_bar_length_mm=reference_bar_length_mm,
        condition=condition,
    )
