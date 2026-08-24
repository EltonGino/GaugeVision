"""ROI / silhouette segmentation — first stage of the measurement pipeline
(CLAUDE.md §4.2).

MVTec "screw" images are grayscale, top-lit, with a controlled dark or light
background, so a global Otsu threshold plus morphological cleanup gives a
clean silhouette. This is intentionally the simplest correct approach for
Phase 1 (see CLAUDE.md's "correct architecture + one defensible working
baseline now" guardrail).
"""

from __future__ import annotations

import cv2
import numpy as np

from gaugevision.measurement.types import SegmentationResult


def segment_screw(image: np.ndarray, morph_kernel_size: int = 5) -> SegmentationResult:
    """Threshold + morphological cleanup + largest-contour selection.

    Args:
        image: grayscale or BGR image containing a single screw on a
            roughly uniform background.
        morph_kernel_size: size of the elliptical kernel used for
            open/close morphological cleanup.

    Returns:
        SegmentationResult with a binary mask (0/255) isolating the screw
        and its axis-aligned bounding box.

    Raises:
        RuntimeError: if no foreground contour is found.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # The screw is assumed to be the minority class in pixel count relative to
    # a large uniform background; if thresholding inverted that assumption,
    # flip so the foreground (screw) is always the 255 region.
    if np.count_nonzero(thresh) > thresh.size // 2:
        thresh = cv2.bitwise_not(thresh)

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(
        cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        raise RuntimeError("segment_screw: no foreground contour found")

    largest = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(cleaned)
    cv2.drawContours(mask, [largest], -1, 255, thickness=cv2.FILLED)
    bbox = cv2.boundingRect(largest)

    return SegmentationResult(mask=mask, bbox=bbox)
