"""Longitudinal axis estimation and derotation — measurement pipeline steps 1-2
(CLAUDE.md §4.2).

Estimates the screw's principal (longest) axis via PCA over the silhouette's
foreground pixel coordinates, then derotates so that axis is vertical. This is
the simplest defensible baseline: PCA gives a stable in-plane orientation
estimate for an elongated silhouette without needing head/tip detection.
Robustness to occlusion/asymmetric heads is Phase-4 depth work — flagged as a
known limitation (CLAUDE.md §11).
"""

from __future__ import annotations

import cv2
import numpy as np

from gaugevision.measurement.types import AxisEstimate, SegmentationResult


def estimate_axis(segmentation: SegmentationResult) -> AxisEstimate:
    """PCA-based longitudinal axis estimate from a binary silhouette mask."""
    ys, xs = np.nonzero(segmentation.mask)
    if len(xs) < 10:
        raise RuntimeError("estimate_axis: too few foreground pixels to estimate axis")

    points = np.column_stack([xs, ys]).astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(points, mean=None)[:2]
    center_xy = (float(mean[0, 0]), float(mean[0, 1]))

    principal = eigenvectors[0]
    angle_deg = float(np.degrees(np.arctan2(principal[1], principal[0])))

    projected = (points - mean) @ principal
    length_px = float(projected.max() - projected.min())

    return AxisEstimate(center_xy=center_xy, angle_deg=angle_deg, length_px=length_px)


def derotate(
    image: np.ndarray, mask: np.ndarray, axis: AxisEstimate
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate image and mask so the estimated axis is vertical.

    Returns:
        (rotated_image, rotated_mask), both cropped-to-content is left to the
        caller (see ``profile.py``), this function only derotates.
    """
    h, w = mask.shape[:2]
    # Rotate by (angle - 90) so the principal axis (currently at angle_deg
    # from the x-axis) ends up aligned with the vertical (y) axis.
    rotation_deg = axis.angle_deg - 90.0
    rot_mat = cv2.getRotationMatrix2D(axis.center_xy, rotation_deg, 1.0)

    # Expand canvas so rotated content isn't clipped.
    corners = np.array(
        [[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32
    )
    ones = np.ones((4, 1), dtype=np.float32)
    rot_corners = (rot_mat @ np.hstack([corners, ones]).T).T
    min_xy = rot_corners.min(axis=0)
    max_xy = rot_corners.max(axis=0)
    new_w = int(np.ceil(max_xy[0] - min_xy[0]))
    new_h = int(np.ceil(max_xy[1] - min_xy[1]))
    rot_mat[0, 2] -= min_xy[0]
    rot_mat[1, 2] -= min_xy[1]

    rotated_image = cv2.warpAffine(image, rot_mat, (new_w, new_h))
    rotated_mask = cv2.warpAffine(mask, rot_mat, (new_w, new_h))
    _, rotated_mask = cv2.threshold(rotated_mask, 127, 255, cv2.THRESH_BINARY)

    return rotated_image, rotated_mask
