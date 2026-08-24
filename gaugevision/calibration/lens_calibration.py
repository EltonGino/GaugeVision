"""Lens / geometric calibration — CLAUDE.md §4.1a.

Standard OpenCV chessboard calibration: intrinsics + distortion coefficients,
used to undistort images. This is a *capability demonstration* only — there is
no physical camera behind this project, so calibration runs against a
procedurally rendered synthetic checkerboard set (multiple perspective views
of a known pattern) rather than photos from a real camera. It is NOT a
calibration of any specific real camera, and it is a separate concern from
metric (px->mm) scale calibration in ``metric_calibration.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraCalibration:
    camera_matrix: np.ndarray
    dist_coeffs: np.ndarray
    rms_reprojection_error: float
    image_size: tuple[int, int]  # (width, height)
    source: str = "synthetic_checkerboard"


def generate_synthetic_checkerboard_views(
    pattern_size: tuple[int, int] = (9, 6),
    square_px: int = 40,
    image_size: tuple[int, int] = (640, 480),
    n_views: int = 12,
    seed: int = 0,
) -> list[np.ndarray]:
    """Render a flat checkerboard under random perspective warps.

    Stands in for photographing a printed checkerboard from several angles,
    which is the standard input to ``cv2.calibrateCamera``.
    """
    rng = np.random.default_rng(seed)
    cols, rows = pattern_size

    board_w = (cols + 1) * square_px
    board_h = (rows + 1) * square_px
    board = np.full((board_h, board_w), 255, dtype=np.uint8)
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                y0, y1 = r * square_px, (r + 1) * square_px
                x0, x1 = c * square_px, (c + 1) * square_px
                board[y0:y1, x0:x1] = 0

    views = []
    W, H = image_size
    src = np.float32([[0, 0], [board_w, 0], [board_w, board_h], [0, board_h]])
    for _ in range(n_views):
        jitter = rng.uniform(-0.12, 0.12, size=(4, 2))
        base = np.float32(
            [
                [0.15 * W, 0.15 * H],
                [0.85 * W, 0.15 * H],
                [0.85 * W, 0.85 * H],
                [0.15 * W, 0.85 * H],
            ]
        )
        dst = base + jitter * np.array([W, H])
        dst = dst.astype(np.float32)
        homography = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(
            board, homography, (W, H), borderValue=200
        )
        views.append(warped)
    return views


def calibrate_camera(
    images: list[np.ndarray],
    pattern_size: tuple[int, int] = (9, 6),
) -> CameraCalibration:
    """Run standard OpenCV chessboard calibration over a set of views."""
    if not images:
        raise ValueError("calibrate_camera requires at least one image")

    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None

    for img in images:
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        image_size = (gray.shape[1], gray.shape[0])
        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            continue
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(objp)
        image_points.append(corners)

    if len(object_points) < 3:
        raise RuntimeError(
            f"Chessboard detected in only {len(object_points)}/{len(images)} views; "
            "need at least 3 for a stable calibration."
        )

    rms, camera_matrix, dist_coeffs, _, _ = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    return CameraCalibration(
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        rms_reprojection_error=float(rms),
        image_size=image_size,
    )


def undistort(image: np.ndarray, calibration: CameraCalibration) -> np.ndarray:
    return cv2.undistort(image, calibration.camera_matrix, calibration.dist_coeffs)
