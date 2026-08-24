"""Stable data contracts for the measurement pipeline (GaugeVision CLAUDE.md §4.2).

These types are fixed from Phase 1 onward. Phase 4 deepens the *implementations*
that produce and consume them; it does not change these shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field


class MeasurementResult(BaseModel):
    """Final output contract of the measurement pipeline (CLAUDE.md §4.2).

    ``calibrated`` reflects whether ``scale_px_per_mm`` came from a validated
    metric-calibration source (Phase 4) as opposed to the Phase-1 demonstration
    reference. Phase-1 results always have ``calibrated=False`` and
    ``calibration_source="demo_reference"`` — the mm values are real
    conversions of real pixel measurements, but the px-per-mm scale itself has
    not been validated against a known physical reference, so they must not be
    read as dimensionally validated ground truth.
    """

    major_diameter_px: float
    major_diameter_mm: float
    pitch_px: float | None
    pitch_mm: float | None
    scale_px_per_mm: float
    calibration_source: str
    calibrated: bool
    confidence: float = Field(ge=0.0, le=1.0)
    notes: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class SegmentationResult:
    """Binary silhouette of the screw and its bounding box in image coordinates."""

    mask: np.ndarray  # uint8, shape (H, W), values in {0, 255}
    bbox: tuple[int, int, int, int]  # x, y, w, h


@dataclass(frozen=True)
class AxisEstimate:
    """Longitudinal axis of the screw, estimated from the segmentation mask."""

    center_xy: tuple[float, float]
    angle_deg: float  # rotation applied to derotate the axis to vertical
    length_px: float


@dataclass(frozen=True)
class ThreadRegion:
    """Row range (in the derotated frame) isolated as thread-bearing, vs. head."""

    row_start: int
    row_end: int
    excluded_head_fraction: float


@dataclass(frozen=True)
class ThreadProfile:
    """1D signal extracted from the thread-bearing region for pitch estimation."""

    signal: np.ndarray  # width (px) per row, within the thread region
    axis_positions_px: np.ndarray  # row index along the derotated axis


@dataclass(frozen=True)
class PitchEstimate:
    """Result of a single PitchEstimator strategy."""

    pitch_px: float | None
    confidence: float
    method: str
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DiameterEstimate:
    """Result of major-diameter estimation."""

    major_diameter_px: float
    method: str
