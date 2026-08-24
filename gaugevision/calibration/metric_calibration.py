"""Metric (px -> mm) calibration — CLAUDE.md §4.1b.

This is deliberately a *separate* concept from lens/geometric calibration
(see ``lens_calibration.py``). Lens calibration corrects distortion; it never
by itself tells you how many millimeters a pixel represents. Recovering that
scale requires a reference object of known real-world size in the scene (or an
equivalent known-dimension target/homography).

Every pixel<->mm conversion in the measurement pipeline must go through
``MetricCalibration`` — never a bare pixel/constant division inline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MetricCalibration:
    """Linear px<->mm scale plus provenance of where that scale came from.

    Phase 1 constructs this from a configured demonstration reference
    (``source="demo_reference"``, ``validated=False``). Phase 4 replaces the
    source with one validated against a synthetic/known-reference target
    (``validated=True``) — same interface, no rewrite required.
    """

    px_per_mm: float
    source: str
    validated: bool = False

    def __post_init__(self) -> None:
        if self.px_per_mm <= 0:
            raise ValueError(f"px_per_mm must be positive, got {self.px_per_mm}")

    @classmethod
    def from_reference(
        cls,
        reference_length_px: float,
        reference_length_mm: float,
        source: str,
        validated: bool = False,
    ) -> MetricCalibration:
        """Derive px_per_mm from a reference object of known physical size."""
        if reference_length_mm <= 0:
            raise ValueError("reference_length_mm must be positive")
        if reference_length_px <= 0:
            raise ValueError("reference_length_px must be positive")
        return cls(
            px_per_mm=reference_length_px / reference_length_mm,
            source=source,
            validated=validated,
        )

    @classmethod
    def demo_reference(cls, px_per_mm: float = 32.4) -> MetricCalibration:
        """Phase-1 demonstration calibration.

        ``px_per_mm=32.4`` is a configured stand-in scale, not derived from a
        measured reference object in the MVTec images (MVTec provides no
        physical scale reference). Downstream consumers must treat resulting
        mm values as illustrative, not dimensionally validated — see
        ``MeasurementResult.calibrated``.
        """
        return cls(px_per_mm=px_per_mm, source="demo_reference", validated=False)

    def pixels_to_mm(self, value_px: float) -> float:
        return value_px / self.px_per_mm

    def mm_to_pixels(self, value_mm: float) -> float:
        return value_mm * self.px_per_mm
