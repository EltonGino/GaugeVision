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
    (``source="demo_reference"``, ``validated=False``). Phase 4 adds a second
    source (``synthetic_reference``, ``validated=True``) derived from a known
    reference object in a *synthetic* scene — same interface, no rewrite
    required.

    Important scope note: the validated synthetic source proves the
    calibration math and downstream measurement pipeline are accurate when a
    real reference object of known size is actually present in the image
    (see ``measurement/validate.py``). It is deliberately **not** substituted
    into the live MVTec inference path (``api/main.py`` still uses
    ``demo_reference``) — MVTec photos contain no physical reference object
    at all (CLAUDE.md §3.2), so there is no real scale to validate against
    for that specific dataset. Applying the synthetic scale to MVTec pixels
    would fabricate a number with no relationship to those photos' actual
    scale; that would be a regression in honesty, not an upgrade in rigor.
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

    @classmethod
    def synthetic_reference(
        cls, reference_length_px: float, reference_length_mm: float
    ) -> MetricCalibration:
        """Phase-4 validated calibration, derived from a known-size reference
        object actually rendered into a synthetic scene (``data/
        synthetic_thread.py``) — used only within the measurement validation
        harness (``measurement/validate.py``), never for real MVTec images
        (see the class docstring)."""
        return cls.from_reference(
            reference_length_px,
            reference_length_mm,
            source="synthetic_reference",
            validated=True,
        )

    def pixels_to_mm(self, value_px: float) -> float:
        return value_px / self.px_per_mm

    def mm_to_pixels(self, value_mm: float) -> float:
        return value_mm * self.px_per_mm
