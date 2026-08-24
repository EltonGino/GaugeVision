"""Measurement validation against synthetic ground truth — CLAUDE.md §4.2,
§8 (Phase 4).

Quantifies measurement error (mm, %) for major diameter and pitch across a
sweep of thread sizes and clean/blur/rotated conditions, and produces the
quantitative ``PeakPitchEstimator`` vs ``FFTPitchEstimator`` comparison
table CLAUDE.md's measurement module asks for. Every number here is produced
by actually running the measurement pipeline against ``data/
synthetic_thread.py`` samples with known ground truth — see
``docs/RESULTS.md`` for the real numbers this produced.

Uses ``MetricCalibration.synthetic_reference`` — validated *within this
harness only*; see that classmethod's docstring for why it must not be
applied to real MVTec images.
"""

from __future__ import annotations

from dataclasses import dataclass

from gaugevision.calibration.metric_calibration import MetricCalibration
from gaugevision.data.synthetic_thread import (
    SyntheticThreadSpec,
    generate_synthetic_thread,
)
from gaugevision.decision.iso965_table import ISO_965_TOLERANCE_TABLE
from gaugevision.measurement.pipeline import run_measurement_pipeline
from gaugevision.measurement.pitch import (
    FFTPitchEstimator,
    PeakPitchEstimator,
    PitchEstimator,
)

CONDITIONS = ("clean", "blur", "rotated")

# Thread sizes reused from the ISO 965 starter table (decision/iso965_table.py)
# for consistency between the decision layer and the validation harness.
# Minor diameter approximated as major - 1.2*pitch, a standard-ish external
# thread proportion — this is a synthetic rendering parameter, not a claim
# about any specific real thread standard's minor-diameter formula.
VALIDATION_SIZES: tuple[str, ...] = ("M3", "M6", "M10")


def _spec_for_designation(designation: str) -> SyntheticThreadSpec:
    entry = ISO_965_TOLERANCE_TABLE[designation]
    major = entry.major_diameter_basic_mm
    pitch = entry.pitch_mm
    minor = major - 1.2 * pitch
    return SyntheticThreadSpec(major_diameter_mm=major, minor_diameter_mm=minor, pitch_mm=pitch)


@dataclass(frozen=True)
class ValidationRecord:
    designation: str
    condition: str
    estimator_name: str
    true_diameter_mm: float
    measured_diameter_mm: float
    diameter_abs_error_mm: float
    diameter_pct_error: float
    true_pitch_mm: float
    measured_pitch_mm: float | None
    pitch_abs_error_mm: float | None
    pitch_pct_error: float | None
    confidence: float


def _run_one(
    designation: str, condition: str, estimator: PitchEstimator, px_per_mm: float
) -> ValidationRecord:
    spec = _spec_for_designation(designation)
    sample = generate_synthetic_thread(spec, px_per_mm=px_per_mm, condition=condition)
    calibration = MetricCalibration.synthetic_reference(
        sample.reference_bar_length_px, sample.reference_bar_length_mm
    )
    result = run_measurement_pipeline(
        sample.image, calibration=calibration, pitch_estimator=estimator
    )

    diameter_abs_error = abs(result.major_diameter_mm - spec.major_diameter_mm)
    diameter_pct_error = 100.0 * diameter_abs_error / spec.major_diameter_mm

    pitch_abs_error = None
    pitch_pct_error = None
    if result.pitch_mm is not None:
        pitch_abs_error = abs(result.pitch_mm - spec.pitch_mm)
        pitch_pct_error = 100.0 * pitch_abs_error / spec.pitch_mm

    return ValidationRecord(
        designation=designation,
        condition=condition,
        estimator_name=estimator.__class__.__name__,
        true_diameter_mm=spec.major_diameter_mm,
        measured_diameter_mm=result.major_diameter_mm,
        diameter_abs_error_mm=diameter_abs_error,
        diameter_pct_error=diameter_pct_error,
        true_pitch_mm=spec.pitch_mm,
        measured_pitch_mm=result.pitch_mm,
        pitch_abs_error_mm=pitch_abs_error,
        pitch_pct_error=pitch_pct_error,
        confidence=result.confidence,
    )


def run_validation_sweep(
    designations: tuple[str, ...] = VALIDATION_SIZES,
    conditions: tuple[str, ...] = CONDITIONS,
    px_per_mm: float = 20.0,
) -> list[ValidationRecord]:
    """Run both pitch estimators against every (size, condition) combination."""
    estimators: list[PitchEstimator] = [PeakPitchEstimator(), FFTPitchEstimator()]
    records: list[ValidationRecord] = []
    for designation in designations:
        for condition in conditions:
            for estimator in estimators:
                records.append(_run_one(designation, condition, estimator, px_per_mm))
    return records


@dataclass(frozen=True)
class EstimatorConditionSummary:
    estimator_name: str
    condition: str
    n_cases: int
    diameter_mae_mm: float
    diameter_mean_pct_error: float
    pitch_mae_mm: float | None  # None if no case produced a pitch estimate
    pitch_mean_pct_error: float | None
    n_pitch_estimates_produced: int


def summarize(records: list[ValidationRecord]) -> list[EstimatorConditionSummary]:
    """Aggregate per (estimator, condition) — mirrors the comparison table
    shape in CLAUDE.md §4.2."""
    keys = sorted({(r.estimator_name, r.condition) for r in records})
    summaries = []
    for estimator_name, condition in keys:
        group = [r for r in records if r.estimator_name == estimator_name and r.condition == condition]
        diameter_mae = sum(r.diameter_abs_error_mm for r in group) / len(group)
        diameter_pct = sum(r.diameter_pct_error for r in group) / len(group)

        pitch_group = [r for r in group if r.pitch_abs_error_mm is not None]
        pitch_mae = (
            sum(r.pitch_abs_error_mm for r in pitch_group) / len(pitch_group)
            if pitch_group
            else None
        )
        pitch_pct = (
            sum(r.pitch_pct_error for r in pitch_group) / len(pitch_group)
            if pitch_group
            else None
        )

        summaries.append(
            EstimatorConditionSummary(
                estimator_name=estimator_name,
                condition=condition,
                n_cases=len(group),
                diameter_mae_mm=diameter_mae,
                diameter_mean_pct_error=diameter_pct,
                pitch_mae_mm=pitch_mae,
                pitch_mean_pct_error=pitch_pct,
                n_pitch_estimates_produced=len(pitch_group),
            )
        )
    return summaries


def format_pitch_comparison_table(summaries: list[EstimatorConditionSummary]) -> str:
    """Markdown table matching CLAUDE.md §4.2's example shape:
    Method | Clean MAE | Blur MAE | Rotated MAE (pitch, mm)."""
    estimators = sorted({s.estimator_name for s in summaries})
    conditions = [c for c in CONDITIONS]

    header = "| Method | " + " | ".join(f"{c.capitalize()} pitch MAE (mm)" for c in conditions) + " |"
    sep = "|---" * (len(conditions) + 1) + "|"
    rows = [header, sep]
    for estimator_name in estimators:
        cells = [estimator_name]
        for condition in conditions:
            match = next(
                (s for s in summaries if s.estimator_name == estimator_name and s.condition == condition),
                None,
            )
            if match is None or match.pitch_mae_mm is None:
                cells.append(f"n/a ({match.n_pitch_estimates_produced if match else 0}/{match.n_cases if match else 0} produced an estimate)")
            else:
                cells.append(
                    f"{match.pitch_mae_mm:.4f} ({match.n_pitch_estimates_produced}/{match.n_cases} produced an estimate)"
                )
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)
