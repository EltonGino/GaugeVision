"""Starter ISO-informed major-diameter tolerance table — CLAUDE.md §4.5.

This is a **starting reference table to extend, not exhaustive ISO 965
coverage**. ISO 965 defines multiple tolerance classes/qualities per size;
this table covers only tolerance class **6g** (the standard external-thread
class for commercial bolts/screws, medium fit) for the coarse-pitch series
M1.6-M12. It is used to demonstrate standards-aware decision logic, not as a
certified dimensional inspection reference.

Basic (nominal) major diameter and coarse pitch values are ISO 724 basic
profile dimensions (uncontested, standard). The 6g major-diameter max/min
limits were cross-referenced against multiple independent public engineering
references (engineersedge.com "External Metric Thread Table Chart",
boltbase.com "Metric Thread Tolerances", amesweb.info metric thread
calculator) rather than re-derived from the ISO 965-1 deviation formulas, and
should be treated as a demonstration-quality starter table — extend/verify
against the ISO 965-1 standard text directly before using for anything beyond
this portfolio demo.

M1 and M1.2 are intentionally omitted: ISO 965-2 specifies those two sizes
under tolerance class 5h6h rather than 6g, so a 6g entry for them would not
be meaningful without pulling in a second tolerance-class table.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThreadToleranceEntry:
    designation: str  # e.g. "M6"
    pitch_mm: float
    major_diameter_basic_mm: float
    major_diameter_max_mm: float
    major_diameter_min_mm: float
    tolerance_class: str = "6g"


# Coarse-pitch series, external thread, tolerance class 6g.
ISO_965_TOLERANCE_TABLE: dict[str, ThreadToleranceEntry] = {
    e.designation: e
    for e in [
        ThreadToleranceEntry("M1.6", 0.35, 1.600, 1.581, 1.496),
        ThreadToleranceEntry("M2", 0.40, 2.000, 1.981, 1.886),
        ThreadToleranceEntry("M2.5", 0.45, 2.500, 2.480, 2.380),
        ThreadToleranceEntry("M3", 0.50, 3.000, 2.980, 2.874),
        ThreadToleranceEntry("M4", 0.70, 4.000, 3.978, 3.838),
        ThreadToleranceEntry("M5", 0.80, 5.000, 4.976, 4.826),
        ThreadToleranceEntry("M6", 1.00, 6.000, 5.974, 5.794),
        ThreadToleranceEntry("M8", 1.25, 8.000, 7.972, 7.760),
        ThreadToleranceEntry("M10", 1.50, 10.000, 9.968, 9.732),
        ThreadToleranceEntry("M12", 1.75, 12.000, 11.966, 11.701),
    ]
}


def lookup_by_designation(designation: str) -> ThreadToleranceEntry:
    try:
        return ISO_965_TOLERANCE_TABLE[designation]
    except KeyError as e:
        raise KeyError(
            f"No starter-table entry for {designation!r}. "
            f"Available: {sorted(ISO_965_TOLERANCE_TABLE)}"
        ) from e


def nearest_designation(major_diameter_mm: float) -> str:
    """Nearest table entry by basic major diameter — used when the nominal
    size isn't known ahead of time (e.g. MVTec images have no size label)."""
    return min(
        ISO_965_TOLERANCE_TABLE,
        key=lambda d: abs(
            ISO_965_TOLERANCE_TABLE[d].major_diameter_basic_mm - major_diameter_mm
        ),
    )
