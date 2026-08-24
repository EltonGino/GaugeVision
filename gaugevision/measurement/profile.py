"""Thread-region isolation and profile extraction — measurement pipeline steps
3-5 (CLAUDE.md §4.2).

Per-row silhouette width, in the derotated (axis-vertical) frame, is used
directly as the 1D signal: thread crests/roots make the shank's width
oscillate at the thread pitch, so width-per-row is already a periodic signal
without needing separate upper/lower edge extraction.

Head vs. shank separation (step 3) uses a simple, documented heuristic:
MVTec screw heads are the widest part of the part, so the row of maximum
width marks the head; a configurable margin fraction of the part's own
foreground *extent* (not the full frame — see ``isolate_thread_region``) is
excluded from that end. This is the "one defensible working baseline" called
for by CLAUDE.md §4.2's scope guardrail — robust head/tip detection is
Phase-4 depth work.
"""

from __future__ import annotations

import numpy as np

from gaugevision.measurement.types import ThreadProfile, ThreadRegion


def compute_width_profile(mask: np.ndarray) -> np.ndarray:
    """Per-row foreground width (px) of a derotated binary mask."""
    binary = mask > 0
    return binary.sum(axis=1).astype(np.float64)


def isolate_thread_region(
    width_profile: np.ndarray, head_margin_fraction: float = 0.22
) -> ThreadRegion:
    """Exclude a margin around the widest row (assumed to be the head).

    The margin is a fraction of the part's own foreground *extent* (the span
    between its first and last nonzero row), not the full frame length. Using
    the full frame length would over- or under-exclude whenever the derotated
    frame has background margin around the part instead of the part filling
    it edge-to-edge — e.g. a screw rendered on a canvas larger than its own
    bounding box would leak head pixels into the "thread" region, inflating
    the measured major diameter to the head's width.

    Args:
        width_profile: per-row width, index 0 = top of the derotated frame.
        head_margin_fraction: fraction of the part's own length excluded
            from the head end.

    Returns:
        ThreadRegion describing the row range treated as thread-bearing.
    """
    n = len(width_profile)
    if n == 0:
        raise RuntimeError("isolate_thread_region: empty width profile")

    nonzero = np.nonzero(width_profile)[0]
    if len(nonzero) == 0:
        raise RuntimeError("isolate_thread_region: width profile has no foreground")
    fg_start, fg_end = int(nonzero[0]), int(nonzero[-1]) + 1
    fg_length = fg_end - fg_start

    head_row = fg_start + int(np.argmax(width_profile[fg_start:fg_end]))
    margin_rows = round(fg_length * head_margin_fraction)

    # Head is nearer whichever end the max-width row falls closer to.
    if head_row <= (fg_start + fg_end) / 2:
        row_start = min(fg_start + margin_rows, fg_end - 1)
        row_end = fg_end
    else:
        row_start = fg_start
        row_end = max(fg_end - margin_rows, row_start + 1)

    if row_end - row_start < max(10, int(0.2 * fg_length)):
        # Degenerate case (e.g. very short part): fall back to the full
        # foreground extent rather than isolating an empty/near-empty region.
        row_start, row_end = fg_start, fg_end

    return ThreadRegion(
        row_start=row_start, row_end=row_end, excluded_head_fraction=head_margin_fraction
    )


def extract_thread_profile(
    width_profile: np.ndarray, region: ThreadRegion
) -> ThreadProfile:
    """Slice the width profile down to the isolated thread-bearing region."""
    signal = width_profile[region.row_start : region.row_end]
    positions = np.arange(region.row_start, region.row_end, dtype=np.int64)
    return ThreadProfile(signal=signal, axis_positions_px=positions)
