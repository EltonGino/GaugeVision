"""Thread-region isolation and profile extraction — measurement pipeline steps
3-5 (CLAUDE.md §4.2).

Per-row silhouette width, in the derotated (axis-vertical) frame, is used
directly as the 1D signal: thread crests/roots make the shank's width
oscillate at the thread pitch, so width-per-row is already a periodic signal
without needing separate upper/lower edge extraction.

Head vs. shank separation (step 3) uses a simple, documented heuristic:
MVTec screw heads are the widest part of the part, so the row of maximum
width marks the head; a configurable margin fraction of the total length is
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

    Args:
        width_profile: per-row width, index 0 = top of the derotated frame.
        head_margin_fraction: fraction of total length excluded from the
            head end of the part.

    Returns:
        ThreadRegion describing the row range treated as thread-bearing.
    """
    n = len(width_profile)
    if n == 0:
        raise RuntimeError("isolate_thread_region: empty width profile")

    head_row = int(np.argmax(width_profile))
    margin_rows = round(n * head_margin_fraction)

    # Head is nearer whichever end the max-width row falls closer to.
    if head_row <= n / 2:
        row_start = min(margin_rows, n - 1)
        row_end = n
    else:
        row_start = 0
        row_end = max(n - margin_rows, row_start + 1)

    if row_end - row_start < max(10, int(0.2 * n)):
        # Degenerate case (e.g. very short part): fall back to the full
        # profile rather than isolating an empty/near-empty region.
        row_start, row_end = 0, n

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
