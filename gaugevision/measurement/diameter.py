"""Major-diameter estimation — measurement pipeline step 6 (CLAUDE.md §4.2).

Major diameter is taken as the largest crest-to-crest width found within the
isolated thread-bearing region (the width profile already computed in
``profile.py``). This is consistent with the thread's major (crest) diameter
by definition, and reuses the same width signal used for pitch estimation
rather than introducing a second, inconsistent measurement method.
"""

from __future__ import annotations

import numpy as np

from gaugevision.measurement.types import DiameterEstimate, ThreadProfile


def estimate_major_diameter(thread_profile: ThreadProfile) -> DiameterEstimate:
    if thread_profile.signal.size == 0:
        raise RuntimeError("estimate_major_diameter: empty thread profile")
    major_diameter_px = float(np.max(thread_profile.signal))
    return DiameterEstimate(
        major_diameter_px=major_diameter_px, method="thread_region_max_width"
    )
