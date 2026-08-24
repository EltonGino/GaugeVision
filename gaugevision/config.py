"""Explicit, centralized Phase-1 configuration for GaugeVision.

Kept deliberately small and flat for Phase 1. Values that become
env/CLI-configurable in later phases (e.g. a validated calibration source in
Phase 4) should still route through this module rather than being scattered
through the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

# Anomaly detection
ANOMALY_MODEL_PATH = Path(os.environ.get("GAUGEVISION_MODEL_PATH", "models/padim_screw.npz"))
ANOMALY_N_FEATURES = 100
ANOMALY_BACKBONE = "resnet18"

# Metric calibration (CLAUDE.md §3.2, §4.1b) — Phase-1 demonstration reference.
# No physical reference object exists behind this scale: MVTec provides no
# real-world mm scale for its images. Phase 4 replaces this with a validated
# synthetic/known-reference calibration source.
DEMO_CALIBRATION_PX_PER_MM = 32.4

# Measurement pipeline
THREAD_HEAD_MARGIN_FRACTION = 0.22

# API
API_HOST = os.environ.get("GAUGEVISION_API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("GAUGEVISION_API_PORT", "8000"))

# Gradio UI -> API
API_BASE_URL = os.environ.get("GAUGEVISION_API_URL", "http://localhost:8000")
