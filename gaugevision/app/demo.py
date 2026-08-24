"""Gradio UI — CLAUDE.md §4.6.

The UI layer only: every inspection call goes through the FastAPI service
over HTTP (``requests``), never imports the pipeline modules directly.
"""

from __future__ import annotations

import base64
import io

import gradio as gr
import requests
from PIL import Image

from gaugevision import config


def inspect(image: Image.Image):
    if image is None:
        return None, "No image provided.", {}, {}

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    try:
        response = requests.post(
            f"{config.API_BASE_URL}/inspect/image",
            files={"file": ("image.png", buf, "image/png")},
            timeout=30,
        )
    except requests.exceptions.ConnectionError:
        return (
            None,
            f"Could not reach GaugeVision API at {config.API_BASE_URL}. Is it running?",
            {},
            {},
        )

    if response.status_code != 200:
        detail = response.text
        return None, f"API error ({response.status_code}): {detail}", {}, {}

    payload = response.json()
    verdict = payload["verdict"]

    heatmap_bytes = base64.b64decode(payload["anomaly_heatmap_png_base64"])
    heatmap_img = Image.open(io.BytesIO(heatmap_bytes))

    verdict_text = (
        f"## Verdict: {verdict['verdict']}\n\n"
        f"**Failed checks:** {', '.join(verdict['failed_checks']) or 'none'}\n\n"
        + "\n".join(f"- {r}" for r in verdict["reasoning"])
    )

    measurements = {
        "major_diameter_mm": verdict["measurements"]["major_diameter_mm"],
        "pitch_mm": verdict["measurements"]["pitch_mm"],
        "matched_thread_designation": verdict["matched_thread_designation"],
        "measurement_confidence": verdict["measurement_confidence"],
        "calibration_source": verdict["calibration_source"],
        "calibrated": verdict["calibrated"],
    }

    scores = {
        "anomaly_score": verdict["anomaly_score"],
        "anomaly_threshold": verdict["anomaly_threshold"],
        "inference_ms": verdict["inference_ms"],
    }

    return heatmap_img, verdict_text, measurements, scores


with gr.Blocks(title="GaugeVision — Screw Inspection Demo") as demo:
    gr.Markdown(
        "# GaugeVision — Industrial Screw Inspection (Phase 1 demo)\n\n"
        "Upload a screw image (e.g. from the MVTec AD 'screw' test set) to "
        "run the full pipeline: measurement + anomaly detection + "
        "ISO-informed Go/No-Go decision fusion.\n\n"
        "**This UI calls the GaugeVision FastAPI service — it does not run "
        "inference itself.**"
    )
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(type="pil", label="Screw image")
            submit_btn = gr.Button("Inspect", variant="primary")
        with gr.Column():
            output_heatmap = gr.Image(type="pil", label="Anomaly heatmap")
            output_verdict = gr.Markdown(label="Verdict")
            output_measurements = gr.JSON(label="Measurements & calibration")
            output_scores = gr.JSON(label="Anomaly score & latency")

    submit_btn.click(
        fn=inspect,
        inputs=[input_image],
        outputs=[output_heatmap, output_verdict, output_measurements, output_scores],
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
