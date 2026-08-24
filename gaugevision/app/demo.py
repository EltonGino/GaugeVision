"""Gradio UI — CLAUDE.md §4.6.

The UI layer only: every inspection call goes through the FastAPI service
over HTTP (``requests``), never imports the pipeline modules directly.
"""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

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


def inspect_video(video_path: str | None):
    if video_path is None:
        return None, "No video provided.", {}

    try:
        with open(video_path, "rb") as f:
            response = requests.post(
                f"{config.API_BASE_URL}/inspect/video",
                files={"file": (Path(video_path).name, f, "video/mp4")},
                timeout=300,
            )
    except requests.exceptions.ConnectionError:
        return (
            None,
            f"Could not reach GaugeVision API at {config.API_BASE_URL}. Is it running?",
            {},
        )

    if response.status_code != 200:
        return None, f"API error ({response.status_code}): {response.text}", {}

    payload = response.json()
    video_bytes = base64.b64decode(payload["annotated_video_base64"])
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        out_path = tmp.name

    summary_text = (
        f"## Overall verdict: {payload['overall_verdict']}\n\n"
        f"**Failed checks (any sampled frame):** "
        f"{', '.join(payload['overall_failed_checks']) or 'none'}\n\n"
        f"- Frames sampled: {payload['n_frames_sampled']} "
        f"(every {payload['sample_interval_frames']} frames, "
        f"source {payload['source_fps']:.1f} fps)\n"
        f"- Inspection throughput: "
        f"{payload['inspection_throughput_fps']:.1f} frames/sec\n"
    )
    if payload["notes"]:
        summary_text += "\n**Notes:**\n" + "\n".join(f"- {n}" for n in payload["notes"])

    frame_results = payload["frame_results"]
    frame_summary = {
        "n_frames_go": sum(1 for fr in frame_results if fr["verdict"]["verdict"] == "GO"),
        "n_frames_no_go": sum(1 for fr in frame_results if fr["verdict"]["verdict"] == "NO_GO"),
    }

    return out_path, summary_text, frame_summary


with gr.Blocks(title="GaugeVision — Screw Inspection Demo") as demo:
    gr.Markdown(
        "# GaugeVision — Industrial Screw Inspection\n\n"
        "Run the full pipeline (measurement + anomaly detection + "
        "ISO-informed Go/No-Go decision fusion) on a screw image or video.\n\n"
        "**This UI calls the GaugeVision FastAPI service — it does not run "
        "inference itself.**"
    )
    with gr.Tabs():
        with gr.Tab("Image"):
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Screw image")
                    image_submit_btn = gr.Button("Inspect", variant="primary")
                with gr.Column():
                    output_heatmap = gr.Image(type="pil", label="Anomaly heatmap")
                    output_verdict = gr.Markdown(label="Verdict")
                    output_measurements = gr.JSON(label="Measurements & calibration")
                    output_scores = gr.JSON(label="Anomaly score & latency")

            image_submit_btn.click(
                fn=inspect,
                inputs=[input_image],
                outputs=[output_heatmap, output_verdict, output_measurements, output_scores],
            )

        with gr.Tab("Video"):
            gr.Markdown(
                "File-based video input, not live streaming (CLAUDE.md §4.7) — "
                "upload a short video file. Frames are sampled at ~2/sec by "
                "default; the annotated output plays back at that sampled rate."
            )
            with gr.Row():
                with gr.Column():
                    input_video = gr.Video(label="Screw video")
                    video_submit_btn = gr.Button("Inspect video", variant="primary")
                with gr.Column():
                    output_video = gr.Video(label="Annotated output")
                    output_video_summary = gr.Markdown(label="Summary")
                    output_frame_summary = gr.JSON(label="Frame-level GO/NO_GO counts")

            video_submit_btn.click(
                fn=inspect_video,
                inputs=[input_video],
                outputs=[output_video, output_video_summary, output_frame_summary],
            )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
