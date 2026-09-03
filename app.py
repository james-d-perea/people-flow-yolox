from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import gradio as gr

from pipeline import MAX_VIDEO_DURATION_SEC, analyze_video_stream
from utils import discover_video_examples


DEFAULT_UNRAISABLE_HOOK = sys.unraisablehook
BASE_DIR = Path(__file__).resolve().parent


DETECTION_MODE_HEURISTIC = "Heuristic CV"
DETECTION_MODE_YOLO = "YOLO"


def suppress_asyncio_cleanup_noise(unraisable) -> None:
    exc = unraisable.exc_value
    obj = unraisable.object
    is_asyncio_del = obj is asyncio.BaseEventLoop.__del__
    is_invalid_fd = isinstance(exc, ValueError) and "Invalid file descriptor" in str(exc)

    if is_asyncio_del and is_invalid_fd:
        return

    DEFAULT_UNRAISABLE_HOOK(unraisable)


sys.unraisablehook = suppress_asyncio_cleanup_noise


def hide_final_video_css() -> str:
    return "<style>#final-video-wrap { display: none !important; }</style>"


def show_final_video_css() -> str:
    return "<style>#final-video-wrap { display: block !important; }</style>"


def find_video_examples() -> list[list[str]]:
    examples = discover_video_examples(BASE_DIR / "assets" / "videos")

    if not examples:
        examples = discover_video_examples(BASE_DIR / "assets")

    return [[str(example)] for example in examples]


def render_progress(percent: int, label: str) -> str:
    percent = max(0, min(int(percent), 100))
    return (
        '<div class="pf-progress" role="progressbar" '
        f'aria-valuemin="0" aria-valuemax="100" aria-valuenow="{percent}">'
        '<div class="pf-progress-meta">'
        f'<span>{label}</span><strong>{percent}%</strong>'
        '</div>'
        '<div class="pf-progress-track">'
        f'<div class="pf-progress-fill" style="width: {percent}%"></div>'
        '</div>'
        '</div>'
    )


def run_video_analysis(video, detection_mode: str):
    if video is None:
        raise gr.Error("Upload a video before running analysis.")

    if detection_mode not in {DETECTION_MODE_HEURISTIC, DETECTION_MODE_YOLO}:
        detection_mode = DETECTION_MODE_YOLO

    yield (
        gr.update(value=hide_final_video_css()),
        gr.update(value=None),
        render_progress(0, f"Starting analysis with {detection_mode}..."),
        gr.update(value=None, visible=False),
        None,
        None,
        None,
        None,
        None,
        "{}",
        "[]",
    )

    last_output = None

    try:
        for update in analyze_video_stream(video, detection_mode=detection_mode):
            stages = update["stages"]
            metrics = update["metrics"]

            if update["is_final"]:
                annotated_video_path = update.get("annotated_video")

                if not annotated_video_path or not Path(annotated_video_path).exists():
                    raise gr.Error("Annotated video file was not created.")

                status = render_progress(100, "Analysis complete. Annotated video is ready.")
                video_css = gr.update(value=show_final_video_css())
                video_output = gr.update(value=str(annotated_video_path))
                preview_output = gr.update(value=None, visible=False)
            else:
                frames_total = max(int(metrics.get("frames_total", 1)), 1)
                current_frame = min(int(metrics.get("current_frame", 0)), frames_total)
                progress = int(round((current_frame / frames_total) * 100))

                label = (
                    f"Processing frame {metrics.get('current_frame', 0)} "
                    f"of {metrics.get('frames_total', frames_total)} "
                    f"at {metrics.get('analysis_fps', '?')} FPS "
                    f"with {detection_mode}..."
                )

                status = render_progress(progress, label)
                video_css = gr.update(value=hide_final_video_css())
                video_output = gr.update(value=None)
                preview_output = gr.update(value=stages["detection_overlay"], visible=True)

            last_output = (
                video_css,
                video_output,
                status,
                preview_output,
                stages["original"],
                stages["enhanced"],
                stages["segmentation_mask"],
                stages["cleaned_mask"],
                stages["detection_overlay"],
                json.dumps(metrics, indent=2, ensure_ascii=False),
                json.dumps(update["detections"], indent=2, ensure_ascii=False),
            )

            yield last_output

    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    if last_output is None:
        raise gr.Error("No video preview was generated.")


video_examples = find_video_examples()
print("VIDEO EXAMPLES:", video_examples, flush=True)


theme = gr.themes.Base(
    primary_hue="cyan",
    secondary_hue="pink",
    neutral_hue="slate",
).set(
    body_background_fill="#080a12",
    body_text_color="#f1e6b8",
    block_background_fill="#10131f",
    block_border_color="#2ad6b7",
    block_label_background_fill="#05070d",
    block_label_text_color="#f1e6b8",
    button_primary_background_fill="#f2b84b",
    button_primary_text_color="#080a12",
)


with gr.Blocks(
    title="People Flow CV",
    analytics_enabled=False,
) as demo:
    gr.Markdown(
        (
            "# Automated People Flow Tracking\n"
            f"Upload a video up to {MAX_VIDEO_DURATION_SEC}s and generate tracked flow overlays."
        ),
        elem_classes="pf-title",
    )

    with gr.Row(elem_classes="pf-workspace"):
        with gr.Column(scale=1, elem_classes="pf-panel"):
            input_video = gr.Video(label="Upload video")

            detection_mode = gr.Radio(
                choices=[DETECTION_MODE_YOLO, DETECTION_MODE_HEURISTIC],
                value=DETECTION_MODE_YOLO,
                label="Detection method",
                info=(
                    "YOLO uses YOLOX-Nano, a pretrained-on-COCO detector released "
                    "under the Apache License 2.0. Heuristic CV is the original "
                    "classical motion-based pipeline, kept as a comparison mode."
                ),
                elem_classes="pf-radio",
            )

            analyze_button = gr.Button("Analyze video", variant="primary")

            if video_examples:
                gr.Examples(
                    examples=video_examples,
                    inputs=[input_video],
                    label="Example videos",
                )
            else:
                gr.Markdown(
                    "No example videos found. Put .mp4 files into `assets/videos/` or `assets/`."
                )

        with gr.Column(scale=1, elem_classes="pf-output-panel"):
            video_visibility_css = gr.HTML(hide_final_video_css())

            output_video = gr.Video(
                label="Annotated output video",
                visible=True,
                elem_id="final-video-wrap",
            )

            processing_status = gr.HTML(
                render_progress(0, "Idle. Upload a video and start analysis."),
                elem_classes="pf-status",
            )

            annotated_preview = gr.Image(
                label="Annotated preview while processing",
                visible=False,
            )

            metrics = gr.Code(label="Video metrics", language="json")

    with gr.Tabs():
        with gr.Tab("Live Preview"):
            with gr.Row():
                original = gr.Image(label="1. Current frame", elem_classes="pf-stage")
                enhanced = gr.Image(label="2. Enhanced frame", elem_classes="pf-stage")

            with gr.Row():
                segmentation_mask = gr.Image(label="3. Segmentation mask", elem_classes="pf-stage")
                cleaned_mask = gr.Image(label="4. Cleaned mask", elem_classes="pf-stage")

            detection_overlay = gr.Image(label="5. Tracking overlay", elem_classes="pf-stage")

        with gr.Tab("Detection Details"):
            detections = gr.Code(label="Current detections and centroids", language="json")

    analyze_button.click(
        fn=run_video_analysis,
        inputs=[input_video, detection_mode],
        outputs=[
            video_visibility_css,
            output_video,
            processing_status,
            annotated_preview,
            original,
            enhanced,
            segmentation_mask,
            cleaned_mask,
            detection_overlay,
            metrics,
            detections,
        ],
        api_name=False,
    )


if __name__ == "__main__":
    launch_kwargs = {
        "server_name": "0.0.0.0",
        "server_port": int(os.environ.get("PORT", 7860)),
        "theme": theme,
        "css_paths": "app.css",
        "show_error": True,
        "ssr_mode": False,
    }

    demo.queue(default_concurrency_limit=1).launch(**launch_kwargs)
