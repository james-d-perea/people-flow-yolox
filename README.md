---
title: People Flow CV
emoji: "🚶"
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.18.0
python_version: 3.11
app_file: app.py
pinned: false
license: mit
---

# People Flow CV

People Flow CV is a Gradio app for analyzing pedestrian movement in short
surveillance-style videos. It segments motion, detects people, tracks them
across frames, infers the dominant flow direction, and produces an annotated
output video.

> **Fork note:** this is a copy of
> [lolita-malaeva/people-flow](https://huggingface.co/spaces/lolita-malaeva/people-flow)
> with the YOLO detector backend replaced. The original YOLO mode loaded
> Ultralytics YOLOv8 weights (`yolov8n.pt`) via the `ultralytics` package,
> which are distributed under the **AGPL-3.0** license. This fork instead
> uses **YOLOX-Nano**, a person/COCO detector pretrained and released by
> Megvii under the **Apache License 2.0** (model, weights, and reference
> pre/post-processing code). See `NOTICE.md` for full attribution. YOLO is
> now also the default detection mode (it was an optional comparison mode
> in the original); the original classical OpenCV pipeline is kept as the
> "Heuristic CV" mode.

The app is designed for quick demos, experiments, and Hugging Face CPU Spaces.
Neither detector requires a GPU.

## What the app produces

- Annotated video with tracked objects, IDs, movement labels, and a counting line.
- Live processing preview while the final video is being generated.
- Frame-stage previews: original frame, enhanced frame, segmentation mask,
  cleaned mask, and tracking overlay.
- JSON metrics with tracked counts, flow direction, inferred camera view,
  crossing counts, FPS information, and processing settings.

## Pipeline overview

1. **Frame sampling**
   The video is read with OpenCV and sampled to a target analysis FPS. This keeps
   processing lightweight on CPU.

2. **Resize**
   Frames are resized to a fixed processing width for predictable runtime.

3. **Enhancement**
   Contrast is improved with CLAHE on the LAB lightness channel, followed by a
   light Gaussian blur.

4. **Motion segmentation**
   Moving regions are extracted with MOG2 background subtraction (computed in
   both modes so the UI can always show the full set of CV stages).

5. **Mask cleanup**
   Morphological opening, closing, dilation, and connected-component filtering
   remove small noise and stabilize motion blobs.

6. **Person detection** — two selectable backends:
   - **YOLO** (default): YOLOX-Nano, a pretrained-on-COCO neural detector
     (Apache-2.0), run on the enhanced frame via `onnxruntime`.
   - **Heuristic CV**: the original classical pipeline, which builds
     pedestrian candidates from cleaned motion components using shape, area,
     mask coverage, and head/body likeness heuristics.

7. **Centroid tracking**
   Detections are linked across frames with a centroid tracker that accounts for
   predicted position, area changes, source consistency, and temporary misses.

8. **Flow decision**
   Confirmed tracks are used to infer the view mode, classify movement direction,
   count line crossings, and select the dominant flow.

9. **Overlay rendering**
   The annotated output video is rendered with bounding boxes, track IDs,
   direction labels, a counting line, and summary metrics.

## Configuration

Main pipeline settings live in `pipeline/config.py`.

```python
PROC_WIDTH = 640
MAX_VIDEO_DURATION_SEC = 60
TARGET_ANALYSIS_FPS = 12.0
MIN_TRACK_HITS = 4
MAX_TRACK_MISSING = 16
MAX_MATCH_DISTANCE = 75.0

# YOLO backend (YOLOX-Nano, Apache-2.0, COCO-pretrained)
YOLOX_MODEL_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"
YOLOX_INPUT_SIZE = 416
YOLOX_SCORE_THRESHOLD = 0.35
YOLOX_NMS_THRESHOLD = 0.45
```

The defaults are tuned for short pedestrian clips and CPU execution:

- `PROC_WIDTH` keeps frame processing fast and consistent.
- `MAX_VIDEO_DURATION_SEC` prevents long uploads from blocking the app.
- `TARGET_ANALYSIS_FPS` reduces workload while preserving motion continuity.
- Tracking thresholds control when a moving candidate becomes a confirmed track
  and how long it can disappear before being dropped.
- `YOLOX_MODEL_URL` points at the Apache-2.0 licensed pretrained weights; they
  are downloaded once and cached under `.cache/yolox/` on first use.

## Project structure

```text
.
├── app.py                  # Gradio UI and streaming updates
├── app.css                 # Retro dark UI styling
├── pipeline/
│   ├── config.py           # Pipeline constants (incl. YOLOX settings)
│   ├── models.py           # Detection and Track dataclasses
│   ├── preprocessing.py    # Resize, enhancement, mask cleanup
│   ├── segmentation.py     # MOG2 background subtraction
│   ├── detection.py        # Motion-driven candidate detection (Heuristic CV)
│   ├── yolo_detection.py   # YOLOX-Nano person detection (YOLO mode)
│   ├── tracking.py         # Centroid tracker
│   ├── flow.py             # View mode, flow labels, crossing counts
│   ├── overlay.py          # Annotated frame rendering
│   ├── zones.py            # Static flicker filtering
│   └── video.py            # Video IO and streaming analysis
├── utils.py                # Asset discovery and image helpers
├── assets/                 # Example images/videos (see ATTRIBUTION.md)
├── requirements.txt
├── NOTICE.md                # Third-party (Apache-2.0) attribution for YOLOX
└── README.md
```

## Local run

```bash
pip install -r requirements.txt
python app.py
```

Then open the local Gradio URL and upload a short video, or use one of the
example videos from the UI (add your own `.mp4`/`.webm` files under
`assets/videos/` — see `assets/ATTRIBUTION.md`).

## Deploy to Hugging Face Spaces

1. Create a new Space at <https://huggingface.co/spaces>.
2. Select **Gradio** as the SDK and CPU hardware.
3. Upload or commit the project files, including `pipeline/`, `assets/`,
   `app.py`, `app.css`, `utils.py`, `requirements.txt`, `NOTICE.md`, and
   `README.md`.
4. Wait for the Space build to finish.
5. Open the Space and run the analyzer from the browser.

## Notes and limits

- Best suited for short videos with a mostly static camera.
- The Heuristic CV mode works on motion cues, so heavy camera shake, fast
  lighting changes, or dense occlusion can reduce its tracking quality; the
  YOLO mode is generally more robust to those since it detects people
  directly rather than inferring them from motion blobs.
- The YOLO mode's first run on a fresh machine needs network access once, to
  download the ~7MB YOLOX-Nano ONNX weights; after that they're cached
  locally under `.cache/yolox/`.

## Advantages

- Runs on CPU with a small dependency set (no PyTorch — just `onnxruntime`
  for the YOLO backend).
- No manual model download step required; weights are fetched and cached
  automatically like the previous backend was.
- Fast enough for short demo clips.
- Transparent pipeline: every major processing stage is visible in the UI.
- Modular code structure for easier tuning and extension.

## License

This project's own code is MIT licensed (see `LICENSE`). The YOLO detection
backend uses YOLOX-Nano, which is Apache-2.0 licensed by Megvii Inc.; see
`NOTICE.md` for details.
