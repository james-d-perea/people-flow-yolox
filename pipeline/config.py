from pathlib import Path

PROC_WIDTH = 640
MIN_BLOB = 500
PREVIEW_EVERY = 12
MAX_VIDEO_DURATION_SEC = 60
TARGET_ANALYSIS_FPS = 12.0
MIN_TRACK_HITS = 4
MAX_TRACK_MISSING = 16
MAX_MATCH_DISTANCE = 75.0
MIN_COMPONENT_AREA = 220
STATIC_FLICKER_MAX_DISPLACEMENT = 18.0
STATIC_FLICKER_MIN_HITS = 5
STATIC_FLICKER_MIN_AREA_CHANGE = 0.35
IGNORED_ZONE_TTL = 90
IGNORED_ZONE_IOU = 0.35
VIDEO_SUFFIX = ".mp4"
VIEW_INFERENCE_MIN_TRACKS = 2

# --- YOLO detector settings ---------------------------------------------
#
# Detector: YOLOX-Nano, pretrained on COCO, released by Megvii under the
# Apache License 2.0 (both the model code and the exported weights below
# come from that project). This replaces the previous backend, which used
# Ultralytics' YOLOv8 weights — those are distributed under the AGPL-3.0
# license, which is not compatible with an Apache-2.0 requirement.
#
#   Project: https://github.com/Megvii-BaseDetection/YOLOX
#   License: https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE
#   Weights: https://github.com/Megvii-BaseDetection/YOLOX/releases/tag/0.1.1rc0
#
# Weights are downloaded lazily on first use (like the previous backend
# did) and cached under YOLOX_WEIGHTS_DIR so repeat runs don't re-download.
YOLOX_MODEL_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx"
YOLOX_WEIGHTS_DIR = Path(__file__).resolve().parent.parent / ".cache" / "yolox"
YOLOX_INPUT_SIZE = 416  # native export resolution for YOLOX-Nano
YOLOX_SCORE_THRESHOLD = 0.35
YOLOX_NMS_THRESHOLD = 0.45
YOLOX_PERSON_CLASS_ID = 0  # "person" is COCO class 0 in YOLOX's class ordering
