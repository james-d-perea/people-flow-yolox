from __future__ import annotations

import urllib.request
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from .config import (
    YOLOX_INPUT_SIZE,
    YOLOX_MODEL_URL,
    YOLOX_NMS_THRESHOLD,
    YOLOX_PERSON_CLASS_ID,
    YOLOX_SCORE_THRESHOLD,
    YOLOX_WEIGHTS_DIR,
)
from .models import Detection

# ---------------------------------------------------------------------------
# YOLO person detector backend: YOLOX-Nano.
#
# YOLOX-Nano is a pretrained-on-COCO, anchor-free YOLO detector released by
# Megvii under the Apache License 2.0 (code and weights both):
#   https://github.com/Megvii-BaseDetection/YOLOX
#
# We run it here through onnxruntime against the official exported ONNX
# weights, rather than depending on the `ultralytics` package, whose
# pretrained YOLO weights (e.g. yolov8n.pt) are distributed under the
# AGPL-3.0 license. The pre-/post-processing helpers below (`_letterbox`,
# `_decode_predictions`, `_nms`) are small numpy re-implementations of the
# reference ONNXRuntime demo shipped in the YOLOX repository
# (demo/ONNXRuntime/onnx_inference.py and yolox/utils/demo_utils.py),
# reused here under the same Apache-2.0 license.
# ---------------------------------------------------------------------------


def _weights_path() -> Path:
    YOLOX_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    return YOLOX_WEIGHTS_DIR / Path(YOLOX_MODEL_URL).name


def _download_weights(destination: Path) -> None:
    tmp_path = destination.with_suffix(".part")
    urllib.request.urlretrieve(YOLOX_MODEL_URL, tmp_path)
    tmp_path.replace(destination)


@lru_cache(maxsize=1)
def get_yolox_session():
    """
    Load the YOLOX ONNX model once and reuse it between frames.
    The first call can be slow because the (Apache-2.0 licensed) pretrained
    weights may need to be downloaded and cached to disk.
    """
    try:
        import onnxruntime
    except ImportError as exc:
        raise RuntimeError(
            "YOLO mode requires the 'onnxruntime' package. "
            "Add 'onnxruntime>=1.17' to requirements.txt."
        ) from exc

    weights_path = _weights_path()
    if not weights_path.exists():
        _download_weights(weights_path)

    return onnxruntime.InferenceSession(str(weights_path), providers=["CPUExecutionProvider"])


def _letterbox(frame: np.ndarray, input_size: tuple[int, int]) -> tuple[np.ndarray, float]:
    padded = np.full((input_size[0], input_size[1], 3), 114, dtype=np.uint8)
    ratio = min(input_size[0] / frame.shape[0], input_size[1] / frame.shape[1])
    new_w, new_h = int(frame.shape[1] * ratio), int(frame.shape[0] * ratio)
    resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded[:new_h, :new_w] = resized
    blob = padded.transpose(2, 0, 1).astype(np.float32)
    return np.ascontiguousarray(blob), ratio


def _decode_predictions(raw_output: np.ndarray, input_size: tuple[int, int]) -> np.ndarray:
    """Turn YOLOX's anchor-free grid-relative head output into image-space boxes."""
    grids = []
    strides_expanded = []
    strides = (8, 16, 32)

    for stride in strides:
        grid_h, grid_w = input_size[0] // stride, input_size[1] // stride
        xv, yv = np.meshgrid(np.arange(grid_w), np.arange(grid_h))
        grid = np.stack((xv, yv), axis=2).reshape(1, -1, 2)
        grids.append(grid)
        strides_expanded.append(np.full((1, grid.shape[1], 1), stride))

    grids = np.concatenate(grids, axis=1)
    strides_expanded = np.concatenate(strides_expanded, axis=1)

    raw_output[..., :2] = (raw_output[..., :2] + grids) * strides_expanded
    raw_output[..., 2:4] = np.exp(raw_output[..., 2:4]) * strides_expanded
    return raw_output


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        overlap = (w * h) / (areas[i] + areas[order[1:]] - w * h)
        order = order[np.where(overlap <= iou_threshold)[0] + 1]

    return keep


def detect_people_yolo(frame: np.ndarray) -> list[Detection]:
    """
    Detect people with YOLOX-Nano, a pretrained-on-COCO, Apache-2.0 licensed
    YOLO detector (see module docstring for provenance).

    Input frame is expected to be a BGR OpenCV image at processing resolution.
    Returns the same Detection objects as the heuristic detector, so the existing
    CentroidTracker and flow decision logic can be reused.
    """
    session = get_yolox_session()
    input_size = (YOLOX_INPUT_SIZE, YOLOX_INPUT_SIZE)

    blob, ratio = _letterbox(frame, input_size)
    input_name = session.get_inputs()[0].name
    raw_output = session.run(None, {input_name: blob[None, :, :, :]})[0]

    predictions = _decode_predictions(raw_output, input_size)[0]
    boxes = predictions[:, :4]
    scores = predictions[:, 4:5] * predictions[:, 5:]

    boxes_xyxy = np.empty_like(boxes)
    boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    boxes_xyxy /= ratio

    person_scores = scores[:, YOLOX_PERSON_CLASS_ID]
    keep_mask = person_scores > YOLOX_SCORE_THRESHOLD
    if not np.any(keep_mask):
        return []

    person_boxes = boxes_xyxy[keep_mask]
    person_scores = person_scores[keep_mask]
    keep_indices = _nms(person_boxes, person_scores, YOLOX_NMS_THRESHOLD)
    if not keep_indices:
        return []

    height, width = frame.shape[:2]
    detections: list[Detection] = []
    for index in keep_indices:
        x1, y1, x2, y2 = person_boxes[index]
        x1 = max(0, min(int(round(x1)), width - 1))
        y1 = max(0, min(int(round(y1)), height - 1))
        x2 = max(0, min(int(round(x2)), width - 1))
        y2 = max(0, min(int(round(y2)), height - 1))

        bbox_width = max(0, x2 - x1)
        bbox_height = max(0, y2 - y1)
        if bbox_width < 4 or bbox_height < 4:
            continue

        score = float(person_scores[index])
        detections.append(
            Detection(
                bbox=(x1, y1, bbox_width, bbox_height),
                centroid=(int(x1 + bbox_width / 2), int(y1 + bbox_height / 2)),
                area=float(bbox_width * bbox_height),
                mask_coverage=1.0,
                head_score=score,
                body_score=score,
                source="yolo",
            )
        )

    return detections
