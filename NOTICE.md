# Third-Party Notices

This project (MIT licensed, see `LICENSE`) uses one third-party pretrained
model and re-implements a small amount of third-party code, both under the
Apache License 2.0.

## YOLOX-Nano (Megvii Inc.)

- Project: https://github.com/Megvii-BaseDetection/YOLOX
- License: Apache License 2.0 — https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE
- Copyright: Copyright (c) 2021-2022 Megvii Inc.
- What we use: the pretrained-on-COCO `yolox_nano.onnx` weights
  (downloaded at runtime from the project's GitHub release
  `0.1.1rc0`), loaded and run in `pipeline/yolo_detection.py`.
- What we changed: the pre-processing (letterbox resize), output decoding
  (`_decode_predictions`), and non-max suppression (`_nms`) in
  `pipeline/yolo_detection.py` are a numpy re-implementation of the logic in
  YOLOX's `demo/ONNXRuntime/onnx_inference.py` and
  `yolox/utils/demo_utils.py`, adapted to plug into this project's existing
  `Detection` dataclass and tracker instead of YOLOX's own visualization
  code. No YOLOX source files are vendored directly; only the small amount
  of logic described above was reused.

This detector replaces the project's previous YOLO backend, which loaded
Ultralytics YOLOv8 weights (`yolov8n.pt`) via the `ultralytics` package.
Ultralytics' pretrained YOLO weights and code are distributed under the
AGPL-3.0 license, which does not satisfy an Apache-2.0 requirement — hence
the switch to YOLOX-Nano.

A full copy of the Apache License 2.0 is available at
https://www.apache.org/licenses/LICENSE-2.0.
