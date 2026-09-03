#!/usr/bin/env python3
"""
Quick smoke test for the YOLO (YOLOX-Nano) person detector.

Runs the detector from pipeline/yolo_detection.py on a single image and
either shows you the raw detections as JSON, or (with --save) writes an
annotated copy of the image with bounding boxes drawn on it.

Usage:
    pip install -r requirements.txt
    python scripts/run_detector_on_image.py path/to/image.jpg
    python scripts/run_detector_on_image.py path/to/image.jpg --save out.jpg
    python scripts/run_detector_on_image.py path/to/image.jpg --mode heuristic

The first run downloads the ~3.6MB Apache-2.0 licensed YOLOX-Nano weights
(cached under .cache/yolox/ afterwards), so it needs network access once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Make the repo root importable when running this script directly, e.g.
# `python scripts/run_detector_on_image.py ...` from anywhere.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image", type=Path, help="Path to an input image (jpg/png/etc).")
    parser.add_argument(
        "--mode",
        choices=["yolo", "heuristic"],
        default="yolo",
        help="Which detector to run: 'yolo' (YOLOX-Nano, default) or 'heuristic' (classical CV pipeline).",
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="If set, write an annotated copy of the image to this path.",
    )
    args = parser.parse_args()

    if not args.image.exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 1

    frame = cv2.imread(str(args.image))
    if frame is None:
        print(f"error: could not read image (unsupported format?): {args.image}", file=sys.stderr)
        return 1

    print(f"Loaded {args.image} -> shape {frame.shape}")

    if args.mode == "yolo":
        from pipeline.yolo_detection import detect_people_yolo

        print("Running YOLOX-Nano (this may take a moment on first run while weights download)...")
        start = time.time()
        detections = detect_people_yolo(frame)
        elapsed = time.time() - start
    else:
        from pipeline.detection import detect_candidates
        from pipeline.preprocessing import clean_mask, enhance
        from pipeline.segmentation import BackgroundSegmenter

        # The heuristic detector is motion-based and needs a background model;
        # for a single still image we just run it once against its own frame
        # so you can sanity-check that the code path executes end to end.
        enhanced = enhance(frame)
        mask = BackgroundSegmenter().apply(enhanced)
        cleaned = clean_mask(mask)
        start = time.time()
        detections = detect_candidates(cleaned, view_mode_hint="mixed")
        elapsed = time.time() - start
        print(
            "Note: Heuristic CV mode detects *motion*, not people in a still image, "
            "so a single static frame will typically yield 0 detections. Use --mode yolo "
            "for a meaningful still-image test."
        )

    print(f"\n{len(detections)} detection(s) in {elapsed * 1000:.1f}ms:\n")
    payload = [
        {
            "bbox_xywh": det.bbox,
            "centroid": det.centroid,
            "score": round(det.person_score, 4),
            "source": det.source,
        }
        for det in detections
    ]
    print(json.dumps(payload, indent=2))

    if args.save:
        annotated = frame.copy()
        for det in detections:
            x, y, w, h = det.bbox
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(
                annotated,
                f"{det.person_score:.2f}",
                (x, max(14, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 200, 0),
                2,
            )
        args.save.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.save), annotated)
        print(f"\nAnnotated image written to {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
