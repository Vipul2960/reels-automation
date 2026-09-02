"""One-time download of the small vision models the framing stage needs.

    python get_models.py

Everything lands in ./assets. Nothing else in the pipeline reaches the network
except yt-dlp (the video) and faster-whisper (its speech model, on first run).
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"

MODELS = [
    (
        "blaze_face_short_range.tflite",
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        "MediaPipe BlazeFace - face detection (required for face tracking)",
    ),
    (
        "efficientdet_lite0.tflite",
        "https://storage.googleapis.com/mediapipe-models/object_detector/"
        "efficientdet_lite0/float32/1/efficientdet_lite0.tflite",
        "MediaPipe EfficientDet-Lite0 - object detection (subject tracking)",
    ),
    (
        "face_detection_yunet_2023mar.onnx",
        "https://github.com/opencv/opencv_zoo/raw/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        "OpenCV YuNet - backup face detector",
    ),
]


def fetch(name: str, url: str, note: str) -> bool:
    target = ASSETS / name
    if target.exists() and target.stat().st_size > 1000:
        print(f"  [skip] {name}  ({target.stat().st_size / 1024:.0f} KB already present)")
        return True
    print(f"  [get ] {name}\n         {note}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "reels-automation/0.1"})
        with urllib.request.urlopen(req, timeout=90) as resp:
            data = resp.read()
        if len(data) < 1000:
            raise RuntimeError(f"suspiciously small response ({len(data)} bytes)")
        target.write_bytes(data)
        print(f"         saved {len(data) / 1024:.0f} KB")
        return True
    except Exception as exc:
        print(f"         FAILED: {exc}")
        return False


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    print(f"Downloading vision models into {ASSETS}\n")
    ok = [fetch(*m) for m in MODELS]
    print()
    if all(ok):
        print("All models ready.")
        return 0
    if ok[0] or ok[2]:
        print("A face model is available - framing will work. "
              "Object detection is optional.")
        return 0
    print("No face model downloaded. Framing will fall back to saliency only.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
