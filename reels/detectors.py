"""Lazy-loaded vision detectors with graceful fallbacks.

Face detection preference:  MediaPipe BlazeFace  ->  OpenCV YuNet  ->  none
Object detection:           MediaPipe EfficientDet-Lite (optional)
Saliency:                   OpenCV spectral-residual (bundled with contrib)
"""
from __future__ import annotations

import numpy as np

from .util import ASSETS, log

MODELS = {
    "face_blaze": ASSETS / "blaze_face_short_range.tflite",
    "face_yunet": ASSETS / "face_detection_yunet_2023mar.onnx",
    "object_det": ASSETS / "efficientdet_lite0.tflite",
}


class FaceDetector:
    """Returns face boxes as (x, y, w, h, score) in pixel coords."""

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence
        self.backend = "none"
        self._impl = None
        self._mp = None
        self._yunet_size = None
        self._load()

    def _load(self) -> None:
        # YuNet first. MediaPipe ships only the "short range" BlazeFace model,
        # which is built for a face filling much of the frame; on a wide shot of
        # people sitting a few metres away it finds one face or none, while
        # YuNet finds them all at ~4% of frame width.
        if MODELS["face_yunet"].exists():
            try:
                import cv2

                self._impl = cv2.FaceDetectorYN.create(
                    str(MODELS["face_yunet"]), "", (320, 320),
                    score_threshold=self.min_confidence,
                    nms_threshold=0.3, top_k=50,
                )
                self.backend = "yunet"
                return
            except Exception as exc:
                log("detect", f"yunet face detector unavailable: {exc}")

        if MODELS["face_blaze"].exists():
            try:
                import mediapipe as mp
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision

                opts = vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(
                        model_asset_path=str(MODELS["face_blaze"])),
                    running_mode=vision.RunningMode.IMAGE,
                    min_detection_confidence=self.min_confidence,
                )
                self._impl = vision.FaceDetector.create_from_options(opts)
                self._mp = mp
                self.backend = "mediapipe"
                return
            except Exception as exc:
                log("detect", f"mediapipe face detector unavailable: {exc}")

        log("detect", "no face model found - framing will rely on saliency/motion only")

    def __call__(self, bgr):
        if self._impl is None:
            return []
        import cv2

        h, w = bgr.shape[:2]
        if self.backend == "mediapipe":
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._impl.detect(image)
            out = []
            for det in result.detections:
                bb = det.bounding_box
                score = det.categories[0].score if det.categories else 1.0
                if score < self.min_confidence:
                    continue
                out.append((int(bb.origin_x), int(bb.origin_y),
                            int(bb.width), int(bb.height), float(score)))
            return out

        if self._yunet_size != (w, h):
            self._impl.setInputSize((w, h))
            self._yunet_size = (w, h)
        _, faces = self._impl.detect(bgr)
        if faces is None:
            return []
        return [(int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1]))
                for f in faces]


class ObjectDetector:
    """Optional COCO object detector; returns (x, y, w, h, score, label)."""

    WANTED = {
        "person", "cat", "dog", "horse", "bird", "car", "motorcycle", "bicycle",
        "bottle", "cup", "cell phone", "laptop", "book", "sports ball",
    }

    def __init__(self, score_threshold: float = 0.4):
        self.score_threshold = score_threshold
        self.available = False
        self._impl = None
        self._mp = None
        if not MODELS["object_det"].exists():
            return
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            opts = vision.ObjectDetectorOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=str(MODELS["object_det"])),
                running_mode=vision.RunningMode.IMAGE,
                score_threshold=score_threshold,
                max_results=8,
            )
            self._impl = vision.ObjectDetector.create_from_options(opts)
            self._mp = mp
            self.available = True
        except Exception as exc:
            log("detect", f"object detector unavailable: {exc}")

    def __call__(self, bgr):
        if not self.available:
            return []
        import cv2

        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._impl.detect(image)
        out = []
        for det in result.detections:
            cat = det.categories[0] if det.categories else None
            label = (cat.category_name or "") if cat else ""
            score = float(cat.score) if cat else 0.0
            if label not in self.WANTED or score < self.score_threshold:
                continue
            bb = det.bounding_box
            out.append((int(bb.origin_x), int(bb.origin_y),
                        int(bb.width), int(bb.height), score, label))
        return out


def mouth_motion(prev_gray, cur_gray, box) -> tuple[float, float]:
    """Change in the mouth band and in the eye/forehead band of `box`.

    Both are returned raw. The eye band is a baseline for how much the whole head
    moved between frames; speech shows up as mouth movement in excess of it. The
    caller combines them, because a person who moves their head a lot (driving,
    gesturing) needs the comparison averaged over time rather than per frame.
    `box` is (x, y, w, h) in pixels of the detection-resolution frame.
    """
    import cv2

    x, y, w, h = box
    if w < 24 or h < 24:
        return 0.0, 0.0
    H, W = cur_gray.shape[:2]

    def band(top_frac: float, bot_frac: float):
        x0 = max(0, int(x + w * 0.18))
        x1 = min(W, int(x + w * 0.82))
        y0 = max(0, int(y + h * top_frac))
        y1 = min(H, int(y + h * bot_frac))
        if x1 - x0 < 8 or y1 - y0 < 6:
            return None
        return (x0, y0, x1, y1)

    mouth = band(0.58, 1.0)
    upper = band(0.10, 0.48)
    if mouth is None or upper is None:
        return 0.0, 0.0

    def diff(region):
        x0, y0, x1, y1 = region
        a = cur_gray[y0:y1, x0:x1]
        b = prev_gray[y0:y1, x0:x1]
        if a.shape != b.shape or a.size == 0:
            return 0.0
        return float(np.mean(cv2.absdiff(a, b)))

    return diff(mouth), diff(upper)


_SALIENCY = None


def saliency_center(gray_small):
    """Spectral-residual saliency -> (cx, cy, coverage) normalised 0-1, or None."""
    global _SALIENCY
    import cv2

    if _SALIENCY is None:
        try:
            _SALIENCY = cv2.saliency.StaticSaliencySpectralResidual_create()
        except Exception:
            _SALIENCY = False
    if _SALIENCY is False:
        return None
    ok, smap = _SALIENCY.computeSaliency(gray_small)
    if not ok or smap is None:
        return None
    spread = float(np.ptp(smap))
    smap = (smap - smap.min()) / (spread + 1e-6)
    mask = (smap > 0.6).astype(np.uint8)
    if int(mask.sum()) < 12:
        return None
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    return (float(xs.mean() / w), float(ys.mean() / h),
            float(mask.sum() / mask.size))


def text_ui_score(gray) -> float:
    """0-1 estimate of "this frame is a screen recording / slide / text overlay".

    Screen content is dominated by axis-aligned edges (text baselines, UI chrome),
    unlike camera footage where gradient orientation is spread out.
    """
    import cv2

    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    thresh = max(24.0, float(np.percentile(mag, 92)))
    strong = mag > thresh
    n_strong = int(strong.sum())
    if n_strong < gray.size * 0.01:
        return 0.0
    ang = np.abs(np.degrees(np.arctan2(gy[strong], gx[strong])))
    ang = np.minimum(ang, 180.0 - ang)          # fold to 0..90
    axis_aligned = float(np.mean((ang < 12.0) | (ang > 78.0)))
    density = min(1.0, n_strong / (gray.size * 0.14))
    return float(axis_aligned * density)


def text_regions(gray):
    """Coarse boxes around dense text/UI clusters (used as no-crop zones)."""
    import cv2

    grad = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, binimg = cv2.threshold(grad, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(binimg, cv2.MORPH_CLOSE,
                              cv2.getStructuringElement(cv2.MORPH_RECT, (21, 5)))
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    min_area = gray.size * 0.0015
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < min_area or h < 6 or w < 3 * h:
            continue
        boxes.append((x, y, w, h))
    return boxes[:40]
