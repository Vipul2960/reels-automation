"""Stage 3a - understand the footage before deciding how to frame it.

Splits a clip into scenes, samples frames, and classifies every scene as one of:

    talking_head   one dominant face, low motion      -> safe to crop tight
    multi_person   two or more faces                  -> must keep the group
    wide_subject   people present but small in frame  -> keep context
    screen_ui      slides / screen recording / text   -> never crop
    action         no faces, strong coherent motion   -> follow the subject
    static_wide    landscape, product, b-roll         -> keep the whole frame
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .detectors import (FaceDetector, ObjectDetector, mouth_motion,
                        saliency_center, text_regions, text_ui_score)
from .util import log, progress, read_json, write_json

SCENE_KINDS = ("talking_head", "multi_person", "wide_subject",
               "screen_ui", "action", "static_wide")


def detect_scenes(clip_path: Path, duration: float, cfg: dict) -> list[tuple[float, float]]:
    """Scene-boundary list as (start, end) seconds. Falls back to one scene."""
    acfg = cfg["analysis"]
    try:
        from scenedetect import ContentDetector, detect

        min_len = max(1, int(acfg.get("min_scene_seconds", 0.8) * 24))
        raw = detect(str(clip_path),
                     ContentDetector(threshold=float(acfg.get("scene_threshold", 27.0)),
                                     min_scene_len=min_len))
        scenes = [(float(a.get_seconds()), float(b.get_seconds())) for a, b in raw]
    except Exception as exc:
        log("analyze", f"scene detection failed ({exc}); treating clip as one scene")
        scenes = []

    if not scenes:
        return [(0.0, duration)]

    # Clamp and drop slivers.
    merged: list[list[float]] = []
    for start, end in scenes:
        end = min(end, duration)
        if end - start < acfg.get("min_scene_seconds", 0.8) and merged:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    merged[-1][1] = duration
    return [(a, b) for a, b in merged]


def _sample_frames(clip_path: Path, cfg: dict, face_det, obj_det) -> tuple[list[dict], dict]:
    """One sequential pass over the clip, analysing every Nth frame."""
    import cv2

    acfg = cfg["analysis"]
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {clip_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    step = max(1, int(round(fps / float(acfg.get("sample_fps", 4.0)))))

    n_frames_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    samples: list[dict] = []
    prev_small = None
    prev_det_gray = None
    idx = -1

    while True:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            continue

        t = idx / fps
        det_w = int(acfg.get("detect_width", 1280))
        scale = det_w / width if width > det_w else 1.0
        det_frame = cv2.resize(frame, (int(width * scale), int(height * scale))) if scale != 1.0 else frame
        dh, dw = det_frame.shape[:2]

        # Lip-motion measurement, off by default. It was tried as a way to spot
        # the active speaker in a group and checked against the footage: a hand
        # or a gesture near the mouth reads as speech, and on a 3-person clip it
        # picked the right person about as often as chance. Kept as groundwork -
        # doing this properly needs audio-visual correlation, not pixel diffs.
        faces = []
        measure_speech = bool(acfg.get("measure_speech", False))
        det_gray = (cv2.cvtColor(det_frame, cv2.COLOR_BGR2GRAY)
                    if measure_speech else None)
        for (x, y, w, h, score) in face_det(det_frame):
            # Clamp each edge separately. Capping x and w independently lets a
            # box near the border end up with x + w past 1.0, which downstream
            # reads as content wider than the frame itself.
            x0, x1 = max(0.0, x / dw), min(1.0, (x + w) / dw)
            y0, y1 = max(0.0, y / dh), min(1.0, (y + h) / dh)
            if x1 <= x0 or y1 <= y0:
                continue
            face = {
                "x": round(x0, 4), "y": round(y0, 4),
                "w": round(x1 - x0, 4), "h": round(y1 - y0, 4),
                "score": round(score, 3),
            }
            if (measure_speech and prev_det_gray is not None
                    and prev_det_gray.shape == det_gray.shape):
                mouth, head = mouth_motion(prev_det_gray, det_gray, (x, y, w, h))
                face["mouth"], face["head"] = round(mouth, 3), round(head, 3)
            faces.append(face)
        prev_det_gray = det_gray

        objects = []
        if obj_det is not None and obj_det.available and not faces:
            for (x, y, w, h, score, label) in obj_det(det_frame):
                ox0, ox1 = max(0.0, x / dw), min(1.0, (x + w) / dw)
                oy0, oy1 = max(0.0, y / dh), min(1.0, (y + h) / dh)
                if ox1 <= ox0 or oy1 <= oy0:
                    continue
                objects.append({
                    "x": round(ox0, 4), "y": round(oy0, 4),
                    "w": round(ox1 - ox0, 4), "h": round(oy1 - oy0, 4),
                    "score": round(score, 3), "label": label,
                })

        gray_mid = cv2.cvtColor(cv2.resize(frame, (480, max(1, int(480 * height / width)))),
                                cv2.COLOR_BGR2GRAY)
        tscore = text_ui_score(gray_mid)

        tboxes = []
        if tscore >= acfg.get("text_ui_score", 0.42) * 0.7:
            gh, gw = gray_mid.shape[:2]
            for (x, y, w, h) in text_regions(gray_mid):
                tboxes.append({"x": x / gw, "y": y / gh, "w": w / gw, "h": h / gh})

        small = cv2.cvtColor(cv2.resize(frame, (160, max(1, int(160 * height / width)))),
                             cv2.COLOR_BGR2GRAY)
        motion = 0.0
        if prev_small is not None:
            motion = float(np.mean(cv2.absdiff(small, prev_small)))
        prev_small = small

        sal = saliency_center(small)

        if n_frames_total:
            progress("analyze", idx + 1, n_frames_total,
                     f"{len(samples)} frames examined")

        samples.append({
            "t": round(t, 3),
            "faces": faces,
            "objects": objects,
            "text_score": round(tscore, 4),
            "text_boxes": tboxes,
            "motion": round(motion, 3),
            "saliency": ({"cx": round(sal[0], 4), "cy": round(sal[1], 4),
                          "cover": round(sal[2], 4)} if sal else None),
        })

    if n_frames_total:
        progress("analyze", n_frames_total, n_frames_total,
                 f"{len(samples)} frames examined")
    cap.release()
    info = {"width": width, "height": height, "fps": fps,
            "duration": (idx + 1) / fps if idx >= 0 else 0.0}
    return samples, info


def _median(values, default=0.0):
    vals = [v for v in values if v is not None]
    return float(np.median(vals)) if vals else default


def _classify(samples: list[dict], cfg: dict) -> tuple[str, dict]:
    """Decide the scene kind from its samples, plus the stats behind the call."""
    acfg = cfg["analysis"]
    if not samples:
        return "static_wide", {}

    face_counts = [len(s["faces"]) for s in samples]
    with_face = [c for c in face_counts if c]
    face_ratio = len(with_face) / len(samples)
    med_faces = _median(with_face, 0.0)

    biggest = []
    for s in samples:
        if s["faces"]:
            biggest.append(max(f["w"] * f["h"] for f in s["faces"]))
    med_face_area = _median(biggest, 0.0)

    med_text = _median([s["text_score"] for s in samples])
    med_motion = _median([s["motion"] for s in samples])
    sal_xs = [s["saliency"]["cx"] for s in samples if s["saliency"]]
    sal_spread = float(np.std(sal_xs)) if len(sal_xs) > 2 else 1.0

    stats = {
        "face_ratio": round(face_ratio, 3),
        "median_faces": round(med_faces, 2),
        "median_face_area": round(med_face_area, 5),
        "median_text_score": round(med_text, 4),
        "median_motion": round(med_motion, 3),
        "saliency_spread": round(sal_spread, 3),
    }

    # 1. Screen / slide content wins outright - cropping it destroys information.
    if med_text >= acfg.get("text_ui_score", 0.42) and face_ratio < 0.4:
        return "screen_ui", stats

    # 2. People on screen.
    if face_ratio >= 0.5:
        if med_faces >= 2:
            return "multi_person", stats
        if med_face_area >= acfg.get("face_big_ratio", 0.012):
            return "talking_head", stats
        return "wide_subject", stats

    # 3. No reliable faces - is something clearly moving?
    if med_motion >= acfg.get("motion_high", 6.0) and sal_spread < 0.22:
        return "action", stats

    # 4. Anything else: keep the frame wide. Uncertainty defaults to safe.
    return "static_wide", stats


def analyze(clip_path: Path, cfg: dict, cache_path: Path | None = None,
            force: bool = False) -> dict:
    """Full analysis for one cut clip."""
    if cache_path and cache_path.exists() and not force:
        data = read_json(cache_path)
        log("analyze", f"cached -> {len(data['scenes'])} scenes")
        return data

    face_det = FaceDetector(min_confidence=cfg["analysis"].get("face_min_confidence", 0.55))
    obj_det = ObjectDetector()

    samples, info = _sample_frames(clip_path, cfg, face_det, obj_det)
    scenes_bounds = detect_scenes(clip_path, info["duration"], cfg)

    scenes = []
    for start, end in scenes_bounds:
        in_scene = [s for s in samples if start <= s["t"] < end]
        if not in_scene and samples:
            nearest = min(samples, key=lambda s: abs(s["t"] - (start + end) / 2))
            in_scene = [nearest]
        kind, stats = _classify(in_scene, cfg)
        scenes.append({
            "start": round(start, 3),
            "end": round(end, 3),
            "kind": kind,
            "stats": stats,
            "samples": in_scene,
        })

    data = {**info, "backend": face_det.backend,
            "objects_enabled": bool(obj_det.available), "scenes": scenes}

    summary = ", ".join(f"{s['kind']}({s['end'] - s['start']:.1f}s)" for s in scenes[:6])
    log("analyze", f"{len(scenes)} scenes | faces={face_det.backend} | {summary}")

    if cache_path:
        write_json(cache_path, data)
    return data
