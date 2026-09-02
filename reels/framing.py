"""Stage 3b - decide how each scene should be framed, then build a crop path.

Three framing widths, from tightest to safest:

    track   full 9:16 crop that fills the screen   (single clear subject)
    wide    gentle ~4:5 crop, letterboxed          (groups, action, context matters)
    fit     no horizontal crop at all              (screens, text, uncertainty)

How a scene is planned:

  1. pick a starting mode from what the scene actually contains
  2. work out, per sampled frame, the range of crop centres that keeps every
     important thing (faces, bodies, objects, text/UI) fully inside the frame
  3. if that range is empty - the content is simply wider than the crop - widen
     the mode and start again; never shave off content to tidy the composition
  4. glide the camera towards the subject with a deadzone, easing and a speed
     limit, clamped every step into the safe range from (2)
  5. if the clamp had to fight the speed limit for much of the scene, the
     subject is moving faster than a calm camera can follow, so widen instead

Scene boundaries are hard cuts: the path resets rather than panning across one.
"""
from __future__ import annotations

import numpy as np

from .util import log

MODE_BY_KIND = {
    "talking_head": "track",
    "multi_person": "track",
    "wide_subject": "wide",
    "action": "wide",
    "screen_ui": "fit",
    "static_wide": "fit",
}

DOWNGRADE = {"track": "wide", "wide": "fit", "fit": "fit"}

# How far past the speed limit the safety clamp may drag the camera before we
# decide the shot is too busy to crop tight.
PAN_OVERSHOOT_TOLERANCE = 2.5
PAN_OVERSHOOT_MAX_FRACTION = 0.12


def _mode_crop_width(mode: str, src_w: int, src_h: int, out_w: int, out_h: int,
                     wide_aspect: float) -> float:
    if mode == "track":
        width = src_h * (out_w / out_h)
    elif mode == "wide":
        width = src_h * wide_aspect
    else:
        width = src_w
    return float(min(src_w, max(16.0, width)))


def _important_spans(sample: dict, kind: str, edge_margin: float = 0.05,
                     object_max_width: float = 0.5) -> list[tuple[float, float]]:
    """Horizontal spans (x0, x1) in 0-1 that must stay inside the frame.

    Faces centred in the outer sliver of the source frame are skipped. Someone
    the original framing already cuts in half at the edge is a bystander, not
    the subject, and letting one of them pin the crop open costs the whole reel.

    A wide object box is skipped too. The object detector happily returns a car
    or a motorcycle filling the entire frame, and treating that as untouchable
    makes every crop impossible - measured on a driving vlog, boxes labelled
    "car" and "motorcycle" spanned 99-100% of the width and forced every scene
    to full frame. Big objects are scenery: they still say where to point the
    camera, they just do not have to survive intact.
    """
    spans = []
    for f in sample.get("faces", []):
        centre = f["x"] + f["w"] / 2
        if edge_margin > 0 and (centre < edge_margin or centre > 1.0 - edge_margin):
            continue
        spans.append((f["x"], f["x"] + f["w"]))
    for o in sample.get("objects", []):
        if o["w"] > object_max_width:
            continue
        spans.append((o["x"], o["x"] + o["w"]))
    # Text only counts as untouchable where it is the point of the shot.
    if kind in ("screen_ui", "static_wide", "wide_subject"):
        for b in sample.get("text_boxes", []):
            spans.append((b["x"], b["x"] + b["w"]))
    return spans


def _feasible_interval(spans, crop_w: float, src_w: int, margin: float):
    """Range of crop centres (px) that keeps every span inside, or None."""
    lo_bound, hi_bound = crop_w / 2, max(crop_w / 2, src_w - crop_w / 2)
    if not spans:
        return lo_bound, hi_bound

    # A detector box can hang a few pixels off the edge of the frame; clamping
    # keeps the geometry honest, since nothing outside the frame can be shown
    # or lost.
    x0 = max(0.0, min(s[0] for s in spans) * src_w)
    x1 = min(float(src_w), max(s[1] for s in spans) * src_w)
    if (x1 - x0) > crop_w:
        return None                      # content is physically too wide

    # Breathing room is a nicety, not a requirement. Ask for the margin first;
    # if honouring it leaves no legal crop position, drop it and try again.
    # Without this a face near the frame edge makes even a full-width crop
    # "impossible", because the margin demands a centre the crop cannot reach.
    for pad in (min(crop_w * margin, max(0.0, (crop_w - (x1 - x0)) / 2)), 0.0):
        lo = max(lo_bound, x1 + pad - crop_w / 2)
        hi = min(hi_bound, x0 - pad + crop_w / 2)
        if lo <= hi:
            return lo, hi
    return None


def _target_center(sample: dict, kind: str) -> float | None:
    """Where the frame wants to be centred, in 0-1 of source width."""
    faces = sample.get("faces", [])
    if faces:
        if kind == "talking_head" and len(faces) == 1:
            face = faces[0]
            return face["x"] + face["w"] / 2
        if kind == "talking_head":
            # Several faces but one speaker: follow the dominant one.
            face = max(faces, key=lambda f: f["w"] * f["h"] * f["score"])
            return face["x"] + face["w"] / 2
        left = min(f["x"] for f in faces)
        right = max(f["x"] + f["w"] for f in faces)
        return (left + right) / 2

    objects = sample.get("objects", [])
    if objects:
        left = min(o["x"] for o in objects)
        right = max(o["x"] + o["w"] for o in objects)
        return (left + right) / 2

    sal = sample.get("saliency")
    if sal and sal.get("cover", 0) > 0.01:
        return sal["cx"]
    return None


def _build_path(samples, kind, crop_w, src_w, fcfg, margin, edge_margin=0.05,
                object_max_width=0.5):
    """Return (times, centers_px, feasible, overshoot_fraction).

    `feasible` is False when some frame simply cannot contain its content at
    this crop width - the caller should widen the mode.
    """
    half = crop_w / 2
    hard_lo, hard_hi = half, max(half, src_w - half)
    deadzone = fcfg.get("deadzone_ratio", 0.06) * src_w
    alpha = fcfg.get("smooth_alpha", 0.12)
    max_pan = fcfg.get("max_pan_px_per_sec", 160)

    intervals = []
    for s in samples:
        iv = _feasible_interval(
            _important_spans(s, kind, edge_margin, object_max_width),
            crop_w, src_w, margin)
        if iv is None:
            return None, None, False, 1.0
        intervals.append(iv)

    raw = []
    for s in samples:
        c = _target_center(s, kind)
        raw.append(None if c is None else float(np.clip(c * src_w, hard_lo, hard_hi)))

    known = [r for r in raw if r is not None]
    if known:
        filled, last = [], known[0]
        for r in raw:
            if r is not None:
                last = r
            filled.append(last)
    else:
        centre = float(np.clip(src_w / 2, hard_lo, hard_hi))
        filled = [centre] * len(samples)

    # Start already inside the safe range so the scene never opens on a clipped frame.
    current = float(np.clip(filled[0], *intervals[0]))
    times, path = [], []
    prev_t = samples[0]["t"] if samples else 0.0
    forced = 0

    for s, target, (lo, hi) in zip(samples, filled, intervals):
        dt = max(1e-3, s["t"] - prev_t)
        step_limit = max_pan * dt
        before = current

        if abs(target - current) > deadzone:
            desired = current + (target - current) * alpha
            step = float(np.clip(desired - current, -step_limit, step_limit))
            current += step

        # Safety wins over smoothness: never leave the feasible window.
        clamped = float(np.clip(current, lo, hi))
        if abs(clamped - before) > step_limit * PAN_OVERSHOOT_TOLERANCE:
            forced += 1
        current = clamped

        times.append(s["t"])
        path.append(current)
        prev_t = s["t"]

    overshoot = forced / len(samples) if samples else 0.0
    return times, path, True, overshoot


def _vertical_spans(sample: dict, kind: str, edge_margin: float):
    """Vertical spans (y0, y1) in 0-1 that must stay inside the frame."""
    spans = []
    for f in sample.get("faces", []):
        centre = f["x"] + f["w"] / 2
        if edge_margin > 0 and (centre < edge_margin or centre > 1.0 - edge_margin):
            continue
        # A little headroom above the hairline, and chin/neck below.
        spans.append((max(0.0, f["y"] - f["h"] * 0.35),
                      min(1.0, f["y"] + f["h"] * 1.25)))
    for o in sample.get("objects", []):
        spans.append((o["y"], o["y"] + o["h"]))
    return spans


def _build_vertical(samples, kind, heights, src_h, fcfg, edge_margin):
    """Per-sample crop TOP (px) that keeps heads in frame as the crop narrows.

    A crop narrower than the output aspect needs cropping vertically too, and a
    blind centre crop takes the top off people's heads. This picks the vertical
    window per frame - biased so faces sit high, the way a camera operator
    frames them - and eases between frames so it never jitters.
    """
    bias = float(fcfg.get("head_bias", 0.42))
    alpha = float(fcfg.get("smooth_alpha", 0.12)) * 1.6
    max_pan = float(fcfg.get("max_pan_px_per_sec", 160)) * 0.6
    margin = float(fcfg.get("safety_margin", 0.06))

    tops, current = [], None
    prev_t = samples[0]["t"] if samples else 0.0
    for s, ch in zip(samples, heights):
        free = max(0.0, src_h - ch)
        spans = _vertical_spans(s, kind, edge_margin)
        if spans:
            y0 = min(a for a, _ in spans) * src_h
            y1 = max(b for _, b in spans) * src_h
            pad = min(ch * margin, max(0.0, (ch - (y1 - y0)) / 2))
            lo = max(0.0, min(free, y1 + pad - ch))
            hi = max(0.0, min(free, y0 - pad))
            if lo > hi:                        # taller than the crop: centre it
                lo = hi = float(np.clip((y0 + y1) / 2 - ch / 2, 0.0, free))
            target = float(np.clip((y0 + y1) / 2 - ch * bias, lo, hi))
        else:
            lo, hi = 0.0, free
            target = float(np.clip(src_h * 0.5 - ch * bias, lo, hi))

        if current is None:
            current = target
        else:
            dt = max(1e-3, s["t"] - prev_t)
            step = float(np.clip((target - current) * alpha,
                                 -max_pan * dt, max_pan * dt))
            current = float(np.clip(current + step, lo, hi))
        tops.append(current)
        prev_t = s["t"]
    return tops


def _tightest_safe_width(samples, kind, src_w, margin, edge_margin,
                         object_max_width=0.5) -> float:
    """Narrowest crop that still holds this scene's content in every frame.

    Punch-ins are allowed to push in only as far as this. It is derived from the
    same spans the safety check uses, so a zoom can never clip a face the plan
    just promised to keep.
    """
    need = 0.0
    for s in samples:
        spans = _important_spans(s, kind, edge_margin, object_max_width)
        if not spans:
            continue
        x0 = min(a for a, _ in spans) * src_w
        x1 = max(b for _, b in spans) * src_w
        need = max(need, (x1 - x0) / max(0.2, 1.0 - 2 * margin))
    return need


def plan(analysis: dict, cfg: dict, out_w: int, out_h: int) -> dict:
    """Return per-frame crop rects plus a per-scene decision log."""
    fcfg = cfg["framing"]
    src_w, src_h = analysis["width"], analysis["height"]
    fps = analysis["fps"] or 30.0
    duration = analysis["duration"]
    n_frames = max(1, int(round(duration * fps)))
    margin = fcfg.get("safety_margin", 0.06)
    edge_margin = fcfg.get("edge_margin", 0.05)
    object_max_width = float(fcfg.get("object_max_width", 0.5))
    wide_aspect = fcfg.get("wide_aspect", 0.8)

    mcfg = cfg.get("motion", {})
    punch_on = bool(mcfg.get("punch_enabled", True))
    punch_zoom = float(mcfg.get("punch_zoom", 0.84))
    push_rate = float(mcfg.get("push_per_second", 0.006))
    min_zoom = float(mcfg.get("min_zoom", 0.7))
    croppable_seen = 0

    all_x = np.zeros(n_frames, dtype=np.float32)
    all_w = np.zeros(n_frames, dtype=np.float32)
    all_y = np.zeros(n_frames, dtype=np.float32)
    all_h = np.full(n_frames, float(src_h), dtype=np.float32)
    decisions = []

    for scene in analysis["scenes"]:
        kind = scene["kind"]
        samples = scene["samples"] or []
        mode = MODE_BY_KIND.get(kind, fcfg.get("default_mode", "fit"))

        # A "subject" nobody can actually see is not worth cropping tight.
        reason = "fits"
        if mode == "track" and scene["stats"].get("face_ratio", 0) < 0.5:
            mode, reason = "wide", "subject visible in too few frames"

        crop_w = _mode_crop_width(mode, src_w, src_h, out_w, out_h, wide_aspect)
        times, path, feasible, overshoot = _build_path(
            samples, kind, crop_w, src_w, fcfg, margin, edge_margin,
                object_max_width)

        guard = 0
        while mode != "fit" and guard < 3:
            guard += 1
            if not feasible:
                reason = "content wider than a tight crop"
            elif overshoot > PAN_OVERSHOOT_MAX_FRACTION:
                reason = f"subject moves too fast to follow calmly ({overshoot:.0%})"
            else:
                break
            mode = DOWNGRADE[mode]
            crop_w = _mode_crop_width(mode, src_w, src_h, out_w, out_h, wide_aspect)
            times, path, feasible, overshoot = _build_path(
                samples, kind, crop_w, src_w, fcfg, margin, edge_margin,
                object_max_width)

        if not feasible or path is None:
            mode, reason = "fit", "no safe crop exists"
            crop_w = float(src_w)
            times = [s["t"] for s in samples] or [scene["start"]]
            path = [src_w / 2] * len(times)

        f0 = int(round(scene["start"] * fps))
        f1 = min(n_frames, max(f0 + 1, int(round(scene["end"] * fps))))
        frame_times = np.arange(f0, f1) / fps
        n_scene = max(1, len(frame_times))
        scene_dur = max(0.1, scene["end"] - scene["start"])

        # Punch-in: alternate scenes sit tighter, so one continuous shot reads
        # like it was cut between two cameras. Never tighter than the content
        # allows, so the zoom cannot clip what the safety check just protected.
        w_start = w_end = crop_w
        zoomed = False
        if punch_on and mode != "fit":
            floor_w = max(_tightest_safe_width(samples, kind, src_w, margin,
                                               edge_margin, object_max_width),
                          crop_w * min_zoom)
            if croppable_seen % 2 == 1:
                w_start = max(floor_w, crop_w * punch_zoom)
            w_end = max(floor_w, w_start * (1.0 - push_rate * scene_dur))
            zoomed = (w_start < crop_w - 1) or (w_end < w_start - 1)
            croppable_seen += 1

        # The camera path was solved at `crop_w`; solve it again at the tightest
        # width this scene will actually reach, so it holds throughout the zoom.
        tight = min(w_start, w_end)
        if zoomed and tight < crop_w - 1:
            t2, p2, feasible2, over2 = _build_path(
                samples, kind, tight, src_w, fcfg, margin, edge_margin,
                object_max_width)
            if feasible2 and over2 <= PAN_OVERSHOOT_MAX_FRACTION and p2:
                times, path = t2, p2
            else:
                w_start = w_end = crop_w
                zoomed = False

        widths = np.linspace(w_start, w_end, n_scene)
        if len(times) >= 2:
            centres = np.interp(frame_times, times, path)
        else:
            centres = np.full(n_scene, path[0] if path else src_w / 2)

        halves = widths / 2.0
        centres = np.clip(centres, halves, np.maximum(halves, src_w - halves))
        all_x[f0:f1] = (centres - halves).astype(np.float32)
        all_w[f0:f1] = widths.astype(np.float32)

        # Height that matches the output aspect at each width. Where that is
        # shorter than the source, the vertical window has to be chosen too.
        heights = np.minimum(src_h, widths * out_h / out_w)
        all_h[f0:f1] = heights.astype(np.float32)
        if float(heights.min()) < src_h - 1:
            samp_h = np.interp([s["t"] for s in samples],
                               frame_times, heights) if len(frame_times) > 1                 else np.full(len(samples), float(heights[0]))
            tops = _build_vertical(samples, kind, samp_h, src_h, fcfg, edge_margin)
            if len(samples) >= 2:
                all_y[f0:f1] = np.interp(frame_times,
                                         [s["t"] for s in samples],
                                         tops).astype(np.float32)
            else:
                all_y[f0:f1] = float(tops[0]) if tops else 0.0
            all_y[f0:f1] = np.clip(all_y[f0:f1], 0, np.maximum(0, src_h - heights))

        decisions.append({
            "start": scene["start"], "end": scene["end"], "kind": kind,
            "mode": mode, "crop_width": round(float(np.mean(widths)), 1),
            "crop_pct": round(100 * float(np.mean(widths)) / src_w, 1),
            "zoom": (f"{100*w_start/src_w:.0f}%->{100*w_end/src_w:.0f}%"
                     if zoomed else "none"),
            "reason": reason,
        })

    if n_frames and float(all_w[0]) == 0.0:
        all_w[all_w == 0] = float(src_w)

    counts: dict[str, int] = {}
    for d in decisions:
        counts[d["mode"]] = counts.get(d["mode"], 0) + 1
    log("framing", "modes: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))

    return {
        "fps": fps,
        "n_frames": n_frames,
        "src_w": src_w,
        "src_h": src_h,
        "crop_x": all_x.tolist(),
        "crop_w": all_w.tolist(),
        "crop_y": all_y.tolist(),
        "crop_h": all_h.tolist(),
        "decisions": decisions,
    }


def apply_shake(plan: dict, times: list[float], cfg: dict) -> dict:
    """Kick the crop for a moment at each impact, then let it settle.

    A decaying oscillation, not random jitter: random noise reads as a bad
    encode, while a damped wobble reads as the camera being hit. Amplitude is a
    fraction of the crop width, so it looks the same at every zoom level, and
    the result is clamped back inside the frame - a shake must never push the
    crop past an edge and expose a black bar.
    """
    mcfg = cfg.get("motion", {})
    if not mcfg.get("shake_enabled", True) or not times:
        return plan

    fps = float(plan.get("fps") or 30.0)
    amp_ratio = float(mcfg.get("shake_amplitude", 0.016))
    dur = float(mcfg.get("shake_seconds", 0.18))
    freq = float(mcfg.get("shake_hz", 22.0))
    src_w, src_h = plan["src_w"], plan["src_h"]

    x = np.asarray(plan["crop_x"], dtype=np.float32)
    w = np.asarray(plan["crop_w"], dtype=np.float32)
    y = np.asarray(plan["crop_y"], dtype=np.float32)
    h = np.asarray(plan["crop_h"], dtype=np.float32)
    n = len(x)
    span = max(1, int(round(dur * fps)))

    for t in times:
        f0 = int(round(t * fps))
        if f0 < 0 or f0 >= n:
            continue
        f1 = min(n, f0 + span)
        k = np.arange(f1 - f0) / fps
        decay = np.exp(-k / max(1e-3, dur * 0.42))
        wobble = np.sin(2 * np.pi * freq * k) * decay
        amp = w[f0:f1] * amp_ratio
        x[f0:f1] += (wobble * amp).astype(np.float32)
        y[f0:f1] += (np.roll(wobble, 3) * amp * 0.6).astype(np.float32)

    plan["crop_x"] = np.clip(x, 0, np.maximum(0, src_w - w)).tolist()
    plan["crop_y"] = np.clip(y, 0, np.maximum(0, src_h - h)).tolist()
    return plan
