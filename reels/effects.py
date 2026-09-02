"""Stage 7c - per-frame transition and impact effects.

Three effects, all applied to the finished 1080x1920 canvas inside the render
loop, driven by a precomputed strength-per-frame timeline:

    zoom blur   a few frames of radial smear over a cut, so the change of
                framing reads as a move rather than a jump
    flash       one or two brightened frames on the cut itself
    glitch      RGB channel separation plus a couple of displaced slices,
                fired on the impacts the sound design already hits

The timeline is built once per clip, so the loop only pays for a frame that
actually has an effect on it - which is a small minority of them.
"""
from __future__ import annotations

import numpy as np

from .util import log


def plan_effects(decisions: list[dict], impacts: list[float], fps: float,
                 n_frames: int, cfg: dict) -> dict:
    """Strength 0-1 per frame for each effect."""
    ecfg = cfg.get("effects", {})
    zoom = np.zeros(n_frames, dtype=np.float32)
    flash = np.zeros(n_frames, dtype=np.float32)
    glitch = np.zeros(n_frames, dtype=np.float32)

    if ecfg.get("zoom_blur_on_cuts", True) or ecfg.get("flash_on_cuts", True):
        blur_frames = max(1, int(round(float(ecfg.get("zoom_blur_seconds", 0.16)) * fps)))
        flash_frames = max(1, int(round(float(ecfg.get("flash_seconds", 0.05)) * fps)))
        for d in decisions[1:]:                    # the first frame is not a cut
            f0 = int(round(float(d["start"]) * fps))
            if f0 <= 0 or f0 >= n_frames:
                continue
            if ecfg.get("zoom_blur_on_cuts", True):
                span = min(blur_frames, n_frames - f0)
                # Strongest on the cut, gone a few frames later.
                zoom[f0:f0 + span] = np.maximum(
                    zoom[f0:f0 + span], np.linspace(1.0, 0.0, span, endpoint=False))
            if ecfg.get("flash_on_cuts", True):
                span = min(flash_frames, n_frames - f0)
                flash[f0:f0 + span] = np.maximum(
                    flash[f0:f0 + span], np.linspace(1.0, 0.0, span, endpoint=False))

    if ecfg.get("glitch_on_impacts", True):
        span = max(1, int(round(float(ecfg.get("glitch_seconds", 0.10)) * fps)))
        for t in impacts:
            f0 = int(round(t * fps))
            if f0 < 0 or f0 >= n_frames:
                continue
            end = min(n_frames, f0 + span)
            glitch[f0:end] = np.maximum(
                glitch[f0:end], np.linspace(1.0, 0.0, end - f0, endpoint=False))

    active = int(np.sum((zoom > 0) | (flash > 0) | (glitch > 0)))
    if active:
        log("effects", f"{active} of {n_frames} frames carry an effect "
                       f"({active / max(1, n_frames):.0%})")
    return {"zoom": zoom, "flash": flash, "glitch": glitch}


def _shift_x(plane, dx: int):
    """Slide horizontally, replicating the edge instead of wrapping.

    np.roll wraps, which drops a strip of one edge onto the other and shows up
    as a bright vertical bar down the side of the frame.
    """
    if dx == 0:
        return plane
    out = np.empty_like(plane)
    if dx > 0:
        out[:, dx:] = plane[:, :-dx]
        out[:, :dx] = plane[:, :1]
    else:
        out[:, :dx] = plane[:, -dx:]
        out[:, dx:] = plane[:, -1:]
    return out


def zoom_blur(img, strength: float, samples: int = 6, max_scale: float = 0.10,
              work_width: int = 540):
    """Radial smear outward from the centre - the classic whip-cut blur.

    Accumulated at reduced resolution: this is the most expensive effect in the
    loop and it is a blur, so the detail thrown away is detail the effect would
    have destroyed anyway. Roughly four times faster, visually identical.
    """
    import cv2

    if strength <= 0.01:
        return img
    h, w = img.shape[:2]
    scale_down = min(1.0, work_width / w)
    sw, sh = max(2, int(w * scale_down)), max(2, int(h * scale_down))
    small = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_AREA)

    acc = small.astype(np.float32)
    count = 1.0
    for i in range(1, samples):
        s = 1.0 + max_scale * strength * (i / (samples - 1))
        m = cv2.getRotationMatrix2D((sw / 2.0, sh / 2.0), 0.0, s)
        acc += cv2.warpAffine(small, m, (sw, sh), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        count += 1.0
    blurred = cv2.resize((acc / count).astype(np.uint8), (w, h),
                         interpolation=cv2.INTER_LINEAR)
    # Blend so a weak strength really is a weak effect.
    return cv2.addWeighted(img, 1.0 - strength, blurred, strength, 0.0)


def flash(img, strength: float, amount: float = 0.55):
    """Blend towards white for a frame or two."""
    import cv2

    if strength <= 0.01:
        return img
    return cv2.addWeighted(img, 1.0 - strength * amount,
                           np.full_like(img, 255), strength * amount, 0.0)


def glitch(img, strength: float, max_shift: int = 14, slices: int = 3,
           rng=None):
    """RGB separation plus a few displaced horizontal bands."""
    import cv2

    if strength <= 0.01:
        return img
    rng = rng or np.random
    h, w = img.shape[:2]
    shift = max(1, int(round(max_shift * strength)))

    b, g, r = cv2.split(img)
    out = cv2.merge([_shift_x(b, -shift), g, _shift_x(r, shift)])

    # Torn bands: a couple of thin strips slid sideways.
    for _ in range(max(1, int(slices * strength))):
        band_h = int(rng.integers(8, 46))
        y0 = int(rng.integers(0, max(1, h - band_h)))
        dx = int(rng.integers(-3 * shift, 3 * shift + 1))
        out[y0:y0 + band_h] = _shift_x(out[y0:y0 + band_h], dx)
    return out


def apply_frame(canvas, i: int, timeline: dict, cfg: dict, rng=None):
    """Run whichever effects are live on frame `i`."""
    ecfg = cfg.get("effects", {})
    z = float(timeline["zoom"][i])
    f = float(timeline["flash"][i])
    g = float(timeline["glitch"][i])
    if z <= 0.01 and f <= 0.01 and g <= 0.01:
        return canvas

    out = canvas
    if z > 0.01:
        out = zoom_blur(out, z * float(ecfg.get("zoom_blur_strength", 1.0)),
                        samples=int(ecfg.get("zoom_blur_samples", 6)),
                        max_scale=float(ecfg.get("zoom_blur_scale", 0.10)))
    if g > 0.01:
        out = glitch(out, g * float(ecfg.get("glitch_strength", 1.0)),
                     max_shift=int(ecfg.get("glitch_shift", 14)),
                     slices=int(ecfg.get("glitch_slices", 3)), rng=rng)
    if f > 0.01:
        out = flash(out, f, amount=float(ecfg.get("flash_amount", 0.55)))
    return out
