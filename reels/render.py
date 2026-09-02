"""Stages 4 & 7 - cut the clip, reframe every frame, burn captions, encode.

Frames are reshaped in Python (full control over per-frame crop and padding) and
streamed as raw BGR into ffmpeg, which burns the ASS captions, muxes the original
audio, and encodes with hardware acceleration when it is available.

Feeding a child process through a pipe has two ways to hang, and both are guarded
here:

  * if nobody reads ffmpeg's stderr, its pipe buffer fills, ffmpeg blocks on the
    write, stops draining stdin, and our next frame write blocks forever - so a
    daemon thread drains stderr continuously
  * if ffmpeg wedges for any other reason a watchdog kills it, turning a silent
    infinite hang into an error we can report and retry
"""
from __future__ import annotations

import subprocess
import threading
import time
from pathlib import Path

import numpy as np

from . import effects as fx
from .util import ffprobe, log, pick_encoder, progress, quality_flags, run, tool

STALL_TIMEOUT = 120.0        # seconds without progress before ffmpeg is killed


def ffpath(path: Path) -> str:
    """Escape a Windows path for use inside an ffmpeg filter argument."""
    text = str(path).replace("\\", "/")
    return text.replace(":", "\\:")


def cut_clip(source: Path, start: float, end: float, out_path: Path,
             cfg: dict) -> Path:
    """Frame-accurate working copy of one clip."""
    encoder, enc_flags = pick_encoder(cfg["output"].get("video_encoder", "auto"))
    duration = max(0.5, end - start)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Drop 50/60 fps sources to the output rate right here. Everything after
    # this - scene detection, face sampling, the frame loop, the encode - costs
    # per frame, so a 60 fps source doubles the entire pipeline for a frame rate
    # no vertical feed benefits from.
    max_fps = float(cfg["output"].get("max_fps", cfg["output"].get("fps", 30)))
    rate = []
    try:
        src_fps = ffprobe(source)["fps"]
        if src_fps > max_fps + 0.5:
            rate = ["-r", f"{max_fps:g}"]
            log("clip", f"source is {src_fps:.0f} fps - working at {max_fps:g}")
    except Exception:
        pass

    cmd = [
        tool("ffmpeg"), "-y", "-v", "error",
        "-ss", f"{start:.3f}", "-i", str(source), "-t", f"{duration:.3f}", *rate,
        "-map", "0:v:0", "-map", "0:a:0?",
        "-c:v", encoder, *enc_flags, *quality_flags(encoder, 20),
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        "-avoid_negative_ts", "make_zero",
        str(out_path),
    ]
    run(cmd)
    return out_path


def _background(frame, out_w: int, out_h: int, blur: int):
    """Cheap blurred cover-fill background: shrink hard, blur, scale back up."""
    import cv2

    small = cv2.resize(frame, (54, 96), interpolation=cv2.INTER_AREA)
    k = max(3, (blur // 4) * 2 + 1)
    small = cv2.GaussianBlur(small, (k, k), 0)
    big = cv2.resize(small, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return cv2.convertScaleAbs(big, alpha=0.55, beta=0)


def _encode(clip_path: Path, plan: dict, ass_path: Path | None, out_path: Path,
            cfg: dict, encoder: str, enc_flags: list[str],
            bed_path: Path | None = None,
            timeline: dict | None = None) -> int:
    """One encode attempt. Returns frames written; raises on failure."""
    import cv2

    ocfg = cfg["output"]
    fcfg = cfg["framing"]
    out_w = int(ocfg.get("width", 1080))
    out_h = int(ocfg.get("height", 1920))
    fps = float(plan.get("fps") or ocfg.get("fps", 30))
    blur = int(fcfg.get("blur_strength", 26))
    pad_style = fcfg.get("pad_style", "blur")
    # Fraction of the leftover space that goes ABOVE the video. 0.5 centres it;
    # lower lifts it, freeing the bottom of the frame for captions.
    pad_top = float(fcfg.get("pad_top_ratio", 0.32))
    stall_timeout = float(ocfg.get("stall_timeout", STALL_TIMEOUT))

    vf = []
    # Grade before the captions so the text keeps its exact colours.
    gcfg = cfg.get("grade", {})
    if gcfg.get("enabled", True):
        vf.append("eq=contrast={c}:saturation={s}:brightness={b}:gamma={g}".format(
            c=gcfg.get("contrast", 1.08), s=gcfg.get("saturation", 1.18),
            b=gcfg.get("brightness", 0.01), g=gcfg.get("gamma", 0.98)))
        sharp = float(gcfg.get("sharpen", 0.6))
        if sharp > 0:
            vf.append(f"unsharp=5:5:{sharp:.2f}:3:3:0.0")
    if ass_path is not None and ass_path.exists():
        vf.append(f"ass='{ffpath(ass_path)}'")
    vf.append("format=yuv420p")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = [
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{out_w}x{out_h}", "-r", f"{fps:.5f}", "-i", "pipe:0",
        "-i", str(clip_path),
    ]

    if bed_path is not None and bed_path.exists():
        # Effects duck the dialogue rather than sitting on top of it: the bed
        # drives a sidechain compressor on the source, then the two are summed.
        # `normalize=0` on amix, or the whole mix drops 6 dB the moment a second
        # input appears.
        scfg = cfg.get("sfx", {})
        gain = float(scfg.get("mix_gain", 0.55))
        duck = float(scfg.get("duck_ratio", 4.0))
        thresh = float(scfg.get("duck_threshold", 0.06))
        inputs += ["-i", str(bed_path)]
        graph = (
            f"[0:v]{','.join(vf)}[v];"
            f"[1:a]aformat=sample_fmts=fltp:sample_rates=48000:"
            f"channel_layouts=stereo[src];"
            f"[2:a]aformat=sample_fmts=fltp:sample_rates=48000:"
            f"channel_layouts=stereo,volume={gain:.3f},asplit=2[fx1][fx2];"
            f"[src][fx1]sidechaincompress=threshold={thresh}:ratio={duck}:"
            f"attack=5:release=260[duck];"
            f"[duck][fx2]amix=inputs=2:duration=first:normalize=0[aout]"
        )
        maps = ["-filter_complex", graph, "-map", "[v]", "-map", "[aout]"]
    else:
        maps = ["-map", "0:v:0", "-map", "1:a:0?", "-vf", ",".join(vf)]

    cmd = [
        tool("ffmpeg"), "-y", "-v", "error",
        *inputs, *maps,
        "-c:v", encoder, *enc_flags, *quality_flags(encoder, int(ocfg.get("quality", 23))),
        "-c:a", "aac", "-b:a", str(ocfg.get("audio_bitrate", "128k")),
        "-ar", "44100", "-ac", "2",
        "-movflags", "+faststart", "-shortest",
        str(out_path),
    ]

    crop_x = np.asarray(plan["crop_x"], dtype=np.float32)
    crop_w = np.asarray(plan["crop_w"], dtype=np.float32)
    crop_y = np.asarray(plan.get("crop_y") or [0.0] * len(crop_x), dtype=np.float32)
    crop_h = np.asarray(plan.get("crop_h") or [], dtype=np.float32)
    n_plan = len(crop_x)

    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {clip_path}")
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    # Drain stderr continuously or its buffer fills and everything deadlocks.
    stderr_parts: list[bytes] = []

    def drain():
        try:
            for chunk in iter(lambda: proc.stderr.read(4096), b""):
                stderr_parts.append(chunk)
        except (OSError, ValueError):
            pass

    drainer = threading.Thread(target=drain, daemon=True)
    drainer.start()

    # Watchdog: no forward progress for a long time means ffmpeg is wedged.
    finished = threading.Event()
    state = {"last": time.monotonic(), "killed": False}

    def watchdog():
        while not finished.wait(2.0):
            if time.monotonic() - state["last"] > stall_timeout:
                state["killed"] = True
                log("render", f"ffmpeg made no progress for {stall_timeout:.0f}s "
                              f"- killing it")
                try:
                    proc.kill()
                except OSError:
                    pass
                return

    threading.Thread(target=watchdog, daemon=True).start()

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    rng = np.random.default_rng(1234)          # same glitch every re-render
    n_fx = len(timeline["zoom"]) if timeline else 0
    written = 0
    early_exit = None

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if proc.poll() is not None:
                early_exit = proc.returncode
                break

            i = min(written, n_plan - 1) if n_plan else 0
            cw = int(round(float(crop_w[i]))) if n_plan else src_w
            cx = int(round(float(crop_x[i]))) if n_plan else 0
            cw = max(16, min(cw, src_w))
            cx = max(0, min(cx, src_w - cw))
            ch = int(round(float(crop_h[i]))) if len(crop_h) > i else src_h
            cy = int(round(float(crop_y[i]))) if len(crop_y) > i else 0
            ch = max(16, min(ch, src_h))
            cy = max(0, min(cy, src_h - ch))

            region = frame[cy:cy + ch, cx:cx + cw]
            fg_h = max(1, int(round(out_w * region.shape[0] / region.shape[1])))
            interp = cv2.INTER_AREA if region.shape[1] > out_w else cv2.INTER_LINEAR
            fg = cv2.resize(region, (out_w, fg_h), interpolation=interp)

            if fg_h >= out_h:
                top = (fg_h - out_h) // 2
                canvas[:] = fg[top:top + out_h]
            else:
                canvas[:] = (_background(frame, out_w, out_h, blur)
                             if pad_style == "blur" else 0)
                top = int(round((out_h - fg_h) * pad_top))
                top = max(0, min(top, out_h - fg_h))
                canvas[top:top + fg_h] = fg

            out = canvas
            if timeline is not None and written < n_fx:
                out = fx.apply_frame(canvas, written, timeline, cfg, rng)
            proc.stdin.write(np.ascontiguousarray(out).tobytes())
            written += 1
            state["last"] = time.monotonic()
            if n_plan and written % 5 == 0:
                progress("render", written, n_plan,
                         f"{written / fps:.0f}s of {n_plan / fps:.0f}s")
    except BrokenPipeError:
        early_exit = proc.poll()
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

    if n_plan:
        progress("render", n_plan, n_plan, f"{written / fps:.0f}s encoded")
    proc.wait()
    finished.set()
    drainer.join(timeout=5)

    stderr = b"".join(stderr_parts).decode("utf-8", "replace").strip()
    if state["killed"]:
        raise RuntimeError(
            f"{encoder} stalled after {written} frames and was killed"
            + (f"\n{stderr[-800:]}" if stderr else ""))
    if proc.returncode != 0:
        tail = "\n".join(stderr.splitlines()[-15:])
        raise RuntimeError(f"{encoder} failed (exit {proc.returncode}):\n{tail}")
    if early_exit is not None and written == 0:
        raise RuntimeError(f"{encoder} exited before any frame was written:\n{stderr}")

    if stderr:
        log("render", f"  ffmpeg said: {stderr.splitlines()[0][:110]}")
    return written


def render_reel(clip_path: Path, plan: dict, ass_path: Path | None,
                out_path: Path, cfg: dict, bed_path: Path | None = None,
                timeline: dict | None = None) -> Path:
    """Reframe + caption + encode one clip, falling back off the GPU encoder."""
    ocfg = cfg["output"]
    encoder, enc_flags = pick_encoder(ocfg.get("video_encoder", "auto"))

    try:
        written = _encode(clip_path, plan, ass_path, out_path, cfg,
                          encoder, enc_flags, bed_path, timeline)
    except RuntimeError as exc:
        if encoder == "libx264":
            raise
        log("render", f"{exc}".splitlines()[0])
        log("render", "retrying on the CPU encoder (libx264)")
        out_path.unlink(missing_ok=True)
        encoder, enc_flags = "libx264", ["-preset", "veryfast"]
        written = _encode(clip_path, plan, ass_path, out_path, cfg,
                          encoder, enc_flags, bed_path, timeline)

    size_mb = out_path.stat().st_size / 1e6
    log("render", f"{out_path.name} -> {written} frames, {size_mb:.1f} MB, {encoder}")
    return out_path
