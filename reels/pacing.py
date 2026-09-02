"""Stage 4b - tighten the clip so it moves.

Dead air is what makes a clip feel slow. This finds every stretch that is both
quiet and wordless, drops it, and rebuilds the clip as a run of hard jump cuts.
A 45-second ramble becomes ~30 seconds that never sits still.

Two signals are combined, because neither is reliable alone:

  * audio loudness - works on any language, catches breaths and pauses that a
    transcript never records
  * transcript words - protects speech the loudness test would clip, and covers
    quiet delivery

A stretch is removed only when BOTH say nothing is happening there. Anything
else (music, laughter, engine noise, a beat) is kept.

The result is written as a real file, so everything downstream - scene
detection, framing, captions, encoding - runs on it unchanged. `TimeMap` carries
the source-to-output time mapping needed to retime the captions.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .util import ffprobe, log, run, tool


@dataclass
class TimeMap:
    """Maps a time in the original clip to its time in the tightened clip.

    Each kept range carries a playback speed, so a slowed beat stretches the
    output timeline and every later stage - captions, effects, sound - lands
    where it should.
    """

    keeps: list = field(default_factory=list)

    def __post_init__(self):
        self.keeps = [(k[0], k[1], k[2] if len(k) > 2 else 1.0) for k in self.keeps]

    def __call__(self, t: float) -> float | None:
        offset = 0.0
        for a, b, speed in self.keeps:
            if t < a:
                return offset                    # landed in a removed gap
            if t <= b:
                return offset + (t - a) / speed
            offset += (b - a) / speed
        return offset

    @property
    def kept_duration(self) -> float:
        return sum((b - a) / speed for a, b, speed in self.keeps)


def _loudness(path: Path, sr: int = 8000, hop: float = 0.02):
    """Per-hop RMS in dB relative to the clip's peak."""
    cmd = [tool("ffmpeg"), "-v", "error", "-i", str(path), "-f", "s16le",
           "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1", "pipe:1"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return None, hop
    x = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    n = max(1, int(sr * hop))
    frames = len(x) // n
    if frames < 2:
        return None, hop
    rms = np.sqrt((x[:frames * n].reshape(frames, n) ** 2).mean(axis=1))
    peak = float(rms.max())
    if peak <= 0:
        return None, hop
    return 20.0 * np.log10(np.maximum(rms / peak, 1e-6)), hop


def find_keeps(clip_path: Path, words: list[dict], clip_start: float,
               duration: float, cfg: dict) -> list[tuple[float, float]]:
    """Ranges of the clip (seconds, clip-relative) worth keeping."""
    pcfg = cfg.get("pacing", {})
    max_silence = float(pcfg.get("max_silence", 0.22))
    breath = float(pcfg.get("keep_breath", 0.06))
    min_seg = float(pcfg.get("min_segment", 0.20))
    max_removed = float(pcfg.get("max_removed_fraction", 0.45))
    min_range = float(pcfg.get("min_dynamic_range_db", 12.0))
    floor_margin = float(pcfg.get("floor_margin", 0.35))

    long_gap = float(pcfg.get("long_gap", 0.8))
    protect = float(pcfg.get("word_protect_max", 0.6))

    db, hop = _loudness(clip_path)
    if db is None:
        return [(0.0, duration)]
    n_hops = len(db)

    def hop_range(a: float, b: float) -> tuple[int, int]:
        return (max(0, int(a / hop)), min(n_hops, int(b / hop) + 1))

    # Speech mask, built from word START times only. YouTube's json3 captions
    # carry no real per-word end - captions.py fills `end` with the next word's
    # start so karaoke highlighting works - so a word's stated duration can be
    # an entire pause. Cap each protected span or the whole clip reads as speech
    # and nothing is ever cut.
    speech = np.zeros(n_hops, dtype=bool)
    for w in words:
        start = w["start"] - clip_start
        end = min(w["end"] - clip_start, start + protect)
        i0, i1 = hop_range(start - 0.05, end + 0.05)
        if i1 > i0:
            speech[i0:i1] = True

    cuttable = np.zeros(n_hops, dtype=bool)

    # Rule A - quiet AND wordless. Needs a real gap between this clip's noise
    # floor and its speech level; a fixed offset from the peak does not work,
    # because room tone, an engine or a music bed hold the quiet parts high.
    floor = float(np.percentile(db, 5))
    level = float(np.percentile(db, 70))
    dyn = level - floor
    if dyn >= min_range:
        quiet = db <= (floor + floor_margin * dyn)
        cuttable |= quiet & ~speech
        log("pacing", f"noise floor {floor:.0f} dB, speech {level:.0f} dB "
                      f"-> silence below {floor + floor_margin * dyn:.0f} dB")
    else:
        log("pacing", f"only {dyn:.0f} dB between noise floor and speech "
                      f"- loudness cannot separate silence here")

    # Rule B - a long stretch with no words at all, however noisy. This catches
    # dead air hiding under continuous background sound, where Rule A is blind.
    #
    # But "no words" is only as good as the transcript, and a weak caption track
    # drops real speech. So a wordless stretch must also be measurably quieter
    # than this clip's speech level before it is cut. If it is as loud as speech,
    # somebody is probably talking and the transcript simply missed it.
    quiet_margin = float(pcfg.get("gap_quiet_margin_db", 2.5))
    runs, rejected, i = [], 0, 0
    while i < n_hops:
        if speech[i]:
            i += 1
            continue
        j = i
        while j < n_hops and not speech[j]:
            j += 1
        if (j - i) * hop >= long_gap:
            if float(np.median(db[i:j])) < level - quiet_margin:
                runs.append((i, j))
            else:
                rejected += 1
        i = j
    for i0, i1 in runs:
        cuttable[i0:i1] = True
    if runs:
        total = sum((b - a) * hop for a, b in runs)
        log("pacing", f"{len(runs)} wordless stretch(es) over {long_gap:.1f}s "
                      f"= {total:.1f}s")
    if rejected:
        log("pacing", f"kept {rejected} wordless stretch(es) that are as loud as "
                      f"speech - transcript probably missed words there")

    loud = ~cuttable

    # Grow loud regions slightly so cuts do not clip word onsets.
    pad = max(1, int(breath / hop))
    padded = loud.copy()
    idx = np.flatnonzero(loud)
    for i in idx:
        padded[max(0, i - pad):min(len(padded), i + pad + 1)] = True

    # Walk the mask into keep ranges, allowing a short pause to survive intact.
    keeps: list[list[float]] = []
    in_run = False
    for i, on in enumerate(padded):
        t = i * hop
        if on and not in_run:
            keeps.append([t, t + hop])
            in_run = True
        elif on:
            keeps[-1][1] = t + hop
        elif in_run:
            in_run = False
    if not keeps:
        return [(0.0, duration)]

    # Re-join runs whose gap is short enough to feel like natural rhythm.
    merged = [keeps[0]]
    for a, b in keeps[1:]:
        if a - merged[-1][1] <= max_silence:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    out = [(max(0.0, a), min(duration, b)) for a, b in merged
           if (b - a) >= min_seg]
    if not out:
        return [(0.0, duration)]

    # Safety valve: if this would gut the clip, the detector is wrong about it.
    kept = sum(b - a for a, b in out)
    if kept < duration * (1.0 - max_removed):
        log("pacing", f"would drop {(1 - kept/duration):.0%} of the clip "
                      f"- too much, keeping it whole")
        return [(0.0, duration)]
    return out


def _atempo_chain(speed: float) -> list[float]:
    """atempo only accepts 0.5-100 per instance, so extreme rates get chained."""
    if abs(speed - 1.0) < 1e-3:
        return []
    factors, remaining = [], speed
    while remaining < 0.5:
        factors.append(0.5)
        remaining /= 0.5
    while remaining > 2.0:
        factors.append(2.0)
        remaining /= 2.0
    factors.append(remaining)
    return factors


def add_speed_ramp(clip_path: Path, keeps, words: list[dict], clip_start: float,
                   cfg: dict):
    """Slow one short pause down, for a beat before or after a hard-hit line.

    Only ever a pause with no speech in it. Time-stretching speech makes it
    sound drunk, so the ramp goes where nobody is talking - which is also where
    a human editor puts one.

    The pause is found in the AUDIO, not in the transcript. YouTube's captions
    give each word an end time equal to the next word's start, so consecutive
    words are always touching and the transcript contains no pauses at all -
    measured on a real clip, 0 of 111 word gaps were above 10 ms.
    """
    pcfg = cfg.get("pacing", {})
    if not pcfg.get("ramp_enabled", True) or not words:
        return keeps, None

    speed = float(pcfg.get("ramp_speed", 0.55))
    lo = float(pcfg.get("ramp_min_gap", 0.24))
    hi = float(pcfg.get("ramp_max_gap", 0.9))

    hits = [w["start"] - clip_start for w in words if w.get("emph")]
    if not hits:
        return keeps, None

    db, hop = _loudness(clip_path)
    if db is None:
        return keeps, None
    floor = float(np.percentile(db, 5))
    level = float(np.percentile(db, 70))
    if level - floor < float(pcfg.get("min_dynamic_range_db", 12.0)):
        log("pacing", "audio too compressed to find a pause for a speed ramp")
        return keeps, None
    quiet = db <= floor + float(pcfg.get("floor_margin", 0.35)) * (level - floor)

    # Every quiet run, as (start, end) seconds.
    runs, i, n = [], 0, len(quiet)
    while i < n:
        if not quiet[i]:
            i += 1
            continue
        j = i
        while j < n and quiet[j]:
            j += 1
        runs.append((i * hop, j * hop))
        i = j

    best = None
    for idx, seg in enumerate(keeps):
        a, b = seg[0], seg[1]
        for rs, re_ in runs:
            s0, s1 = max(a, rs), min(b, re_)
            if not (lo <= s1 - s0 <= hi):
                continue
            near = min((abs(s0 - h) for h in hits), default=1e9)
            if best is None or near < best[0]:
                best = (near, idx, s0, s1)

    if best is None or best[0] > float(pcfg.get("ramp_max_distance", 2.5)):
        return keeps, None

    _, idx, gs, ge = best
    a, b = keeps[idx][0], keeps[idx][1]
    rebuilt = []
    for j, k in enumerate(keeps):
        if j != idx:
            rebuilt.append((k[0], k[1], k[2] if len(k) > 2 else 1.0))
            continue
        if gs - a > 0.05:
            rebuilt.append((a, gs, 1.0))
        rebuilt.append((gs, ge, speed))
        if b - ge > 0.05:
            rebuilt.append((ge, b, 1.0))
    log("pacing", f"speed ramp: {ge - gs:.2f}s pause at {gs:.1f}s slowed to "
                  f"{speed:.2f}x")
    return rebuilt, (gs, ge, speed)


def tighten(clip_path: Path, out_path: Path, words: list[dict],
            clip_start: float, cfg: dict) -> tuple[Path, TimeMap]:
    """Write a jump-cut version of `clip_path`; returns it and its time map."""
    from .util import pick_encoder, quality_flags

    pcfg = cfg.get("pacing", {})
    info = ffprobe(clip_path)
    duration = info["duration"]

    if not pcfg.get("enabled", True):
        return clip_path, TimeMap([(0.0, duration)])

    keeps = find_keeps(clip_path, words, clip_start, duration, cfg)
    segments, ramp = add_speed_ramp(clip_path, keeps, words, clip_start, cfg)
    tmap = TimeMap(segments)

    untouched = (len(segments) == 1 and ramp is None
                 and segments[0][1] - segments[0][0] >= duration - 0.05)
    if untouched:
        log("pacing", "nothing worth cutting")
        return clip_path, tmap

    encoder, enc_flags = pick_encoder(cfg["output"].get("video_encoder", "auto"))
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # One trim per kept range, each with its own playback speed, concatenated.
    # A plain select filter cannot vary speed, and re-encoding twice to add the
    # ramp would cost another generation of quality.
    parts, pairs = [], []
    for i, (a, b, speed) in enumerate(tmap.keeps):
        parts.append(f"[0:v]trim=start={a:.3f}:end={b:.3f},"
                     f"setpts=(PTS-STARTPTS)/{speed:.4f}[v{i}]")
        chain = "".join(f",atempo={f:.4f}" for f in _atempo_chain(speed))
        parts.append(f"[0:a]atrim=start={a:.3f}:end={b:.3f},"
                     f"asetpts=PTS-STARTPTS{chain}[a{i}]")
        pairs.append(f"[v{i}][a{i}]")
    graph = (";".join(parts) + ";" + "".join(pairs)
             + f"concat=n={len(tmap.keeps)}:v=1:a=1[v][a]")

    run([
        tool("ffmpeg"), "-y", "-v", "error", "-i", str(clip_path),
        "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
        "-c:v", encoder, *enc_flags, *quality_flags(encoder, 20),
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", "-ar", "48000",
        str(out_path),
    ])

    kept = tmap.kept_duration
    log("pacing", f"{len(tmap.keeps)} segments: {duration:.1f}s -> {kept:.1f}s "
                  f"({(1 - kept/duration):+.0%} change)")
    return out_path, tmap


def retime_words(words: list[dict], tmap: TimeMap, clip_start: float) -> list[dict]:
    """Shift caption words onto the tightened timeline, dropping cut-out ones."""
    out = []
    for w in words:
        a = tmap(w["start"] - clip_start)
        b = tmap(w["end"] - clip_start)
        if a is None or b is None or b - a < 0.02:
            continue                             # this word was cut away
        out.append({**w, "start": round(a, 3), "end": round(b, 3)})
    return out
