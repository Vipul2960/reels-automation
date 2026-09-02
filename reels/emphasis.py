"""Stage 5b - work out which words the speaker actually hits hard.

Picking emphasis words from a vocabulary list needs a language model or a word
list per language, and this pipeline handles Gujarati, Hindi and whatever else
the source happens to be. Loudness does not care: the word somebody raises their
voice on is the word to make big and yellow, in any language.

Per word we take its loudness above the clip's own speaking level, and mark the
top slice. Digits are always marked - numbers are what viewers stop for.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np

from .util import log, tool

DIGITS = re.compile(r"\d")


def _envelope(path: Path, sr: int = 8000, hop: float = 0.02):
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
    return rms, hop


def mark_emphasis(clip_path: Path, words: list[dict], cfg: dict) -> list[dict]:
    """Return `words` with an `emph` flag on the ones that land hardest."""
    scfg = cfg.get("subtitles", {})
    if not scfg.get("emphasis_enabled", True) or not words:
        return words

    top_fraction = float(scfg.get("emphasis_fraction", 0.18))
    rms, hop = _envelope(clip_path)
    out = [dict(w) for w in words]

    if rms is None:
        for w in out:
            w["emph"] = bool(DIGITS.search(w["word"]))
        return out

    loud = []
    for w in out:
        i0 = max(0, int(w["start"] / hop))
        i1 = min(len(rms), max(i0 + 1, int(w["end"] / hop)))
        span = rms[i0:i1]
        loud.append(float(span.max()) if len(span) else 0.0)

    loud_arr = np.asarray(loud)
    speaking = loud_arr[loud_arr > 0]
    if len(speaking) < 4:
        cutoff = np.inf
    else:
        cutoff = float(np.quantile(speaking, 1.0 - top_fraction))

    n_marked = 0
    for w, level in zip(out, loud):
        w["emph"] = bool(level >= cutoff or DIGITS.search(w["word"]))
        n_marked += w["emph"]

    log("emphasis", f"{n_marked}/{len(out)} words marked as emphasis "
                    f"(loudest {top_fraction:.0%} + any number)")
    return out
