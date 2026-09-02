"""Stage 3 - choose which moments become reels.

Two strategies:
  * AI      transcript goes to Claude, which returns ranked moments with reasons
  * offline speech-rate + audio-energy + cue-word scoring over sliding windows

The offline path always runs first so the AI has a sane fallback and so the
pipeline works with no API key at all.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from .transcribe import flat_words, snap_to_words
from .util import ROOT, log, tool

CUE_WORDS = {
    "secret", "mistake", "never", "always", "biggest", "worst", "best", "reason",
    "because", "actually", "truth", "problem", "wrong", "right", "important",
    "money", "free", "first", "finally", "realise", "realize", "realised",
    "realized", "learned", "story", "crazy", "insane", "shocking", "nobody",
    "everyone", "trick", "hack", "step", "warning", "danger", "proof",
}


def audio_envelope(path: Path, sr: int = 8000, hop_seconds: float = 0.5):
    """RMS loudness per `hop_seconds` bucket, normalised 0-1."""
    cmd = [tool("ffmpeg"), "-v", "error", "-i", str(path), "-f", "s16le",
           "-acodec", "pcm_s16le", "-ar", str(sr), "-ac", "1", "pipe:1"]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0 or not proc.stdout:
        return np.zeros(1, dtype=np.float32), hop_seconds
    samples = np.frombuffer(proc.stdout, dtype=np.int16).astype(np.float32) / 32768.0
    hop = max(1, int(sr * hop_seconds))
    n = len(samples) // hop
    if n < 1:
        return np.zeros(1, dtype=np.float32), hop_seconds
    rms = np.sqrt((samples[:n * hop].reshape(n, hop) ** 2).mean(axis=1))
    peak = float(rms.max()) or 1.0
    return (rms / peak).astype(np.float32), hop_seconds


def _sentences(transcript: dict) -> list[dict]:
    """Sentence-ish units with timings, built from word timestamps."""
    out, buf = [], []
    for seg in transcript["segments"]:
        for w in seg["words"]:
            buf.append(w)
            if w["word"][-1:] in ".!?" or len(buf) >= 40:
                out.append({"start": buf[0]["start"], "end": buf[-1]["end"],
                            "text": " ".join(x["word"] for x in buf)})
                buf = []
    if buf:
        out.append({"start": buf[0]["start"], "end": buf[-1]["end"],
                    "text": " ".join(x["word"] for x in buf)})
    return out


# --------------------------------------------------------------------------- #
# offline scoring
# --------------------------------------------------------------------------- #

def select_heuristic(transcript: dict, meta: dict, cfg: dict) -> list[dict]:
    scfg = cfg["select"]
    want = int(scfg.get("clips", 5))
    lo = float(scfg.get("min_seconds", 20))
    hi = float(scfg.get("max_seconds", 60))

    words = flat_words(transcript)
    if not words:
        return []
    sents = _sentences(transcript)
    env, hop = audio_envelope(Path(meta["path"]))

    def energy(a: float, b: float) -> tuple[float, float]:
        i0, i1 = int(a / hop), max(int(a / hop) + 1, int(b / hop))
        window = env[i0:i1]
        if not len(window):
            return 0.0, 0.0
        return float(window.mean()), float(window.std())

    # Grow each candidate towards a target length instead of stopping the moment
    # it clears the minimum - otherwise every clip comes out near `min_seconds`
    # however high `max_seconds` is set, and the reels cover little of the video.
    target = float(scfg.get("target_seconds", (lo + hi) / 2))
    target = max(lo, min(target, hi))

    candidates = []
    for i, s in enumerate(sents):
        start = s["start"]
        end = start
        for j in range(i, len(sents)):
            nxt = sents[j]["end"]
            if nxt - start > hi:
                break
            end = nxt
            if end - start >= target:
                break
        if end - start < lo * 0.8 or end - start > hi * 1.5:
            continue
        end = min(end, start + hi)

        span = [w for w in words if w["start"] >= start and w["end"] <= end]
        if len(span) < 12:
            continue
        text = " ".join(w["word"] for w in span).lower()
        duration = max(1.0, end - start)

        wps = len(span) / duration
        mean_e, var_e = energy(start, end)
        cues = sum(1 for token in re.findall(r"[a-z']+", text) if token in CUE_WORDS)
        questions = text.count("?")
        numbers = len(re.findall(r"\d", text))

        score = (
            min(wps / 3.2, 1.0) * 2.0
            + mean_e * 1.5
            + min(var_e * 4.0, 1.0) * 1.2
            + min(cues / 3.0, 1.0) * 2.2
            + min(questions, 2) * 0.6
            + min(numbers / 6.0, 1.0) * 0.7
        )
        candidates.append({
            "start": start, "end": end, "score": round(float(score), 3),
            "title": " ".join(w["word"] for w in span[:8]),
            "hook": "", "reason": f"wps={wps:.1f} energy={mean_e:.2f} cues={cues}",
            "source": "heuristic",
        })

    candidates.sort(key=lambda c: c["score"], reverse=True)
    duration = float(transcript.get("duration") or meta.get("duration") or 0)
    picked: list[dict] = []

    def free(cand) -> bool:
        return all(cand["end"] <= p["start"] or cand["start"] >= p["end"]
                   for p in picked)

    # Pass 1 - one clip from each slice of the runtime. Taking the global top N
    # by score clusters them wherever the audio happens to be liveliest and can
    # leave half the video untouched; this covers it end to end.
    if scfg.get("spread", True) and duration > 0 and want > 1:
        edges = np.linspace(0.0, duration, want + 1)
        for lo_t, hi_t in zip(edges[:-1], edges[1:]):
            for cand in candidates:
                mid = (cand["start"] + cand["end"]) / 2
                if lo_t <= mid < hi_t and free(cand):
                    picked.append(cand)
                    break

    # Pass 2 - fill any shortfall with the best of what is left, anywhere.
    for cand in candidates:
        if len(picked) >= want:
            break
        if free(cand):
            picked.append(cand)

    picked.sort(key=lambda c: c["start"])
    return picked[:want]


# --------------------------------------------------------------------------- #
# AI scoring
# --------------------------------------------------------------------------- #

def _api_key() -> str | None:
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key.strip()
    key_file = ROOT / ".anthropic_key"
    if key_file.exists():
        text = key_file.read_text(encoding="utf-8").strip()
        if text:
            return text
    return None


PROMPT = """You are a short-form video editor. Below is a timestamped transcript \
of a video titled "{title}".

Pick the {n} strongest standalone moments to cut as vertical Reels/Shorts.

Rules:
- Each clip must be between {lo} and {hi} seconds long.
- A clip must make sense on its own to someone who has not seen the video.
- Start on a complete thought, not mid-sentence. Prefer starting on a hook: a \
question, a bold claim, a surprising number, or the start of a story.
- End on a payoff or a clean stopping point, never mid-sentence.
- Clips must not overlap and should be spread across the video.
- Skip intros, sponsor reads, sign-offs, and rambling.

Transcript (each line is "[start_seconds] text"):
{transcript}

Reply with ONLY a JSON array, no prose, no markdown fences. Each element:
{{"start": <float seconds>, "end": <float seconds>, "title": "<4-8 word label>", \
"hook": "<punchy on-screen hook, max 55 chars>", "reason": "<why this works, one \
sentence>", "score": <1-10>}}"""


def _extract_json(text: str):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()
    start, end = text.find("["), text.rfind("]")
    if start == -1 or end == -1:
        raise ValueError("no JSON array in model reply")
    return json.loads(text[start:end + 1])


def select_ai(transcript: dict, meta: dict, cfg: dict) -> list[dict]:
    scfg = cfg["select"]
    key = _api_key()
    if not key:
        raise RuntimeError("no ANTHROPIC_API_KEY (env var or .anthropic_key file)")

    import anthropic

    lines = [f"[{s['start']:.1f}] {s['text']}" for s in _sentences(transcript)]
    body = "\n".join(lines)
    if len(body) > 240_000:
        body = body[:240_000]
        log("select", "transcript truncated to fit the model context")

    prompt = PROMPT.format(
        title=meta.get("title", "video"), n=int(scfg.get("clips", 5)),
        lo=int(scfg.get("min_seconds", 20)), hi=int(scfg.get("max_seconds", 60)),
        transcript=body,
    )

    client = anthropic.Anthropic(api_key=key)
    resp = client.messages.create(
        model=scfg.get("ai_model", "claude-sonnet-5"),
        max_tokens=4000,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text")
    items = _extract_json(raw)

    usage = getattr(resp, "usage", None)
    if usage:
        log("select", f"claude usage: in={usage.input_tokens} out={usage.output_tokens}")

    out = []
    for item in items:
        try:
            out.append({
                "start": float(item["start"]), "end": float(item["end"]),
                "title": str(item.get("title", ""))[:80],
                "hook": str(item.get("hook", ""))[:80],
                "reason": str(item.get("reason", ""))[:200],
                "score": float(item.get("score", 5)),
                "source": "ai",
            })
        except (KeyError, TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- #

def select_clips(transcript: dict, meta: dict, cfg: dict,
                 use_ai: bool | None = None) -> list[dict]:
    """Pick clips, validate them against the transcript, and snap to word edges."""
    scfg = cfg["select"]
    want = int(scfg.get("clips", 5))
    lo = float(scfg.get("min_seconds", 20))
    hi = float(scfg.get("max_seconds", 60))
    duration = transcript.get("duration") or meta.get("duration", 0)
    if use_ai is None:
        use_ai = bool(scfg.get("use_ai", True))

    # Ask for more than we need. Snapping to word boundaries nudges clips into
    # each other and the overlap check then drops them, so a request for N
    # candidates reliably yields fewer than N finished clips.
    spare_cfg = {**cfg, "select": {**scfg, "clips": want * 2 + 4}}

    clips: list[dict] = []
    if use_ai:
        try:
            clips = select_ai(transcript, meta, spare_cfg)
            log("select", f"claude proposed {len(clips)} moments")
        except Exception as exc:
            log("select", f"AI selection unavailable ({exc}); using offline scoring")
            clips = []
    if not clips:
        clips = select_heuristic(transcript, meta, spare_cfg)
        log("select", f"offline scoring proposed {len(clips)} moments")

    # Clean every candidate first, then choose - stopping at `want` here would
    # leave the spread pass below nothing to spread over.
    cleaned: list[dict] = []
    for c in sorted(clips, key=lambda c: -c.get("score", 0)):
        start = max(0.0, float(c["start"]))
        end = min(float(c["end"]), duration or float(c["end"]))
        if end - start < lo * 0.6:
            continue
        if end - start > hi:
            end = start + hi
        start, end = snap_to_words(transcript, start, end,
                                   float(scfg.get("pad_start", 0.35)),
                                   float(scfg.get("pad_end", 0.45)))
        end = min(end, duration or end)
        if end - start < lo * 0.6:
            continue
        if any(start < p["end_t"] and end > p["start_t"] for p in cleaned):
            continue
        c = dict(c)
        c["start_t"], c["end_t"] = round(start, 3), round(end, 3)
        c["duration"] = round(end - start, 2)
        cleaned.append(c)

    # Spread the final choice across the runtime. `cleaned` is ranked by score,
    # so taking the head of it would cluster the reels wherever the video
    # happens to be loudest and leave the rest of it unused.
    if scfg.get("spread", True) and duration > 0 and want > 1 and len(cleaned) > want:
        chosen: list[dict] = []
        edges = np.linspace(0.0, duration, want + 1)
        for lo_t, hi_t in zip(edges[:-1], edges[1:]):
            for c in cleaned:
                if c in chosen:
                    continue
                if lo_t <= (c["start_t"] + c["end_t"]) / 2 < hi_t:
                    chosen.append(c)
                    break
        for c in cleaned:
            if len(chosen) >= want:
                break
            if c not in chosen:
                chosen.append(c)
        cleaned = chosen[:want]
    else:
        cleaned = cleaned[:want]

    cleaned.sort(key=lambda c: c["start_t"])
    for i, c in enumerate(cleaned, 1):
        c["index"] = i
    return cleaned
