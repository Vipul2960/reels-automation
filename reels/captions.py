"""Stage 2a - use YouTube's own captions instead of transcribing locally.

YouTube's ASR is far stronger than a local `small` Whisper model on languages
with little training data, it already carries per-word timing in the `json3`
format, and it costs seconds instead of an hour of CPU. So this is tried first
and local Whisper is the fallback.

The parsed result has exactly the shape `transcribe()` returns, so the rest of
the pipeline cannot tell the difference.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from .util import log, write_json

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) reels-automation/0.1"


def _pick_track(auto: dict, manual: dict, want: str | None,
                video_lang: str | None) -> tuple[str, list] | None:
    """Prefer human captions, then the original-language ASR track."""
    for source in (manual, auto):
        if not source:
            continue
        if want:
            for key in (f"{want}-orig", want):
                if key in source:
                    return key, source[key]
        # "-orig" is YouTube's untranslated track. A video with dubbed audio
        # tracks has several ("hi-orig", "bn-orig", "en-US-orig"), so match the
        # video's own language first - picking whichever came back first would
        # caption a Hindi vlog in Bengali.
        if video_lang:
            base = video_lang.split("-")[0].lower()
            for key in source:
                if key.endswith("-orig") and key.split("-")[0].lower() == base:
                    return key, source[key]
            if video_lang in source:
                return video_lang, source[video_lang]
        for key in source:
            if key.endswith("-orig"):
                return key, source[key]
        if source is manual and source:
            key = next(iter(source))
            return key, source[key]
    return None


def _parse_json3(data: dict) -> list[dict]:
    """YouTube json3 -> our segment/word structure."""
    segments = []
    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs or event.get("aAppend"):
            continue                     # append events only carry newlines
        base = float(event.get("tStartMs", 0)) / 1000.0
        span = float(event.get("dDurationMs", 0)) / 1000.0

        words = []
        for seg in segs:
            token = (seg.get("utf8") or "").strip()
            if not token:
                continue
            start = base + float(seg.get("tOffsetMs", 0)) / 1000.0
            conf = seg.get("acAsrConf")
            words.append({
                "word": token,
                "start": round(start, 3),
                "end": round(start, 3),          # filled in below
                "prob": round(float(conf) / 255.0, 3) if conf else 1.0,
            })
        if not words:
            continue

        # A word ends where the next one starts; the last one runs to the event end.
        for i, w in enumerate(words):
            if i + 1 < len(words):
                w["end"] = round(max(w["start"] + 0.06, words[i + 1]["start"]), 3)
            else:
                w["end"] = round(max(w["start"] + 0.18, base + span), 3)

        segments.append({
            "start": words[0]["start"],
            "end": words[-1]["end"],
            "text": " ".join(w["word"] for w in words),
            "words": words,
        })

    segments.sort(key=lambda s: s["start"])

    # YouTube's rolling captions overlap on screen: each event's stated duration
    # runs past the next event's start. Left alone that makes caption lines pile
    # on top of each other, so trim every event where the next one begins.
    for i, seg in enumerate(segments[:-1]):
        next_start = segments[i + 1]["start"]
        if seg["end"] > next_start:
            seg["end"] = round(max(seg["words"][-1]["start"] + 0.06, next_start), 3)
            last = seg["words"][-1]
            last["end"] = min(last["end"], seg["end"])
    return segments


def fetch(meta: dict, cfg: dict, force: bool = False) -> dict | None:
    """Return a transcript from YouTube captions, or None if there are none."""
    import yt_dlp

    tcfg = cfg.get("transcribe", {})
    workdir = Path(meta["workdir"])
    out_path = workdir / "transcript.json"

    opts = {"quiet": True, "no_warnings": True, "skip_download": True}
    if cfg.get("download", {}).get("cookies_from_browser"):
        opts["cookiesfrombrowser"] = (cfg["download"]["cookies_from_browser"],)

    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(meta["url"], download=False)
    except Exception as exc:
        log("captions", f"could not read caption list ({exc})")
        return None

    picked = _pick_track(info.get("automatic_captions") or {},
                         info.get("subtitles") or {},
                         tcfg.get("language"), info.get("language"))
    if not picked:
        log("captions", "this video has no captions")
        return None

    lang, formats = picked
    track = next((f for f in formats if f.get("ext") == "json3"), None)
    if track is None:
        log("captions", f"'{lang}' captions exist but not in json3 "
                        f"(no word timings) - falling back to Whisper")
        return None

    log("captions", f"fetching YouTube captions: {lang}")

    # Retry transient failures. A 502/503 from YouTube means "try again", not
    # "this video has no captions" - and treating it as the latter silently
    # starts an hour of local Whisper on a long video.
    attempts = int(tcfg.get("caption_retries", 4))
    data = None
    for attempt in range(1, attempts + 1):
        try:
            req = urllib.request.Request(track["url"], headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as exc:
            transient = isinstance(exc, (urllib.error.URLError, TimeoutError)) or (
                isinstance(exc, urllib.error.HTTPError) and exc.code >= 500)
            if attempt < attempts and transient:
                wait = 2.0 * attempt
                log("captions", f"attempt {attempt}/{attempts} failed ({exc}) "
                                f"- retrying in {wait:.0f}s")
                time.sleep(wait)
                continue
            log("captions", f"download failed after {attempt} attempt(s): {exc}")
            return None
    if data is None:
        return None

    segments = _parse_json3(data)
    words = sum(len(s["words"]) for s in segments)
    min_words = int(tcfg.get("min_caption_words", 200))
    if words < min_words:
        log("captions", f"only {words} words in the caption track "
                        f"(under {min_words}) - falling back to Whisper")
        return None

    covered = sum(s["end"] - s["start"] for s in segments)
    duration = meta.get("duration") or (segments[-1]["end"] if segments else 0)
    transcript = {
        "language": lang.replace("-orig", ""),
        "language_probability": 1.0,
        "duration": float(duration),
        "segments": segments,
        "source": f"youtube:{lang}",
    }
    write_json(out_path, transcript)
    log("captions", f"{len(segments)} segments, {words} words, "
                    f"{covered/60:.1f} min of speech across {duration/60:.1f} min "
                    f"({covered/max(duration,1):.0%} coverage)")
    return transcript
