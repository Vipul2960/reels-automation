"""Stage 2 - speech to text with word-level timestamps (faster-whisper, CPU int8).

Long videos are transcribed in chunks and checkpointed after every chunk, so an
interrupted run (crash, power cut, Ctrl+C) resumes where it stopped instead of
starting the whole file again. Chunks overlap slightly and words inside the
overlap are dropped, so no word is cut in half at a boundary.
"""
from __future__ import annotations

import os
import wave
from pathlib import Path

import numpy as np

from .util import log, progress, read_json, run, write_json, tool

SAMPLE_RATE = 16000
CHUNK_SECONDS = 300.0        # checkpoint every 5 minutes of audio
OVERLAP_SECONDS = 3.0        # re-read this much so boundary words stay whole

_MODEL_CACHE: dict[tuple, object] = {}


def _get_model(name: str, compute_type: str):
    key = (name, compute_type)
    if key not in _MODEL_CACHE:
        from faster_whisper import WhisperModel

        threads = max(2, min(8, (os.cpu_count() or 4)))
        log("transcribe", f"loading model '{name}' ({compute_type}, {threads} threads)")
        _MODEL_CACHE[key] = WhisperModel(
            name, device="cpu", compute_type=compute_type, cpu_threads=threads
        )
    return _MODEL_CACHE[key]


def _extract_audio(video_path: str, wav_path: Path) -> Path:
    """16 kHz mono PCM - what Whisper wants, and cheap to slice."""
    if wav_path.exists() and wav_path.stat().st_size > 1000:
        return wav_path
    log("transcribe", "extracting audio")
    run([tool("ffmpeg"), "-y", "-v", "error", "-i", str(video_path),
         "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE),
         "-acodec", "pcm_s16le", str(wav_path)])
    return wav_path


def _load_audio(wav_path: Path) -> np.ndarray:
    with wave.open(str(wav_path), "rb") as wf:
        raw = wf.readframes(wf.getnframes())
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


def transcribe(meta: dict, cfg: dict, force: bool = False) -> dict:
    """Return {'language', 'segments': [{start,end,text,words:[{word,start,end}]}]}."""
    tcfg = cfg.get("transcribe", {})
    workdir = Path(meta["workdir"])
    out_path = workdir / "transcript.json"
    part_path = workdir / "transcript.partial.json"

    if out_path.exists() and not force:
        data = read_json(out_path)
        log("transcribe", f"cached -> {len(data['segments'])} segments, "
                          f"lang={data.get('language')}")
        return data

    if force:
        part_path.unlink(missing_ok=True)

    audio = _load_audio(_extract_audio(meta["path"], workdir / "audio.wav"))
    total_seconds = len(audio) / SAMPLE_RATE
    n_chunks = max(1, int(np.ceil(total_seconds / CHUNK_SECONDS)))

    # Resume from whatever survived the last run.
    done_chunks: set[int] = set()
    segments: list[dict] = []
    language = tcfg.get("language")
    lang_prob = 0.0
    if part_path.exists() and not force:
        try:
            partial = read_json(part_path)
            if int(partial.get("n_chunks", -1)) == n_chunks:
                done_chunks = set(partial.get("done_chunks", []))
                segments = partial.get("segments", [])
                language = partial.get("language") or language
                lang_prob = partial.get("language_probability", 0.0)
                log("transcribe", f"resuming - {len(done_chunks)}/{n_chunks} chunks "
                                  f"already done ({len(segments)} segments kept)")
        except Exception as exc:
            log("transcribe", f"checkpoint unreadable ({exc}); starting over")

    todo = n_chunks - len(done_chunks)
    if todo > 2:
        # Roughly measured on this machine: ~6 min of wall clock per 5 min chunk
        # on hard audio. Worth saying out loud before committing to it.
        log("transcribe", f"WARNING: {todo} chunks left to transcribe locally "
                          f"(~{todo * 6} min). YouTube captions would be seconds - "
                          f"stop and retry if the caption fetch just failed.")

    model = _get_model(tcfg.get("model", "small"), tcfg.get("compute_type", "int8"))

    for idx in range(n_chunks):
        if idx in done_chunks:
            continue

        chunk_start = idx * CHUNK_SECONDS
        read_from = max(0.0, chunk_start - (OVERLAP_SECONDS if idx else 0.0))
        read_to = min(total_seconds, (idx + 1) * CHUNK_SECONDS)
        block = audio[int(read_from * SAMPLE_RATE):int(read_to * SAMPLE_RATE)]
        if len(block) < SAMPLE_RATE // 2:
            done_chunks.add(idx)
            continue

        log("transcribe", f"chunk {idx + 1}/{n_chunks} "
                          f"({chunk_start / 60:.1f}-{read_to / 60:.1f} min)")

        seg_iter, info = model.transcribe(
            block,
            language=language,
            beam_size=int(tcfg.get("beam_size", 1)),
            word_timestamps=True,
            vad_filter=bool(tcfg.get("vad_filter", True)),
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,
        )
        if language is None:
            language = info.language
            lang_prob = float(info.language_probability or 0)
            log("transcribe", f"detected language: {language} ({lang_prob:.0%})")

        added = 0
        for seg in seg_iter:
            words = []
            for w in (seg.words or []):
                token = (w.word or "").strip()
                if not token:
                    continue
                start = float(w.start) + read_from
                # Drop anything already covered by the previous chunk.
                if start < chunk_start - 1e-6:
                    continue
                words.append({
                    "word": token,
                    "start": round(start, 3),
                    "end": round(float(w.end) + read_from, 3),
                    "prob": round(float(getattr(w, "probability", 1.0) or 1.0), 3),
                })
            if not words:
                continue
            segments.append({
                "start": words[0]["start"],
                "end": words[-1]["end"],
                "text": (seg.text or "").strip(),
                "words": words,
            })
            added += 1

        done_chunks.add(idx)
        segments.sort(key=lambda s: s["start"])
        write_json(part_path, {
            "n_chunks": n_chunks,
            "done_chunks": sorted(done_chunks),
            "language": language,
            "language_probability": lang_prob,
            "segments": segments,
        })
        words_so_far = sum(len(s["words"]) for s in segments)
        progress("transcribe", len(done_chunks), n_chunks,
                 f"{words_so_far} words so far")

    data = {
        "language": language,
        "language_probability": round(lang_prob, 3),
        "duration": total_seconds,
        "segments": segments,
    }
    write_json(out_path, data)
    part_path.unlink(missing_ok=True)

    total_words = sum(len(s["words"]) for s in segments)
    log("transcribe", f"done -> {len(segments)} segments, {total_words} words, "
                      f"lang={language}")
    return data


def flat_words(transcript: dict) -> list[dict]:
    """All words across all segments, in order."""
    return [w for seg in transcript["segments"] for w in seg["words"]]


def words_in_range(transcript: dict, start: float, end: float) -> list[dict]:
    return [w for w in flat_words(transcript) if w["end"] > start and w["start"] < end]


def snap_to_words(transcript: dict, start: float, end: float,
                  pad_start: float = 0.3, pad_end: float = 0.4) -> tuple[float, float]:
    """Nudge clip boundaries so we never cut a word in half."""
    words = flat_words(transcript)
    if not words:
        return start, end
    inside = [w for w in words if w["start"] >= start - 1.0 and w["end"] <= end + 1.0]
    if not inside:
        return start, end
    first = min((w for w in inside if w["end"] > start),
                key=lambda w: w["start"], default=inside[0])
    last = max((w for w in inside if w["start"] < end),
               key=lambda w: w["end"], default=inside[-1])
    return max(0.0, first["start"] - pad_start), last["end"] + pad_end
