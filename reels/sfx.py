"""Stage 7b - the sound design layer.

Every effect here is synthesised from noise and sine waves rather than
downloaded, which sidesteps sample licensing entirely and makes each sound a
handful of tunable numbers instead of a fixed file. Drop real .wav files into
assets/sfx/ (named whoosh.wav, impact.wav, riser.wav, pop.wav) and those are
used instead.

The effects are rendered into a single stereo bed the length of the clip, so
the mix is one extra ffmpeg input rather than a graph of N delayed sources.
"""
from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .util import ASSETS, log

SR = 48000


def _svf(x: np.ndarray, cutoff: np.ndarray, q: float = 1.4) -> np.ndarray:
    """Chamberlin state-variable low-pass with a per-sample cutoff."""
    f = 2.0 * np.sin(np.pi * np.clip(cutoff, 20.0, SR * 0.45) / SR)
    damp = 1.0 / q
    low = band = 0.0
    out = np.empty_like(x)
    for i in range(len(x)):
        high = x[i] - low - damp * band
        band += f[i] * high
        low += f[i] * band
        out[i] = band                       # band-pass output: airier than low
    return out


def _lowpass(x: np.ndarray, cutoff: float) -> np.ndarray:
    """One-pole low-pass. Tames the hiss a band-pass on noise leaves behind."""
    a = 1.0 - np.exp(-2.0 * np.pi * cutoff / SR)
    out = np.empty_like(x)
    y = 0.0
    for i in range(len(x)):
        y += a * (x[i] - y)
        out[i] = y
    return out


def _norm(x: np.ndarray) -> np.ndarray:
    """Every effect leaves at peak 1.0, so config gains mean what they say."""
    peak = float(np.abs(x).max())
    return (x / peak).astype(np.float32) if peak > 1e-6 else x.astype(np.float32)


def _env(n: int, attack: float, decay: float, power: float = 2.0) -> np.ndarray:
    a = max(1, int(n * attack))
    d = max(1, n - a)
    return np.concatenate([
        np.linspace(0.0, 1.0, a) ** 0.6,
        (np.linspace(1.0, 0.0, d) ** power),
    ])[:n]


def make_whoosh(seconds: float = 0.38, rng=None) -> np.ndarray:
    """Filtered noise sweeping up then down - a pass-by."""
    rng = rng or np.random.default_rng(7)
    n = int(SR * seconds)
    noise = rng.standard_normal(n).astype(np.float32)
    t = np.linspace(0.0, 1.0, n)
    cutoff = 280.0 + 2600.0 * np.sin(np.pi * t) ** 1.5
    swept = _svf(noise, cutoff, q=1.1)
    body = _lowpass(swept, 5000.0) + 0.35 * _lowpass(swept, 900.0)
    return _norm(body * _env(n, 0.28, 0.72, 1.6))


def make_impact(seconds: float = 0.55, rng=None) -> np.ndarray:
    """Low sine dropping in pitch, with a noise transient on the front."""
    rng = rng or np.random.default_rng(11)
    n = int(SR * seconds)
    t = np.arange(n) / SR
    freq = 78.0 * np.exp(-3.2 * t) + 34.0
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) * np.exp(-5.0 * t)

    click_n = int(SR * 0.035)
    click = rng.standard_normal(click_n).astype(np.float32)
    click = _svf(click, np.full(click_n, 2400.0), q=0.9) * _env(click_n, 0.02, 0.98, 3.0)

    out = body.astype(np.float32)
    out[:click_n] += click * 0.55
    return _norm(out)


def make_riser(seconds: float = 1.1, rng=None) -> np.ndarray:
    """Noise climbing in pitch and volume - tension before a hit."""
    rng = rng or np.random.default_rng(13)
    n = int(SR * seconds)
    noise = rng.standard_normal(n).astype(np.float32)
    t = np.linspace(0.0, 1.0, n)
    cutoff = 220.0 * np.exp(2.9 * t)
    swept = _lowpass(_svf(noise, cutoff, q=1.6), 6500.0)
    return _norm(swept * (t ** 2.2))


def make_pop(seconds: float = 0.07, rng=None) -> np.ndarray:
    """Short bright click for caption beats."""
    n = int(SR * seconds)
    t = np.arange(n) / SR
    tone = np.sin(2 * np.pi * (900.0 * np.exp(-24.0 * t)) * t)
    return _norm(tone * np.exp(-42.0 * t))


BUILDERS = {"whoosh": make_whoosh, "impact": make_impact,
            "riser": make_riser, "pop": make_pop}


def _write_wav(path: Path, mono: np.ndarray) -> None:
    data = np.clip(mono, -1.0, 1.0)
    pcm = (data * 32767).astype(np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm.tobytes())


def _read_wav(path: Path) -> np.ndarray | None:
    try:
        with wave.open(str(path), "rb") as wf:
            if wf.getframerate() != SR or wf.getsampwidth() != 2:
                return None
            raw = wf.readframes(wf.getnframes())
            x = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
            if wf.getnchannels() == 2:
                x = x.reshape(-1, 2).mean(axis=1)
            return x
    except Exception:
        return None


_CACHE: dict[str, np.ndarray] = {}


def load(name: str) -> np.ndarray:
    """A user-supplied sample if there is one, otherwise the synthesised sound."""
    if name in _CACHE:
        return _CACHE[name]
    user = ASSETS / "sfx" / f"{name}.wav"
    if user.exists():
        got = _read_wav(user)
        if got is not None and len(got):
            log("sfx", f"using your own {user.name}")
            _CACHE[name] = got
            return got
    generated = ASSETS / "sfx" / f"{name}.generated.wav"
    if not generated.exists():
        _write_wav(generated, BUILDERS[name]())
    got = _read_wav(generated)
    if got is None:
        got = BUILDERS[name]()
    _CACHE[name] = got
    return got


def build_bed(events: list[dict], duration: float, cfg: dict) -> np.ndarray | None:
    """Mix the placed effects into one mono track `duration` seconds long."""
    scfg = cfg.get("sfx", {})
    if not scfg.get("enabled", True) or not events:
        return None

    gains = scfg.get("gains", {})
    n = int(SR * duration) + SR // 2
    bed = np.zeros(n, dtype=np.float32)

    for ev in events:
        sample = load(ev["kind"])
        gain = float(ev.get("gain", 1.0)) * float(gains.get(ev["kind"], 1.0))
        start = int(max(0.0, ev["t"]) * SR)
        end = min(n, start + len(sample))
        if end <= start:
            continue
        bed[start:end] += sample[:end - start] * gain

    peak = float(np.abs(bed).max())
    if peak < 1e-6:
        return None
    headroom = float(scfg.get("peak", 0.72))
    return (bed / peak * headroom).astype(np.float32)


def write_bed(events: list[dict], duration: float, cfg: dict,
              path: Path) -> Path | None:
    bed = build_bed(events, duration, cfg)
    if bed is None:
        return None
    _write_wav(path, bed)
    kinds: dict[str, int] = {}
    for e in events:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    log("sfx", "placed " + ", ".join(f"{v}x {k}" for k, v in sorted(kinds.items())))
    return path


def plan_events(decisions: list[dict], words: list[dict], duration: float,
                cfg: dict) -> list[dict]:
    """Decide where effects go: cuts get a whoosh, hard-hit words get an impact."""
    scfg = cfg.get("sfx", {})
    events: list[dict] = []

    if scfg.get("whoosh_on_cuts", True):
        for d in decisions[1:]:                     # nothing before the first frame
            t = float(d["start"])
            if 0.05 < t < duration - 0.15:
                # Land slightly before the cut so the sound leads the picture.
                events.append({"t": t - 0.12, "kind": "whoosh", "gain": 1.0})

    if scfg.get("impact_on_emphasis", True):
        emph = [w for w in words if w.get("emph")]
        max_hits = int(scfg.get("max_impacts", 4))
        min_gap = float(scfg.get("impact_min_gap", 2.5))
        chosen: list[float] = []
        for w in emph:
            t = float(w["start"])
            if t < 0.2 or t > duration - 0.3:
                continue
            if any(abs(t - c) < min_gap for c in chosen):
                continue
            chosen.append(t)
            if len(chosen) >= max_hits:
                break
        for t in chosen:
            events.append({"t": t - 0.04, "kind": "impact", "gain": 1.0})

    events.sort(key=lambda e: e["t"])
    return events
