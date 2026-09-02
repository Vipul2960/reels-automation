"""Shared helpers: paths, logging, config, ffmpeg/ffprobe wrappers."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "work"
OUTPUT = ROOT / "output"
ASSETS = ROOT / "assets"

_T0 = time.time()


_bar_open = [False]


def log(stage: str, msg: str) -> None:
    if _bar_open[0]:                 # finish the progress line first
        print(flush=True)
        _bar_open[0] = False
    print(f"[{time.time() - _T0:6.1f}s] [{stage:<9}] {msg}", flush=True)


_bar_last = [0.0]
BAR_INTERVAL = 0.25          # seconds between redraws


def progress(stage: str, done: float, total: float, note: str = "",
             width: int = 26) -> None:
    """One self-overwriting progress line, so a long stage never looks hung.

    ASCII only: this runs in cmd.exe as often as not, and a block character
    that renders as a question mark is worse than a hash.

    Redraws are throttled. yt-dlp calls its hook many times a second, and when
    output is redirected to a file `\\r` does not overwrite anything - every
    update becomes another line and a single download writes a 160 KB log.
    """
    total = max(1e-9, float(total))
    frac = min(1.0, max(0.0, done / total))

    now = time.time()
    if frac < 1.0 and now - _bar_last[0] < BAR_INTERVAL:
        return
    _bar_last[0] = now
    filled = int(round(width * frac))
    bar = "#" * filled + "." * (width - filled)
    line = (f"[{time.time() - _T0:6.1f}s] [{stage:<9}] [{bar}] {frac * 100:5.1f}%"
            f"{'  ' + note if note else ''}")
    sys.stdout.write("\r" + line.ljust(96))
    sys.stdout.flush()
    _bar_open[0] = True
    if frac >= 1.0:
        sys.stdout.write("\n")
        sys.stdout.flush()
        _bar_open[0] = False


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or (ROOT / "config.json")
    with open(cfg_path, "r", encoding="utf-8") as fh:
        return json.load(fh)


_TOOLS: dict[str, str] = {}


def tool(name: str) -> str:
    """Locate ffmpeg/ffprobe: bundled `bin/` first, then PATH.

    A portable copy ships its own binaries in `bin/`, so the project runs on a
    machine where nothing is installed. On a normal setup that folder is absent
    and the PATH copy is used.
    """
    if name in _TOOLS:
        return _TOOLS[name]
    local = ROOT / "bin" / f"{name}.exe"
    resolved = str(local) if local.exists() else name
    _TOOLS[name] = resolved
    return resolved


def check_tools() -> list[str]:
    """Names of required tools that cannot be found. Empty means good to go."""
    missing = []
    for name in ("ffmpeg", "ffprobe"):
        try:
            subprocess.run([tool(name), "-version"], stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL, check=True)
        except (OSError, subprocess.CalledProcessError):
            missing.append(name)
    return missing


def run(cmd: list[str], quiet: bool = True) -> subprocess.CompletedProcess:
    """Run a command, raising with captured stderr on failure."""
    if cmd and cmd[0] in ("ffmpeg", "ffprobe"):
        cmd = [tool(cmd[0]), *cmd[1:]]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE if quiet else None,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-15:]
        raise RuntimeError(
            f"command failed ({proc.returncode}): {' '.join(cmd[:6])} ...\n"
            + "\n".join(tail)
        )
    return proc


def ffprobe(path: Path) -> dict:
    """Return {width, height, fps, duration, has_audio} for a media file."""
    out = run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_streams", "-show_format", str(path),
    ]).stdout
    data = json.loads(out)
    video = next((s for s in data["streams"] if s["codec_type"] == "video"), None)
    audio = next((s for s in data["streams"] if s["codec_type"] == "audio"), None)
    if video is None:
        raise RuntimeError(f"no video stream in {path}")
    num, _, den = video.get("r_frame_rate", "30/1").partition("/")
    fps = float(num) / float(den or 1) if float(den or 1) else 30.0
    duration = float(data.get("format", {}).get("duration") or video.get("duration") or 0)
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": round(fps, 4) or 30.0,
        "duration": duration,
        "has_audio": audio is not None,
    }


_ENCODER_CACHE: dict[str, bool] = {}


def has_encoder(name: str) -> bool:
    if name not in _ENCODER_CACHE:
        out = run([tool("ffmpeg"), "-hide_banner", "-encoders"]).stdout
        for line in out.splitlines():
            parts = line.split()
            if len(parts) >= 2:
                _ENCODER_CACHE[parts[1]] = True
        _ENCODER_CACHE.setdefault(name, False)
    return _ENCODER_CACHE.get(name, False)


def pick_encoder(preference: str = "auto") -> tuple[str, list[str]]:
    """Choose the fastest available H.264 encoder and its quality flags."""
    if preference != "auto":
        return preference, []
    for name, flags in (
        ("h264_qsv", ["-preset", "medium", "-look_ahead", "0"]),
        ("h264_nvenc", ["-preset", "p4", "-rc", "vbr"]),
        ("h264_amf", ["-quality", "balanced"]),
        ("libx264", ["-preset", "veryfast"]),
    ):
        if has_encoder(name):
            return name, flags
    return "libx264", ["-preset", "veryfast"]


def quality_flags(encoder: str, q: int) -> list[str]:
    if encoder.endswith("_qsv"):
        return ["-global_quality", str(q)]
    if encoder.endswith("_nvenc"):
        return ["-cq", str(q)]
    if encoder.endswith("_amf"):
        return ["-qp_i", str(q), "-qp_p", str(q)]
    return ["-crf", str(q)]


def slugify(text: str, limit: int = 48) -> str:
    keep = []
    for ch in text:
        if ch.isalnum():
            keep.append(ch.lower())
        elif ch in " -_" and (not keep or keep[-1] != "-"):
            keep.append("-")
    return "".join(keep).strip("-")[:limit] or "clip"


def ensure_dirs() -> None:
    for d in (WORK, OUTPUT, ASSETS):
        d.mkdir(parents=True, exist_ok=True)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, ensure_ascii=False, indent=2)


def fmt_ts(seconds: float) -> str:
    m, s = divmod(max(0.0, seconds), 60)
    h, m = divmod(int(m), 60)
    return f"{h:d}:{m:02d}:{s:05.2f}" if h else f"{m:02d}:{s:05.2f}"
