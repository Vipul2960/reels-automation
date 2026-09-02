"""Stage 1 - download the source video with yt-dlp (cached by video id)."""
from __future__ import annotations

import time
from pathlib import Path

from .util import WORK, ffprobe, log, progress, read_json, write_json, tool


def verify(path: Path, probe_seconds: float = 4.0) -> tuple[bool, str]:
    """Decode-check the head and tail so a half-written file is caught early.

    A power cut or killed download leaves a file whose container header still
    parses - ffprobe reports the full duration - while the last written frames
    are garbage. Only decoding catches that.
    """
    import subprocess

    if not path.exists() or path.stat().st_size < 100_000:
        return False, "file missing or too small"
    try:
        info = ffprobe(path)
    except Exception as exc:
        return False, f"unreadable container: {exc}"
    if info["duration"] < 1.0:
        return False, "no duration"

    checks = (("start", ["-ss", "0"]), ("end", ["-sseof", f"-{probe_seconds:g}"]))
    for label, seek in checks:
        proc = subprocess.run(
            [tool("ffmpeg"), "-v", "error", *seek, "-i", str(path),
             "-t", f"{probe_seconds:g}", "-f", "null", "-"],
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
            text=True, encoding="utf-8", errors="replace",
        )
        errors = [ln for ln in (proc.stderr or "").splitlines() if ln.strip()]
        if errors:
            return False, f"{len(errors)} decode errors near the {label} ({errors[0][:70]})"
    return True, "ok"


def _progress(status: dict) -> None:
    """Live download bar."""
    if status.get("status") == "finished":
        return
    if status.get("status") != "downloading":
        return
    total = status.get("total_bytes") or status.get("total_bytes_estimate") or 0
    done = status.get("downloaded_bytes") or 0
    if not total:
        return
    speed = (status.get("speed") or 0) / 1e6
    eta = status.get("eta")
    note = f"{done/1e6:.0f}/{total/1e6:.0f} MB"
    if speed:
        note += f"  {speed:.1f} MB/s"
    if eta:
        note += f"  eta {int(eta)}s"
    progress("download", done, total, note)


def clean_url(url: str) -> str:
    """Reduce a browser-copied YouTube link to the single video it points at.

    Copying a link while autoplay/radio is on gives something like
    `watch?v=ID&list=RD...&start_radio=1`. yt-dlp reads that as a playlist and
    would grab the whole mix, but the `v=` says exactly which video is wanted.
    Real playlist links (`/playlist?list=...`, no `v=`) are left alone so the
    caller still gets a clear error.
    """
    import urllib.parse as up

    try:
        parts = up.urlsplit(url)
    except ValueError:
        return url
    host = parts.netloc.lower().removeprefix("www.").removeprefix("m.")

    if host in ("youtu.be",):
        vid = parts.path.strip("/").split("/")[0]
        return f"https://www.youtube.com/watch?v={vid}" if vid else url

    if host.endswith("youtube.com"):
        segments = [s for s in parts.path.split("/") if s]
        if segments and segments[0] in ("shorts", "live", "embed") and len(segments) > 1:
            return f"https://www.youtube.com/watch?v={segments[1]}"
        vid = up.parse_qs(parts.query).get("v", [None])[0]
        if vid:
            return f"https://www.youtube.com/watch?v={vid}"
    return url


def download(url: str, cfg: dict, force: bool = False) -> dict:
    """Download `url` to work/<id>/source.mp4 and return its metadata."""
    import yt_dlp

    dl_cfg = cfg.get("download", {})
    max_h = dl_cfg.get("max_height", 1080)

    tidy = clean_url(url)
    if tidy != url:
        log("download", "link had playlist/radio parameters - using just the video")
    url = tidy

    probe_opts = {"quiet": True, "no_warnings": True, "skip_download": True,
                  "noplaylist": True}
    if dl_cfg.get("cookies_from_browser"):
        probe_opts["cookiesfrombrowser"] = (dl_cfg["cookies_from_browser"],)

    log("download", "reading video info ...")
    with yt_dlp.YoutubeDL(probe_opts) as ydl:
        info = ydl.extract_info(url, download=False)

    if info.get("_type") == "playlist":
        entries = [e for e in (info.get("entries") or []) if e]
        if len(entries) == 1:
            info = entries[0]
        else:
            raise RuntimeError(
                "That link is a playlist with no single video in it. "
                "Open one video and copy its URL instead.")

    vid = info["id"]
    workdir = WORK / vid
    workdir.mkdir(parents=True, exist_ok=True)
    target = workdir / "source.mp4"
    meta_path = workdir / "meta.json"

    meta = {
        "id": vid,
        "url": info.get("webpage_url", url),
        "title": info.get("title", vid),
        "uploader": info.get("uploader") or info.get("channel") or "",
        "duration": float(info.get("duration") or 0),
        "description": (info.get("description") or "")[:4000],
        "path": str(target),
        "workdir": str(workdir),
    }

    if target.exists() and target.stat().st_size > 0 and not force:
        ok, why = verify(target)
        if ok:
            log("download", f"cached -> {target.name} "
                            f"({target.stat().st_size/1e6:.1f} MB)")
            if meta_path.exists():
                meta.update({k: v for k, v in read_json(meta_path).items()
                             if k not in meta or not meta[k]})
            write_json(meta_path, meta)
            return meta
        log("download", f"cached file is damaged ({why}) - downloading again")
        target.unlink(missing_ok=True)

    log("download", f"{meta['title'][:70]}  ({meta['duration']/60:.1f} min)")

    # H.264 first: at the same resolution it is roughly half the bytes of the
    # VP9/AV1 "premium" variants and decodes far faster in OpenCV.
    if dl_cfg.get("prefer_h264", True):
        fmt = (f"bestvideo[height<={max_h}][vcodec^=avc1]+bestaudio[ext=m4a]/"
               f"bestvideo[height<={max_h}][vcodec^=avc1]+bestaudio/"
               f"bestvideo[height<={max_h}][ext=mp4]+bestaudio[ext=m4a]/"
               f"bestvideo[height<={max_h}]+bestaudio/best[height<={max_h}]/best")
    else:
        fmt = (f"bestvideo[height<={max_h}][ext=mp4]+bestaudio[ext=m4a]/"
               f"bestvideo[height<={max_h}]+bestaudio/best[height<={max_h}]/best")

    opts = {
        "format": fmt,
        "merge_output_format": "mp4",
        "outtmpl": str(workdir / "source.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "progress_hooks": [_progress],
        "concurrent_fragment_downloads": 4,
        # YouTube drops long downloads. yt-dlp keeps a .part file, so retrying
        # resumes rather than restarting - be generous with attempts.
        "retries": 10,
        "fragment_retries": 10,
        "file_access_retries": 5,
        "socket_timeout": 30,
        "continuedl": True,
        "overwrites": False,
    }
    if dl_cfg.get("cookies_from_browser"):
        opts["cookiesfrombrowser"] = (dl_cfg["cookies_from_browser"],)

    # Outer retry on top of yt-dlp's own: a connection reset mid-download raises
    # out of yt_dlp entirely, and the partial file it leaves behind means the
    # next attempt picks up where this one died instead of starting over.
    attempts = int(dl_cfg.get("download_attempts", 4))
    last_error = None
    for attempt in range(1, attempts + 1):
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([meta["url"]])
            last_error = None
            break
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                break
            log("download", f"attempt {attempt}/{attempts} failed ({exc}) "
                            f"- resuming in 5s")
            time.sleep(5)
    if last_error is not None:
        raise RuntimeError(f"download failed after {attempts} attempts: {last_error}")

    if not target.exists():
        # yt-dlp may have kept a different container extension.
        candidates = sorted(workdir.glob("source.*"), key=lambda p: p.stat().st_size, reverse=True)
        if not candidates:
            raise RuntimeError("download finished but no source file was produced")
        candidates[0].rename(target)

    ok, why = verify(target)
    if not ok:
        raise RuntimeError(
            f"download finished but the file will not decode cleanly ({why}). "
            f"Delete {target} and try again.")

    log("download", f"saved -> {target.name} "
                    f"({target.stat().st_size/1e6:.1f} MB, verified)")
    write_json(meta_path, meta)
    return meta
