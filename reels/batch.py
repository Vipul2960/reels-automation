"""Run many links in one go, unattended.

Built for leaving it running overnight, which changes what matters:

  * one bad link must not end the night - every video is isolated and the run
    carries on to the next
  * Windows must not fall asleep half way through
  * anything that would block on a question has to be decided up front
  * the morning needs a summary, not a scrollback
"""
from __future__ import annotations

import ctypes
import re
import shutil
from datetime import datetime
from pathlib import Path

from .download import clean_url
from .util import OUTPUT, WORK, log, read_json

URL_RE = re.compile(r"https?://\S+")

# Windows SetThreadExecutionState flags
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001
ES_DISPLAY_REQUIRED = 0x00000002


class KeepAwake:
    """Stop Windows sleeping mid-run. Restores normal behaviour on exit."""

    def __enter__(self):
        self.ok = False
        try:
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            self.ok = True
            log("batch", "sleep disabled for the duration of this run")
        except Exception:
            log("batch", "could not disable sleep - check Windows power settings")
        return self

    def __exit__(self, *exc):
        if self.ok:
            try:
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            except Exception:
                pass
        return False


def read_links(args_links: list[str], links_file: Path | None) -> list[str]:
    """Collect URLs from the command line and/or a text file, de-duplicated.

    The file is read loosely: one per line, blank lines and #comments ignored,
    and a URL is picked out of any surrounding text so a pasted list works.
    """
    raw: list[str] = list(args_links or [])
    if links_file and links_file.exists():
        for line in links_file.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            found = URL_RE.search(line)
            raw.append(found.group(0) if found else line)

    seen, out = set(), []
    for url in raw:
        tidy = clean_url(url.strip().strip('"').strip("'"))
        if tidy and tidy not in seen:
            seen.add(tidy)
            out.append(tidy)
    return out


def video_id(url: str) -> str | None:
    import urllib.parse as up
    try:
        return up.parse_qs(up.urlsplit(url).query).get("v", [None])[0]
    except ValueError:
        return None


def already_done(url: str) -> tuple[bool, str, int]:
    """(done, folder name, reel count) for a link processed before."""
    index_path = OUTPUT / "index.json"
    if not index_path.exists():
        return False, "", 0
    vid = video_id(url)
    if not vid:
        return False, "", 0
    entry = read_json(index_path).get(vid)
    if not entry:
        return False, "", 0
    folder = OUTPUT / entry.get("folder", "")
    if not folder.is_dir():
        return False, "", 0
    reels = sorted(folder.glob("[0-9][0-9].mp4"))
    return bool(reels), folder.name, len(reels)


def disk_report(links: int) -> tuple[float, float]:
    """(free GB, roughly needed GB). A source video runs 0.4-1.2 GB."""
    free = shutil.disk_usage(str(WORK.parent)).free / 1e9
    return free, links * 1.0


def cleanup_source(meta: dict) -> float:
    """Delete the downloaded source after its reels are made. Returns GB freed."""
    path = Path(meta.get("path", ""))
    freed = 0.0
    for target in (path, path.with_suffix(".wav"),
                   Path(meta.get("workdir", "")) / "audio.wav"):
        if target.is_file():
            freed += target.stat().st_size / 1e9
            target.unlink(missing_ok=True)
    return freed


def write_summary(results: list[dict], started: datetime) -> Path:
    """The thing to read in the morning."""
    path = OUTPUT / "batch-summary.txt"
    ok = [r for r in results if r["status"] == "done"]
    skipped = [r for r in results if r["status"] == "skipped"]
    failed = [r for r in results if r["status"] == "failed"]
    reels = sum(r.get("reels", 0) for r in ok)
    minutes = (datetime.now() - started).total_seconds() / 60

    lines = [
        "=" * 68,
        f"  BATCH SUMMARY   {started:%d %b %Y, %H:%M} - {datetime.now():%H:%M}",
        "=" * 68,
        f"  {len(ok)} done, {len(skipped)} skipped, {len(failed)} failed",
        f"  {reels} reels in {minutes:.0f} minutes",
        "",
    ]
    for group, title in ((ok, "DONE"), (skipped, "SKIPPED"), (failed, "FAILED")):
        if not group:
            continue
        lines += [title, "-" * len(title)]
        for r in group:
            note = r.get("note", "")
            head = f"  {r.get('folder') or r['url'][:58]}"
            if r["status"] == "done":
                head += f"   {r.get('reels', 0)} reels"
            lines.append(head)
            if note:
                lines.append(f"      {note}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path
