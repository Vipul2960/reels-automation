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


def read_links(args_links: list[str], links_file: Path | None) -> list[dict]:
    """Collect URLs from the command line and/or a text file, de-duplicated.

    The file is read loosely: one per line, blank lines and #comments ignored,
    and a URL is picked out of any surrounding text so a pasted list works.

    Returns the same shape as prompt_links so the caller treats both alike;
    `clips` is None here, meaning "use the run-wide default".
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
            out.append({"url": tidy, "clips": None})
    return out


def probe(url: str) -> dict | None:
    """Title and duration without downloading anything. Takes a second or two."""
    import yt_dlp

    class _Silent:
        """yt-dlp prints its own ERROR line otherwise; we report failures."""
        def debug(self, msg): pass
        def info(self, msg): pass
        def warning(self, msg): pass
        def error(self, msg): pass

    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                               "skip_download": True, "noplaylist": True,
                               "logger": _Silent()}) as ydl:
            info = ydl.extract_info(url, download=False)
        if info.get("_type") == "playlist":
            entries = [e for e in (info.get("entries") or []) if e]
            if not entries:
                return None
            info = entries[0]
        return {"title": info.get("title", ""),
                "duration": float(info.get("duration") or 0)}
    except Exception:
        return None


def suggest_clips(seconds: float) -> tuple[int, int, str]:
    """(suggested, ceiling, why) for a video of this length.

    The ceiling is what the selector can actually produce - measured at roughly
    one reel per minute of source, since clips cannot overlap. The suggestion is
    well under that: taking every possible clip means taking the weak ones too.
    """
    minutes = max(0.5, seconds / 60.0)
    ceiling = max(1, int(minutes))
    suggested = max(3, min(30, int(round(minutes * 0.55))))
    suggested = min(suggested, ceiling)

    if minutes < 5:
        why = "short video - only a few strong moments in it"
    elif minutes < 15:
        why = "about half the video, keeping the better moments"
    elif minutes < 40:
        why = "roughly half the video; ask for more if you want fuller coverage"
    else:
        why = "long video - this is already a lot of posts"
    return suggested, ceiling, why


def prompt_links(default_clips: int = 12) -> list[dict]:
    """Ask for links one at a time, each with its own reel count.

    Each link is looked up as it is entered, so the title confirms the right
    video was pasted and the reel count can be suggested from its real length.
    """
    print()
    print("  Paste one link at a time. After each, say how many reels you want")
    print("  from it. Press Enter on an empty link to finish.")

    entries: list[dict] = []
    while True:
        print()
        try:
            raw = input(f"    Link {len(entries) + 1}: ").strip()
        except EOFError:
            break
        if not raw:
            break

        url = clean_url(raw.strip('"').strip("'"))
        if any(e["url"] == url for e in entries):
            print("      already in the list - skipped")
            continue

        print("      looking it up ...", end="\r")
        info = probe(url)
        if info is None:
            print("      could not read that link - is it a real video URL?")
            try:
                keep = input("      Add it anyway? [y/N]: ").strip().lower()
            except EOFError:
                break
            if keep not in ("y", "yes"):
                continue
            info = {"title": "(unknown)", "duration": 0.0}

        minutes = info["duration"] / 60.0
        suggested, ceiling, why = suggest_clips(info["duration"])
        print(f"      {info['title'][:58]}")
        print(f"      {minutes:.0f} min  ->  suggested {suggested} reels "
              f"(max about {ceiling}) - {why}")

        try:
            answer = input(f"      How many reels? [{suggested}]: ").strip()
        except EOFError:
            answer = ""
        try:
            count = int(answer) if answer else suggested
        except ValueError:
            count = suggested
        count = max(1, min(count, 60))
        if count > ceiling:
            print(f"      note: about {ceiling} is all this video can give - "
                  f"you will get what exists")

        entries.append({"url": url, "clips": count,
                        "title": info["title"], "duration": info["duration"]})

        try:
            more = input("    Add another link? [Y/n]: ").strip().lower()
        except EOFError:
            break
        if more in ("n", "no"):
            break

    if entries:
        print()
        print("  Queued:")
        for i, e in enumerate(entries, 1):
            print(f"    {i}. {e['clips']:>2} reels  {e['duration']/60:>5.0f} min  "
                  f"{e['title'][:46]}")
    return entries


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
