"""Reclaim disk by deleting the working media, keeping everything cheap.

    python cleanup.py            # show what would go
    python cleanup.py --yes      # actually delete

Only `work/` is touched - finished reels in `output/` are never read here.

Deleted: downloaded sources, extracted audio, cut clips, tightened clips and
sound beds. All of it is regenerable from the URL.

Kept: transcript.json, meta.json, clips.json and the per-clip analysis - a few
hundred KB per video that saves the slowest steps if a video is ever redone.
"""
from __future__ import annotations

import sys

from reels.util import WORK, read_json

MEDIA = {".mp4", ".wav", ".mkv", ".webm", ".m4a", ".part", ".ytdl"}


def human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def busy() -> list[str]:
    """Signs that a run is using work/ right now.

    Deleting a source video out from under a live render kills every clip it has
    not reached yet - this exists because that happened.
    """
    import subprocess
    import time

    reasons = []
    cutoff = time.time() - 15 * 60
    recent = [f for f in WORK.rglob("*")
              if f.is_file() and f.stat().st_mtime > cutoff]
    if recent:
        reasons.append(f"{len(recent)} file(s) in work/ changed in the last 15 min")

    try:
        out = subprocess.run(["tasklist", "/fi", "imagename eq ffmpeg.exe"],
                             capture_output=True, text=True, timeout=10).stdout
        if "ffmpeg.exe" in out:
            reasons.append("ffmpeg is running")
    except Exception:
        pass
    return reasons


def main() -> int:
    do_it = "--yes" in sys.argv or "-y" in sys.argv
    force = "--force" in sys.argv

    if do_it and not force:
        reasons = busy()
        if reasons:
            print("A run looks active right now:")
            for r in reasons:
                print(f"  - {r}")
            print()
            print("Deleting now would break it. Wait for it to finish, or pass")
            print("--force if you are sure nothing is running.")
            return 1

    if not WORK.is_dir():
        print("Nothing to clean - no work folder.")
        return 0

    total = 0
    rows = []
    for folder in sorted(p for p in WORK.iterdir() if p.is_dir()):
        victims = [f for f in folder.rglob("*")
                   if f.is_file() and f.suffix.lower() in MEDIA]
        if not victims:
            continue
        size = sum(f.stat().st_size for f in victims)
        total += size

        title = folder.name
        meta_path = folder / "meta.json"
        if meta_path.exists():
            try:
                title = read_json(meta_path).get("title", folder.name)
            except Exception:
                pass
        rows.append((title, len(victims), size, victims))

    if not rows:
        print("Nothing to clean - no media files in work/.")
        return 0

    print(f"{'video':<50}{'files':>7}{'size':>12}")
    print("-" * 69)
    for title, count, size, _ in rows:
        print(f"{title[:48]:<50}{count:>7}{human(size):>12}")
    print("-" * 69)
    print(f"{'total':<50}{sum(r[1] for r in rows):>7}{human(total):>12}")

    if not do_it:
        print("\nThis was a preview. Run with --yes to delete.")
        return 0

    removed = 0
    failed = []
    for _, _, _, victims in rows:
        for f in victims:
            try:
                removed += f.stat().st_size
                f.unlink()
            except OSError as exc:
                failed.append((f, exc))

    # Drop the now-empty cuts folders too.
    for folder in WORK.iterdir():
        cuts = folder / "cuts"
        if cuts.is_dir() and not any(cuts.iterdir()):
            cuts.rmdir()

    print(f"\nFreed {human(removed)}.")
    if failed:
        print(f"{len(failed)} file(s) could not be deleted (in use?):")
        for f, exc in failed[:5]:
            print(f"  {f.name}: {exc}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
