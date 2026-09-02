"""Write a copy-paste sheet for uploading the reels by hand.

One `upload-details.txt` per output folder: for every reel, a suggested title,
a ready description, the tags, and the words actually spoken in that clip.

The spoken text matters. Without an API key the suggested title is just the
opening words of the clip - serviceable, rarely good. Having the full text in
front of you makes writing a real title a five-second job instead of a
re-watch.
"""
from __future__ import annotations

from pathlib import Path

from .util import fmt_ts, log

RULE = "=" * 72


def suggest_title(clip: dict, cfg: dict) -> str:
    """Best title we can offer without understanding the language."""
    ucfg = cfg.get("upload", {})
    suffix = ucfg.get("title_suffix", " #Shorts")

    text = (clip.get("hook") or clip.get("title") or "").strip()
    text = " ".join(text.split())
    # Trim to the last sentence break so it does not end mid-thought.
    for stop in ("।", ".", "?", "!"):
        if stop in text[:70]:
            text = text[: text.index(stop) + 1]
            break
    text = text.rstrip(" ,-–—")
    return (text[: 100 - len(suffix)] + suffix) if text else f"Reel{suffix}"


def clip_text(transcript: dict | None, clip: dict, limit: int = 700) -> str:
    """What is actually said in this clip, from the transcript."""
    if not transcript:
        return ""
    words = []
    for seg in transcript.get("segments", []):
        for w in seg.get("words", []):
            if clip["start_t"] <= w["start"] < clip["end_t"]:
                words.append(w["word"])
    text = " ".join(words).strip()
    return text[:limit] + (" ..." if len(text) > limit else "")


def build_description(clip: dict, meta: dict, cfg: dict,
                      spoken: str = "") -> tuple[str, list[str]]:
    ucfg = cfg.get("upload", {})
    tags = list(ucfg.get("tags", ["shorts", "reels", "viral", "trending"]))

    lines = []
    hook = " ".join((clip.get("hook") or clip.get("title") or "").split())
    if hook:
        lines.append(hook)
    if spoken and spoken[:60] != hook[:60]:
        lines += ["", spoken[:300] + ("..." if len(spoken) > 300 else "")]
    if ucfg.get("credit_source", True) and meta.get("title"):
        lines += ["", f"From: {meta['title']}"]
        if meta.get("url"):
            lines.append(meta["url"])
    if ucfg.get("description_footer"):
        lines += ["", ucfg["description_footer"]]
    lines += ["", " ".join(f"#{t}" for t in tags[:15])]
    return "\n".join(lines).strip(), tags


def write_details(outdir: Path, clips: list[dict], meta: dict, cfg: dict,
                  transcript: dict | None = None) -> Path:
    """Write upload-details.txt next to the reels."""
    path = outdir / "upload-details.txt"
    out = []

    out.append(RULE)
    out.append(f"  UPLOAD SHEET  -  {len(clips)} reel(s)")
    out.append(RULE)
    out.append(f"  Source : {meta.get('title', '')}")
    if meta.get("url"):
        out.append(f"  URL    : {meta['url']}")
    out.append(f"  Folder : {outdir.name}")
    out.append("")
    out.append("  Paste the TITLE and DESCRIPTION into YouTube Studio for each")
    out.append("  file. A vertical video under 3 minutes becomes a Short on its")
    out.append("  own - no extra setting needed.")
    out.append("")

    for clip in clips:
        reel = Path(clip.get("output", ""))
        name = reel.name if reel.name else f"{clip['index']:02d}.mp4"
        size = f"{reel.stat().st_size / 1e6:.1f} MB" if reel.exists() else "-"
        spoken = clip_text(transcript, clip)
        description, tags = build_description(clip, meta, cfg, spoken)

        out.append("")
        out.append(RULE)
        out.append(f"  REEL {clip['index']:02d}   file: {name}   "
                   f"{clip.get('duration', 0):.0f}s   {size}")
        out.append(f"  from {fmt_ts(clip['start_t'])} - {fmt_ts(clip['end_t'])} "
                   f"of the source")
        out.append(RULE)
        out.append("")
        out.append("TITLE")
        out.append("-----")
        out.append(suggest_title(clip, cfg))
        out.append("")
        out.append("DESCRIPTION")
        out.append("-----------")
        out.append(description)
        out.append("")
        out.append("TAGS  (comma separated)")
        out.append("----")
        out.append(", ".join(tags))
        if spoken:
            out.append("")
            out.append("WHAT IS SAID IN THIS CLIP  (to write your own title)")
            out.append("----")
            out.append(spoken)
        out.append("")

    path.write_text("\n".join(out), encoding="utf-8")
    log("details", f"upload sheet -> {path.name}")
    return path
