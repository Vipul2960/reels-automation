"""YouTube -> vertical Reels, end to end.

    python run.py "<youtube url>" --clips 5
    python run.py "<youtube url>" --dry-run          # pick moments, render nothing
    python run.py "<youtube url>" --no-ai            # offline clip selection

Only feed this videos you own or are licensed to reuse.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
import traceback
from pathlib import Path

from reels import captions
from reels.analyze import analyze
from reels.download import download, verify
from reels.emphasis import mark_emphasis
from reels import batch, details, effects, sfx, upload as yt
from reels.framing import apply_shake
from reels.framing import plan as plan_framing
from reels.pacing import retime_words, tighten
from reels.render import cut_clip, render_reel
from reels.select import select_clips
from reels.subtitles import build_ass, caption_margin
from reels.transcribe import transcribe, words_in_range
from reels.util import (OUTPUT, ensure_dirs, fmt_ts, load_config, log,
                        pick_encoder, read_json, slugify, write_json)


def _use_utf8_console() -> None:
    """Windows consoles default to cp1252 and blow up on non-Latin titles."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Turn a YouTube video into vertical reels.")
    p.add_argument("urls", nargs="*", help="one or more YouTube video URLs")
    p.add_argument("--links", type=Path,
                   help="text file of URLs, one per line (blank lines and "
                        "#comments ignored)")
    p.add_argument("--clips", type=int, help="how many reels to produce")
    p.add_argument("--min", dest="min_seconds", type=float, help="minimum reel length")
    p.add_argument("--max", dest="max_seconds", type=float, help="maximum reel length")
    p.add_argument("--model", help="whisper model: tiny|base|small|medium")
    p.add_argument("--lang", help="force transcript language, e.g. en or hi")
    p.add_argument("--source", choices=("auto", "youtube", "whisper"),
                   help="where the transcript comes from (default: auto)")
    p.add_argument("--no-ai", action="store_true", help="skip Claude clip selection")
    p.add_argument("--no-subs", action="store_true", help="do not burn captions")
    p.add_argument("--dry-run", action="store_true", help="stop after choosing moments")
    p.add_argument("--only", type=int, help="render just this clip number")
    p.add_argument("--force", action="store_true", help="ignore cached download/transcript")
    p.add_argument("--upload", action="store_true",
                   help="after each reel, ask whether to upload it to YouTube")
    p.add_argument("--skip-existing", action="store_true",
                   help="silently skip links whose reels already exist")
    p.add_argument("--redo", action="store_true",
                   help="rebuild links whose reels already exist")
    p.add_argument("--clean", action="store_true",
                   help="delete each downloaded source video once its reels "
                        "are made (saves ~1 GB per video)")
    p.add_argument("--config", type=Path, help="alternate config.json")
    return p.parse_args(argv)


def apply_overrides(cfg: dict, args) -> dict:
    if args.clips:
        cfg["select"]["clips"] = args.clips
    if args.min_seconds:
        cfg["select"]["min_seconds"] = args.min_seconds
    if args.max_seconds:
        cfg["select"]["max_seconds"] = args.max_seconds
    if args.model:
        cfg["transcribe"]["model"] = args.model
    if args.lang:
        cfg["transcribe"]["language"] = args.lang
    if args.no_subs:
        cfg["subtitles"]["enabled"] = False
    if args.source:
        cfg["transcribe"]["source"] = args.source
    return cfg


def get_transcript(meta: dict, cfg: dict, args) -> dict:
    """YouTube captions when they exist and are usable, else local Whisper."""
    source = cfg["transcribe"].get("source", "auto")
    cached = Path(meta["workdir"]) / "transcript.json"

    if cached.exists() and not args.force:
        data = read_json(cached)
        log("transcribe", f"cached -> {len(data['segments'])} segments, "
                          f"lang={data.get('language')}, "
                          f"source={data.get('source', 'whisper')}")
        return data

    if source in ("auto", "youtube"):
        found = captions.fetch(meta, cfg, force=args.force)
        if found:
            return found
        if source == "youtube":
            raise RuntimeError(
                "no usable YouTube captions for this video; "
                "run again with --source whisper to transcribe locally")
        log("transcribe", "no usable captions - transcribing locally instead")

    return transcribe(meta, cfg, force=args.force)


ILLEGAL_IN_FILENAME = r'\/:*?"<>|' 


def safe_title(title: str, limit: int = 60) -> str:
    """Windows-safe folder name that still reads as the video's real title.

    The original script is kept - transliterating Devanagari into ASCII made
    folder names nobody could match to a video.
    """
    cleaned = "".join(" " if ch in ILLEGAL_IN_FILENAME else ch for ch in title)
    cleaned = " ".join(cleaned.split())
    cleaned = cleaned[:limit].rstrip(" .")
    return cleaned or "video"


def output_folder(meta: dict) -> Path:
    """One folder per video, named "0001 - Real Title".

    The number sorts and stays stable per video id, so re-running the same URL
    writes back into the same folder even if the title changed; the title is
    there so the folder is recognisable at a glance.
    """
    index_path = OUTPUT / "index.json"
    index = read_json(index_path) if index_path.exists() else {}

    entry = index.get(meta["id"])
    if entry is None:
        used = set()
        for v in index.values():
            head = str(v.get("folder", "")).split(" - ")[0]
            if head.isdigit():
                used.add(int(head))
        if OUTPUT.exists():
            for d in OUTPUT.iterdir():
                head = d.name.split(" - ")[0]
                if d.is_dir() and head.isdigit():
                    used.add(int(head))
        entry = {"number": max(used) + 1 if used else 1}

    number = int(entry.get("number") or str(entry.get("folder", "1")).split(" - ")[0])
    name = f"{number:04d} - {safe_title(meta['title'])}"

    folder = OUTPUT / name
    # Only rename when this video already had a folder. `Path(x) / ""` returns
    # x itself, so an absent entry would make this try to rename the whole
    # output directory into a subfolder of itself.
    old_name = str(entry.get("folder") or "")
    if old_name and old_name != name:
        old_folder = OUTPUT / old_name
        if old_folder.is_dir() and not folder.exists():
            old_folder.rename(folder)      # title changed: keep the same folder

    entry.update({"number": number, "folder": name, "id": meta["id"],
                  "title": meta["title"], "url": meta["url"]})
    index[meta["id"]] = entry
    write_json(index_path, index)

    folder.mkdir(parents=True, exist_ok=True)
    log("setup", f"output folder: {name}")
    return folder


def maybe_upload(reel: Path, clip: dict, meta: dict, cfg: dict) -> None:
    """Ask, then upload. Nothing leaves this machine without a typed yes."""
    left = yt.quota_left() // yt.UPLOAD_COST
    hook = (clip.get("hook") or clip.get("title") or reel.stem)[:70]
    print()
    print(f"    Upload this reel to YouTube?")
    print(f"      file  : {reel.name}  ({reel.stat().st_size / 1e6:.1f} MB, "
          f"{clip.get('duration', 0):.0f}s)")
    print(f"      title : {hook}")
    print(f"      quota : {left} upload(s) left today")
    try:
        answer = input("      [y/N]: ").strip().lower()
    except EOFError:
        answer = ""
    if answer not in ("y", "yes"):
        log("upload", "skipped")
        return
    try:
        yt.upload(reel, clip, meta.get("title", ""), cfg)
    except Exception as exc:
        log("upload", f"failed: {exc}")


def process_one(url: str, cfg: dict, args) -> dict:
    """Everything for a single link. Returns a batch result record."""
    meta = download(url, cfg, force=args.force)
    transcript = get_transcript(meta, cfg, args)

    clips = select_clips(transcript, meta, cfg, use_ai=not args.no_ai)
    if not clips:
        log("select", "no usable moments found - try --no-ai or a longer video")
        return {"url": url, "status": "failed", "note": "no usable moments"}

    print()
    log("select", f"{len(clips)} moments chosen:")
    for c in clips:
        print(f"    #{c['index']}  {fmt_ts(c['start_t'])} - {fmt_ts(c['end_t'])} "
              f"({c['duration']:>5.1f}s)  {c.get('title', '')}")
        if c.get("reason"):
            print(f"          why: {c['reason']}")
    print()

    workdir = Path(meta["workdir"])
    outdir = output_folder(meta)
    write_json(workdir / "clips.json", clips)

    if args.dry_run:
        log("done", "dry run - nothing rendered")
        return {"url": url, "status": "skipped", "folder": outdir.name,
                "note": "dry run"}

    results = []
    for c in clips:
        if args.only and c["index"] != args.only:
            continue
        idx = c["index"]
        tag = f"{idx:02d}"
        position = clips.index(c) + 1
        log("clip", f"[{position}/{len(clips)}]  #{idx}  "
                    f"{fmt_ts(c['start_t'])} -> {fmt_ts(c['end_t'])} "
                    f"({c['duration']:.1f}s)")

        final_path = outdir / f"{tag}.mp4"
        if final_path.exists() and not args.force:
            ok, why = verify(final_path, probe_seconds=1.5)
            if ok:
                log("clip", f"#{idx} already rendered - skipping")
                results.append({**c, "output": str(final_path), "framing": []})
                continue
            log("clip", f"#{idx} previous render is damaged ({why}) - redoing")
            final_path.unlink(missing_ok=True)

        try:
            cut_path = workdir / "cuts" / f"{tag}.mp4"
            if cut_path.exists() and not args.force and verify(cut_path, 1.0)[0]:
                raw = cut_path
            else:
                raw = cut_clip(Path(meta["path"]), c["start_t"], c["end_t"],
                               cut_path, cfg)

            words = words_in_range(transcript, c["start_t"], c["end_t"])

            # Emphasis has to be measured BEFORE the clip is tightened, because
            # the speed ramp picks its pause relative to a hard-hit word. Marking
            # afterwards leaves `tighten` with unflagged words and the ramp
            # silently never fires.
            start_t = c["start_t"]
            rel = [dict(w, start=w["start"] - start_t, end=w["end"] - start_t)
                   for w in words]
            rel = mark_emphasis(raw, rel, cfg)
            words = [dict(w, start=w["start"] + start_t, end=w["end"] + start_t)
                     for w in rel]

            # Jump-cut the dead air out before anything looks at the footage,
            # so framing and captions are planned against the final timeline.
            paced, tmap = tighten(raw, workdir / "cuts" / f"{tag}.tight.mp4",
                                  words, start_t, cfg)
            words = retime_words(words, tmap, start_t)
            c["tight_duration"] = round(tmap.kept_duration, 2)

            analysis = analyze(paced, cfg,
                               cache_path=workdir / "cuts" / f"{tag}.analysis.json",
                               force=args.force)
            frame_plan = plan_framing(analysis, cfg,
                                      int(cfg["output"]["width"]),
                                      int(cfg["output"]["height"]))

            ass_path = None
            if cfg["subtitles"].get("enabled", True):
                clip_cfg = {**cfg, "subtitles": {
                    **cfg["subtitles"],
                    "margin_v": caption_margin(frame_plan, cfg,
                                               int(cfg["output"]["width"]),
                                               int(cfg["output"]["height"]))}}
                ass_path = build_ass(words, clip_cfg,
                                     int(cfg["output"]["width"]),
                                     int(cfg["output"]["height"]),
                                     workdir / "cuts" / f"{tag}.ass",
                                     clip_start=0.0,
                                     hook_text=c.get("hook") or c.get("title"),
                                     language=transcript.get("language"))

            # Sound design and the shake that goes with it are planned from the
            # same events, so the picture kicks exactly when the hit lands.
            events = sfx.plan_events(frame_plan["decisions"], words,
                                     tmap.kept_duration, cfg)
            bed_path = sfx.write_bed(events, tmap.kept_duration, cfg,
                                     workdir / "cuts" / f"{tag}.sfx.wav")
            impacts = [e["t"] + 0.04 for e in events if e["kind"] == "impact"]
            frame_plan = apply_shake(frame_plan, impacts, cfg)

            timeline = effects.plan_effects(
                frame_plan["decisions"], impacts, frame_plan["fps"],
                frame_plan["n_frames"], cfg)

            final = render_reel(paced, frame_plan, ass_path, final_path, cfg,
                                bed_path=bed_path, timeline=timeline)
            results.append({**c, "output": str(final),
                            "framing": frame_plan["decisions"]})

            if args.upload:
                maybe_upload(Path(final), c, meta, cfg)
        except Exception as exc:
            log("clip", f"#{idx} FAILED: {exc}")
            traceback.print_exc(limit=3)

    write_json(outdir / "report.json",
               {"video": meta, "clips": results})
    if results:
        details.write_details(outdir, results, meta, cfg, transcript)

    print()
    log("done", f"{len(results)}/{len(clips)} reels written")
    print(f"    folder: {outdir}")
    total_mb = 0.0
    for r in results:
        path = Path(r["output"])
        mb = path.stat().st_size / 1e6 if path.exists() else 0.0
        total_mb += mb
        modes = ", ".join(sorted({d["mode"] for d in r["framing"]})) or "-"
        print(f"    {path.name:<10} {r['duration']:>5.1f}s  {mb:>5.1f} MB  "
              f"[{modes}]")
    print(f"    {'total':<10} {sum(r['duration'] for r in results):>5.0f}s  "
          f"{total_mb:>5.1f} MB")

    if args.clean:
        freed = batch.cleanup_source(meta)
        if freed:
            log("clean", f"removed the source video ({freed:.2f} GB freed)")

    return {"url": url, "status": "done" if results else "failed",
            "folder": outdir.name, "reels": len(results),
            "note": "" if results else "nothing rendered"}


def decide_existing(url: str, args) -> str:
    """keep / redo for a link whose reels already exist.

    A batch left running overnight must never sit on a question, so the flags
    win outright and the prompt only appears when neither was given.
    """
    done, folder, count = batch.already_done(url)
    if not done:
        return "redo"
    if args.skip_existing:
        log("batch", f"already done ({count} reels in {folder}) - skipping")
        return "keep"
    if args.redo or args.force:
        log("batch", f"already done ({count} reels) - rebuilding")
        return "redo"

    print()
    print(f"    This video already has reels:")
    print(f"      folder : {folder}")
    print(f"      reels  : {count}")
    try:
        answer = input("    Make them again? [y/N]: ").strip().lower()
    except EOFError:
        answer = ""
    return "redo" if answer in ("y", "yes") else "keep"


def main(argv=None) -> int:
    _use_utf8_console()
    args = parse_args(argv)
    ensure_dirs()
    cfg = apply_overrides(load_config(args.config), args)

    links = batch.read_links(args.urls, args.links)
    if not links:
        log("batch", "no links given. Pass URLs, or --links links.txt")
        return 1

    encoder, _ = pick_encoder(cfg["output"].get("video_encoder", "auto"))
    log("setup", f"video encoder: {encoder}")
    log("batch", f"{len(links)} link(s) queued")

    free, needed = batch.disk_report(len(links))
    log("batch", f"disk: {free:.0f} GB free, roughly {needed:.0f} GB needed"
                 + ("" if free > needed * 1.5 else "  <-- tight, consider --clean"))

    started = datetime.now()
    results: list[dict] = []

    with batch.KeepAwake():
        for i, url in enumerate(links, 1):
            print()
            print("=" * 68)
            log("batch", f"[{i}/{len(links)}]  {url}")
            print("=" * 68)

            if decide_existing(url, args) == "keep":
                done, folder, count = batch.already_done(url)
                results.append({"url": url, "status": "skipped",
                                "folder": folder, "reels": count,
                                "note": "already had reels"})
                continue

            try:
                results.append(process_one(url, cfg, args))
            except KeyboardInterrupt:
                log("batch", "stopped by you")
                results.append({"url": url, "status": "failed",
                                "note": "interrupted"})
                break
            except Exception as exc:
                # One bad link must not end the night.
                log("batch", f"FAILED: {exc}")
                traceback.print_exc(limit=3)
                results.append({"url": url, "status": "failed",
                                "note": str(exc)[:160]})

    summary = batch.write_summary(results, started)
    ok = [r for r in results if r["status"] == "done"]
    reels = sum(r.get("reels", 0) for r in ok)

    print()
    print("=" * 68)
    log("batch", f"finished: {len(ok)}/{len(links)} videos, {reels} reels, "
                 f"{(datetime.now() - started).total_seconds() / 60:.0f} min")
    print(f"    summary: {summary}")
    print("=" * 68)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
