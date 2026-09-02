"""Write upload-details.txt for reels that were rendered before this existed.

    python regenerate-details.py
"""
from __future__ import annotations

import sys
from pathlib import Path

from reels.details import write_details
from reels.util import OUTPUT, load_config, log, read_json

cfg = load_config()
made = 0

for folder in sorted(p for p in OUTPUT.iterdir() if p.is_dir()):
    report_path = folder / "report.json"
    if not report_path.exists():
        log("details", f"{folder.name}: no report.json, skipping")
        continue

    report = read_json(report_path)
    meta = report.get("video", {})
    clips = report.get("clips", [])
    if not clips:
        continue

    transcript = None
    tpath = Path(meta.get("workdir", "")) / "transcript.json"
    if tpath.exists():
        transcript = read_json(tpath)

    write_details(folder, clips, meta, cfg, transcript)
    made += 1

print(f"\n{made} upload sheet(s) written.")
sys.exit(0)
