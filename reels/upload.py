"""Stage 8 - upload a finished reel to YouTube as a Short.

Setup, once:

  1. https://console.cloud.google.com  ->  new project
  2. APIs & Services  ->  Library  ->  enable "YouTube Data API v3"
  3. APIs & Services  ->  OAuth consent screen  ->  External, add yourself
     as a test user
  4. Credentials  ->  Create credentials  ->  OAuth client ID  ->  Desktop app
  5. Download the JSON, save it here as `client_secret.json`

The first upload opens a browser to sign in; the refresh token is then kept in
`.youtube_token.json` and no browser is needed again.

Two limits worth knowing before relying on this:

  * Each upload costs 1600 of the 10,000 daily quota units, so roughly **six
    uploads a day** on a default project.
  * A project that has not passed Google's API audit has every upload forced to
    **private**, whatever privacy status is requested. Requesting public is
    still correct - it takes effect the moment the audit passes - but until then
    the video lands private and this module says so plainly.
"""
from __future__ import annotations

import json
import re
from datetime import date
from pathlib import Path

from .util import ROOT, log

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]
CLIENT_SECRET = ROOT / "client_secret.json"
TOKEN_FILE = ROOT / ".youtube_token.json"
LEDGER = ROOT / ".youtube_uploads.json"

UPLOAD_COST = 1600            # quota units charged per videos.insert
DAILY_QUOTA = 10000


def _ledger() -> dict:
    if LEDGER.exists():
        try:
            return json.loads(LEDGER.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"day": "", "used": 0, "done": {}}


def _save_ledger(data: dict) -> None:
    LEDGER.write_text(json.dumps(data, ensure_ascii=False, indent=2),
                      encoding="utf-8")


def quota_left() -> int:
    data = _ledger()
    if data.get("day") != date.today().isoformat():
        return DAILY_QUOTA
    return max(0, DAILY_QUOTA - int(data.get("used", 0)))


def already_uploaded(reel: Path) -> str | None:
    """The video id if this exact file was uploaded before, else None."""
    return _ledger().get("done", {}).get(str(reel.resolve()))


def _record(reel: Path, video_id: str) -> None:
    data = _ledger()
    today = date.today().isoformat()
    if data.get("day") != today:
        data["day"], data["used"] = today, 0
    data["used"] = int(data.get("used", 0)) + UPLOAD_COST
    data.setdefault("done", {})[str(reel.resolve())] = video_id
    _save_ledger(data)


def _credentials():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    creds = None
    if TOKEN_FILE.exists():
        try:
            creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
        except Exception:
            creds = None

    if creds and creds.valid:
        return creds
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
        return creds

    if not CLIENT_SECRET.exists():
        raise RuntimeError(
            f"{CLIENT_SECRET.name} not found. Create an OAuth client ID "
            f"(Desktop app) in Google Cloud, download the JSON, and save it as "
            f"{CLIENT_SECRET}")

    log("upload", "opening a browser to sign in to YouTube (first time only)")
    flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET), SCOPES)
    creds = flow.run_local_server(port=0)
    TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")
    log("upload", f"signed in - token saved to {TOKEN_FILE.name}")
    return creds


def build_metadata(clip: dict, video_title: str, cfg: dict) -> dict:
    """Title, description and tags for one reel."""
    ucfg = cfg.get("upload", {})
    hook = (clip.get("hook") or clip.get("title") or "").strip()
    hook = re.sub(r"\s+", " ", hook)

    title = hook or video_title
    suffix = ucfg.get("title_suffix", " #Shorts")
    # YouTube hard-caps titles at 100 characters.
    title = title[: 100 - len(suffix)].rstrip(" .,-") + suffix

    tags = list(ucfg.get("tags", ["shorts", "reels", "viral"]))
    lines = [hook] if hook else []
    if ucfg.get("credit_source", True):
        lines += ["", f"From: {video_title}"]
    if ucfg.get("description_footer"):
        lines += ["", ucfg["description_footer"]]
    lines += ["", " ".join(f"#{t}" for t in tags[:15])]

    return {
        "snippet": {
            "title": title,
            "description": "\n".join(lines)[:4900],
            "tags": tags[:30],
            "categoryId": str(ucfg.get("category_id", 22)),
        },
        "status": {
            "privacyStatus": ucfg.get("privacy", "public"),
            "selfDeclaredMadeForKids": False,
        },
    }


def upload(reel: Path, clip: dict, video_title: str, cfg: dict) -> str | None:
    """Upload one reel. Returns the video id, or None if it was skipped."""
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload

    existing = already_uploaded(reel)
    if existing:
        log("upload", f"{reel.name} was already uploaded ({existing}) - skipping")
        return existing

    left = quota_left()
    if left < UPLOAD_COST:
        log("upload", f"daily quota spent ({left} units left, an upload needs "
                      f"{UPLOAD_COST}). Try again tomorrow.")
        return None

    body = build_metadata(clip, video_title, cfg)
    wanted = body["status"]["privacyStatus"]
    log("upload", f"uploading {reel.name} as \"{body['snippet']['title']}\"")

    youtube = build("youtube", "v3", credentials=_credentials(),
                    cache_discovery=False)
    media = MediaFileUpload(str(reel), chunksize=4 * 1024 * 1024, resumable=True,
                            mimetype="video/mp4")
    request = youtube.videos().insert(part="snippet,status", body=body,
                                      media_body=media)

    from .util import progress

    response = None
    try:
        while response is None:
            status, response = request.next_chunk()
            if status:
                progress("upload", status.resumable_progress,
                         status.total_size or 1, reel.name)
        progress("upload", 1, 1, reel.name)
    except HttpError as exc:
        log("upload", f"failed: {exc}")
        return None

    video_id = response.get("id")
    got = (response.get("status") or {}).get("privacyStatus", "?")
    _record(reel, video_id)

    log("upload", f"done -> https://youtu.be/{video_id}")
    if got != wanted:
        log("upload", f"NOTE: asked for '{wanted}' but YouTube set '{got}'. "
                      f"An un-audited API project forces uploads private - "
                      f"make it public in YouTube Studio, or request an audit "
                      f"at https://support.google.com/youtube/contact/yt_api_form")
    remaining = quota_left() // UPLOAD_COST
    log("upload", f"{remaining} more upload(s) possible today")
    return video_id
