# reels-automation

Turn a long video into ready-to-post vertical Reels/Shorts — automatically.

One command in, a folder of finished 1080×1920 clips out: chosen moments, adaptive
reframing that keeps faces in shot, burned-in word-by-word captions, jump-cut
pacing, punch-in zooms and transitions.

Runs entirely on your own machine. No account, no upload, no API key required.

---

## Quick start (Windows)

```bat
git clone https://github.com/<you>/reels-automation.git
cd reels-automation

python -m venv venv
venv\Scripts\python.exe -m pip install -r requirements.txt
venv\Scripts\python.exe get_models.py
```

You also need **ffmpeg** and **ffprobe** on your PATH — <https://ffmpeg.org/download.html>
(or drop `ffmpeg.exe` and `ffprobe.exe` into a `bin/` folder here and they will be
found automatically).

Then either double-click **`make-reels.bat`**, or:

```bat
venv\Scripts\python.exe run.py "https://www.youtube.com/watch?v=..." --clips 12
```

Reels land in `output/0001 - Video Title/01.mp4`, `02.mp4`, …

---

## What it does

| Stage | File | What happens |
|---|---|---|
| 1 | `reels/download.py` | yt-dlp download, resumable, decode-verified |
| 2 | `reels/captions.py` | YouTube's own captions (word-level timing, instant) |
| 2b | `reels/transcribe.py` | faster-whisper fallback — chunked and resumable |
| 3 | `reels/select.py` | picks the moments, spread across the whole video |
| 4 | `reels/pacing.py` | cuts dead air into jump cuts; one slow-motion beat |
| 5 | `reels/analyze.py` | scene split, then classifies what each scene contains |
| 6 | `reels/framing.py` | per-scene crop path — the core of the project |
| 7 | `reels/emphasis.py` | finds the words the speaker actually hits hard |
| 8 | `reels/subtitles.py` | ASS captions with per-word pop and accent colour |
| 9 | `reels/sfx.py` | synthesised whoosh / impact / riser (off by default) |
| 10 | `reels/effects.py` | zoom-blur transitions, flash, glitch |
| 11 | `reels/render.py` | reframe every frame, burn captions, hardware encode |

### Adaptive framing

A 16:9 source cropped blindly to 9:16 loses a third of the width, and whoever was
standing there with it. So every scene is classified first, then framed:

| Scene | Mode | Crop |
|---|---|---|
| `talking_head` | `track` | 31.6% — full 9:16, fills the screen, follows the subject |
| `multi_person` | `track` or wider | tight if the group fits, wider if not |
| `wide_subject` / `action` | `wide` | 45% — gentle 4:5 crop, letterboxed |
| `screen_ui` / `static_wide` | `fit` | 100% — nothing cropped |

Guarantees the code actually enforces:

- Every frame is checked **geometrically** before cropping. If the content will not
  fit, the mode widens. Content is never trimmed to tidy the composition.
- The crop path is clamped into the safe range each step, horizontally **and**
  vertically — a punch-in never takes the top off someone's head.
- Panning is speed-limited. A subject moving faster than the camera can calmly
  follow makes the frame widen instead of whipping after them.
- Scene boundaries are hard cuts; the camera never pans across one.
- Weak evidence → the wider frame wins.

---

## Many links at once

Queue a night's work and read the summary in the morning.

```bat
venv\Scripts\python.exe run.py --links links.txt --clips 12 --skip-existing --clean
```

`links.txt` is one URL per line; blank lines and `#comments` are ignored, and a
URL is picked out of surrounding text so a pasted list works. Duplicates are
collapsed, and playlist/radio parameters are stripped.

Built to be left alone:

- Windows sleep is disabled for the run and restored afterwards.
- A link that fails does not stop the batch — it is recorded and the run moves on.
- `--skip-existing` passes over videos that already have reels, so nothing waits
  on a question. Without it you are asked; with `--redo` they are rebuilt.
- `--clean` deletes each source video once its reels exist, about 1 GB back per
  video.
- `output/batch-summary.txt` says what was done, skipped and failed.

## Options

| Flag | Effect |
|---|---|
| `--clips 12` | how many reels to make |
| `--min 20 --max 90` | reel length bounds |
| `--dry-run` | pick moments, render nothing (fast preview) |
| `--only 3` | render just clip 3 |
| `--no-subs` | skip captions |
| `--source whisper` | ignore YouTube captions, transcribe locally |
| `--lang hi` | force the transcript language |
| `--force` | ignore every cache and redo |
| `--links f.txt` | read URLs from a file, one per line |
| `--skip-existing` | pass over videos that already have reels |
| `--redo` | rebuild videos that already have reels |
| `--clean` | delete each source video after its reels are made |

**How many reels can one video give?** Roughly one per minute of source. Measured:
a 10-minute video tops out near 10, a 43-minute video near 40. Ask for more and
you simply get what exists — clips cannot overlap.

---

## Tuning

Everything lives in `config.json`.

```jsonc
"select":  { "clips": 12, "target_seconds": 45, "spread": true }
"framing": { "edge_margin": 0.05, "object_max_width": 0.5, "pad_top_ratio": 0.32 }
"motion":  { "punch_zoom": 0.76, "shake_amplitude": 0.024 }
"effects": { "glitch_strength": 1.5, "flash_amount": 0.68 }
"sfx":     { "enabled": false }          // synthesised sound design, off by default
"grade":   { "saturation": 1.28, "contrast": 1.12 }
"output":  { "quality": 23, "max_fps": 30 }   // raise quality → smaller files
```

Notable ones:

- `framing.object_max_width` — an object box wider than this is scenery, not a
  subject. Without it a car filling the frame forces every scene to full width.
- `framing.edge_margin` — a face centred in the outer 5% of frame is treated as a
  bystander. Set to `0` to keep everyone, at the cost of wider framing.
- `output.quality` — 23 is generous. 27 roughly halves file size.
- `output.max_fps` — 60 fps sources are dropped to 30. Doubling the frame rate
  doubles the entire pipeline for no benefit in a vertical feed.

---

## Optional: smarter clip selection

Without a key, moments are chosen offline from audio energy, speech rate and cue
words. That works, and costs nothing.

With an Anthropic API key the transcript is read by Claude, which picks moments by
meaning — hooks, punchlines, stories — and writes an on-screen hook for each.
Roughly ₹3–11 (US$0.04–0.13) per video.

```bat
echo sk-ant-YOUR-KEY > .anthropic_key
```

Nothing else changes; the file is gitignored.

---

## Optional: upload straight to YouTube

`--upload` asks, for each finished reel, whether to publish it. Nothing is sent
without a typed `y`.

Set up once:

1. <https://console.cloud.google.com> → new project
2. **APIs & Services → Library** → enable *YouTube Data API v3*
3. **OAuth consent screen** → External, add yourself as a test user
4. **Credentials → Create credentials → OAuth client ID → Desktop app**
5. Download the JSON and save it here as `client_secret.json`

```bat
venv\Scripts\python.exe run.py "URL" --clips 12 --upload
```

Two limits to know:

- An upload costs **1600 of the 10,000 daily quota units** — about **6 uploads a
  day** on a default project. The tool tracks this and stops rather than failing.
- A project that has not passed Google's API audit has uploads **forced to
  private**, whatever privacy is requested. Make them public in YouTube Studio,
  or request an audit at
  <https://support.google.com/youtube/contact/yt_api_form>.

Uploaded files are remembered in `.youtube_uploads.json`, so re-running never
double-posts.

**Before automating this:** clips cut from someone else's video can earn
copyright strikes on your channel, and three strikes deletes it. The prompt
exists so every upload is a deliberate choice.

## Requirements

- Python 3.12–3.14
- ffmpeg + ffprobe
- ~2 GB disk for dependencies and models
- Any modern CPU. A GPU is not required — hardware encoding is used when present
  (`h264_qsv` / `h264_nvenc` / `h264_amf`), otherwise `libx264`.

---

## Legal

Only feed this videos you own or are licensed to reuse. Downloading someone
else's video may breach YouTube's Terms of Service, and reposting it can be
copyright infringement. This tool does not check, and the responsibility is
entirely the user's.

The vision models are downloaded from Google (MediaPipe) and OpenCV at setup and
are not redistributed here. All sound effects are synthesised from noise and sine
waves, so no samples are bundled or licensed.

## Author

Built by **Vipul Solanki**.

- Telegram — [@the_vipul_solanki](https://t.me/the_vipul_solanki)
- Instagram — [@thevipulsolanki](https://instagram.com/thevipulsolanki)

Issues and pull requests welcome.

## Licence

MIT — see [LICENSE](LICENSE).
