"""Stage 6 - burn-in captions as an ASS file with word-by-word highlighting.

One Dialogue event is emitted per word: the whole line is drawn every time, with
only the currently-spoken word recoloured and slightly enlarged. That is verbose
but exact, and it survives any renderer that can read ASS.
"""
from __future__ import annotations

from pathlib import Path

ASS_HEADER = """[Script Info]
ScriptType: v4.00+
PlayResX: {w}
PlayResY: {h}
WrapStyle: 2
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.709

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Cap,{font},{size},{primary},{highlight},{outline_col},&H64000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},2,70,70,{margin_v},1
Style: Hook,{font},{hook_size},{primary},{primary},{outline_col},&H64000000,-1,0,0,0,100,100,0,0,1,{outline},{shadow},8,80,80,{hook_margin},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, Effect, Text
"""


# Whisper language codes whose scripts Arial Black cannot render at all.
INDIC_LANGS = {"hi", "gu", "mr", "bn", "ta", "te", "kn", "ml", "pa", "or",
               "as", "ne", "si", "sa"}
CJK_LANGS = {"zh", "ja", "ko"}
RTL_LANGS = {"ar", "fa", "ur", "he"}


def font_for_language(language: str | None, cfg: dict) -> tuple[str, bool]:
    """Pick a font that can actually draw the script. Returns (font, uppercase_ok).

    Uppercasing is only meaningful for bicameral scripts, so it is switched off
    everywhere else.
    """
    scfg = cfg["subtitles"]
    lang = (language or "").lower()
    if lang in INDIC_LANGS:
        return scfg.get("font_indic", "Nirmala UI"), False
    if lang in CJK_LANGS:
        return scfg.get("font_cjk", "Microsoft YaHei"), False
    if lang in RTL_LANGS:
        return scfg.get("font_rtl", "Segoe UI"), False
    return scfg.get("font", "Arial Black"), bool(scfg.get("uppercase", True))


def caption_margin(plan: dict, cfg: dict, out_w: int, out_h: int) -> int:
    """ASS MarginV (distance from the frame bottom) that clears the video.

    A clip's scenes can use different crop widths, so the layout follows whichever
    mode holds the screen longest. Where the video is letterboxed the captions go
    just underneath it; where it fills the frame they sit near the bottom edge.
    """
    fcfg = cfg["framing"]
    scfg = cfg["subtitles"]
    pinned = scfg.get("margin_v", "auto")
    if isinstance(pinned, (int, float)):
        return int(pinned)

    src_h = plan.get("src_h") or out_h
    widths: dict[float, float] = {}
    for d in plan.get("decisions", []):
        widths[d["crop_width"]] = widths.get(d["crop_width"], 0.0) + (d["end"] - d["start"])
    if not widths:
        return 420
    crop_w = max(widths, key=lambda k: widths[k])

    fg_h = out_w * src_h / max(1.0, crop_w)
    if fg_h >= out_h:
        return 420                                  # video fills the frame

    pad_top = float(fcfg.get("pad_top_ratio", 0.32))
    top = (out_h - fg_h) * pad_top
    video_bottom = top + fg_h
    text_height = float(scfg.get("font_size", 78)) * 1.45
    margin = out_h - (video_bottom + 48 + text_height)
    return int(max(120, min(margin, out_h - text_height - 60)))


def _ts(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h:d}:{m:02d}:{s:05.2f}"


EMOJI_FONT = "Segoe UI Emoji"

# Emoji are OFF by default and should stay off while captions are drawn by
# libass: libass rasterises glyph outlines only, so a colour emoji font comes
# out as a flat monochrome shape with the caption's own outline around it -
# it reads as a smudge, not an emoji. Colour emoji would have to be composited
# as images in the frame loop instead. The mechanism below is kept for that.
#
# The table only fires on words it recognises, so it stays silent on languages
# it has no entries for. Add your own under config: subtitles.emoji_map.
DEFAULT_EMOJI = {
    "money": "💰", "rupees": "💰", "crore": "💰", "lakh": "💰",
    "free": "🎁", "fire": "🔥", "best": "🏆", "win": "🏆", "winner": "🏆",
    "love": "❤️", "crazy": "🤯", "insane": "🤯", "shocking": "😱",
    "secret": "🤫", "warning": "⚠️", "danger": "⚠️", "wrong": "❌",
    "yes": "✅", "right": "✅", "think": "🤔", "big": "📈", "huge": "📈",
    "fast": "⚡",
}

_PUNCT = ".,!?;:\"'()[]—–»«।"


def _emoji_for(word: str, table: dict) -> str | None:
    key = word.strip(_PUNCT).lower()
    return table.get(key) if key else None


def _escape(text: str) -> str:
    return (text.replace("\\", "/").replace("{", "(").replace("}", ")")
                .replace("\n", " ").strip())


def group_lines(words: list[dict], per_line: int = 3,
                gap_break: float = 0.55, max_line_seconds: float = 2.6):
    """Split words into caption lines on count, pauses, and sentence ends."""
    lines: list[list[dict]] = []
    current: list[dict] = []
    for i, w in enumerate(words):
        if current:
            gap = w["start"] - current[-1]["end"]
            span = w["end"] - current[0]["start"]
            ends_sentence = current[-1]["word"][-1:] in ".!?"
            if (len(current) >= per_line or gap > gap_break
                    or span > max_line_seconds or ends_sentence):
                lines.append(current)
                current = []
        current.append(w)
    if current:
        lines.append(current)
    return lines


def build_ass(words: list[dict], cfg: dict, out_w: int, out_h: int,
              path: Path, clip_start: float = 0.0,
              hook_text: str | None = None, hook_seconds: float = 4.0,
              language: str | None = None) -> Path:
    """Write an ASS subtitle file; `words` carry absolute source timestamps."""
    scfg = cfg["subtitles"]
    ocfg = cfg.get("overlay", {})
    font, upper = font_for_language(language, cfg)

    header = ASS_HEADER.format(
        w=out_w, h=out_h,
        font=font,
        size=scfg.get("font_size", 78),
        hook_size=ocfg.get("hook_font_size", 62),
        primary=scfg.get("primary_color", "&H00FFFFFF"),
        highlight=scfg.get("highlight_color", "&H0000E5FF"),
        outline_col=scfg.get("outline_color", "&H00000000"),
        outline=scfg.get("outline", 5),
        shadow=scfg.get("shadow", 2),
        margin_v=scfg.get("margin_v", 420),
        hook_margin=ocfg.get("hook_margin_top", 190),
    )

    events: list[str] = []

    if hook_text and ocfg.get("hook_enabled", True):
        text = _escape(hook_text)[: ocfg.get("hook_max_chars", 60)]
        events.append(
            f"Dialogue: 0,{_ts(0.0)},{_ts(hook_seconds)},Hook,,0,0,,"
            f"{{\\fad(250,350)}}{text}"
        )

    if scfg.get("enabled", True) and words:
        primary = scfg.get("primary_color", "&H00FFFFFF")
        highlight = scfg.get("highlight_color", "&H0000E5FF")
        accent = scfg.get("emphasis_color", highlight)
        outline_col = scfg.get("outline_color", "&H00000000")
        base_bord = int(scfg.get("outline", 6))
        emph_scale = int(scfg.get("emphasis_scale", 112))
        pop = int(scfg.get("pop_scale", 118))
        settle = int(scfg.get("settle_scale", 106))
        pop_ms = int(scfg.get("pop_ms", 70))
        emoji_table = {**DEFAULT_EMOJI, **(scfg.get("emoji_map") or {})}
        use_emoji = bool(scfg.get("emoji_enabled", False))

        for line in group_lines(words, int(scfg.get("words_per_line", 3))):
            tokens = []
            for w in line:
                token = _escape(w["word"])
                if upper:
                    token = token.upper()
                emoji = _emoji_for(w["word"], emoji_table) if use_emoji else None
                tokens.append((token, bool(w.get("emph")), emoji))

            for i, w in enumerate(line):
                start = max(0.0, w["start"] - clip_start)
                # Hold the last word of a line a beat longer so it does not flicker.
                if i + 1 < len(line):
                    end = max(start + 0.05, line[i + 1]["start"] - clip_start)
                else:
                    end = max(start + 0.18, w["end"] - clip_start + 0.12)

                parts = []
                for j, (token, is_emph, emoji) in enumerate(tokens):
                    if j == i:
                        # Active word snaps up then settles. \t times are relative
                        # to this Dialogue line, which begins on the word itself.
                        colour = accent if is_emph else highlight
                        bord = base_bord + (2 if is_emph else 0)
                        piece = (
                            f"{{\\c{colour}\\bord{bord}"
                            f"\\fscx{settle}\\fscy{settle}"
                            f"\\t(0,{pop_ms},\\fscx{pop}\\fscy{pop})"
                            f"\\t({pop_ms},{pop_ms * 2},"
                            f"\\fscx{settle}\\fscy{settle})}}{token}"
                            f"{{\\c{primary}\\bord{base_bord}"
                            f"\\fscx100\\fscy100}}"
                        )
                    elif is_emph:
                        # A stressed word stays bigger and keeps the accent fill,
                        # before and after it is spoken, so the line has shape.
                        # Recolouring the FILL, not the outline: a thick coloured
                        # border fills the gaps between letters and the word turns
                        # into a blob.
                        piece = (
                            f"{{\\c{accent}"
                            f"\\fscx{emph_scale}\\fscy{emph_scale}}}{token}"
                            f"{{\\c{primary}\\fscx100\\fscy100}}"
                        )
                    else:
                        piece = token
                    if emoji:
                        piece += f"{{\\fn{EMOJI_FONT}}}{emoji}{{\\fn{font}}}"
                    parts.append(piece)

                events.append(
                    f"Dialogue: 0,{_ts(start)},{_ts(end)},Cap,,0,0,,"
                    + " ".join(parts)
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig") as fh:
        fh.write(header)
        fh.write("\n".join(events))
        fh.write("\n")
    return path
