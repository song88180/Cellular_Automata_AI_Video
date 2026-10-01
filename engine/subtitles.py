"""Bilingual subtitle formatting and SRT export."""
import unicodedata
from typing import List, Tuple

Cue = Tuple[float, float, str, str]  # (start_sec, end_sec, zh, en)


def display_width(s: str) -> int:
    width = 0
    for ch in s:
        width += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return width


def truncate_to_width(s: str, width: int) -> str:
    if display_width(s) <= width:
        return s
    out = []
    w = 0
    for ch in s:
        cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
        if w + cw > width:
            break
        out.append(ch)
        w += cw
    return "".join(out)


def center_line(s: str, width: int) -> str:
    s = truncate_to_width(s, width)
    pad = width - display_width(s)
    left = pad // 2
    right = pad - left
    return " " * left + s + " " * right


def format_ts(t: float) -> str:
    # Round the *total* to ms first; rounding the fraction alone turns
    # 22.9996 into "22,1000", which SRT parsers reject.
    total_ms = int(round(max(0.0, t) * 1000))
    s, ms = divmod(total_ms, 1000)
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def parse_ts(s: str) -> float:
    hms, ms = s.strip().split(",")
    h, m, sec = (int(x) for x in hms.split(":"))
    return h * 3600 + m * 60 + sec + int(ms) / 1000.0


def build_srt(cues: List[Cue]) -> str:
    lines = []
    n = 0
    for start, end, zh, en in cues:
        text = "\n".join(t for t in (zh, en) if t)
        if not text:
            continue
        n += 1
        lines.append(str(n))
        lines.append(f"{format_ts(start)} --> {format_ts(end)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def export_srt(cues: List[Cue], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(build_srt(cues))
