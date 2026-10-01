"""mp4 export, preview stills and contact sheets for a Film."""
import os
import subprocess
from multiprocessing import get_context
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .raster import MONO, SUB_FADE_TICKS, VIDEO_H, VIDEO_W, film_image
from .timeline import TICK_FPS, Film

_FILM: Optional[Film] = None


def _init_worker(path: str) -> None:
    global _FILM
    _FILM = Film.load(Path(path))


def _render_chunk(args) -> bytes:
    t0, t1, glow = args
    out, prev_key, prev = [], None, b""
    for t in range(t0, t1):
        # held frames (same picture, settled subtitles) are rendered once
        now, before = _FILM.subtitle(t), _FILM.subtitle(max(0, t - SUB_FADE_TICKS - 1))
        key = (_FILM.frame_index(t), now, _FILM.chapter_at(t), t if (now != before or t <= SUB_FADE_TICKS) else -1)
        if key != prev_key:
            prev = film_image(_FILM, t, glow).tobytes()
            prev_key = key
        out.append(prev)
    return b"".join(out)


def render_mp4(film_path: Path, out: Path, audio_wav: Optional[Path] = None, glow: bool = True,
               workers: int = 0, progress=None) -> None:
    film = Film.load(film_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{VIDEO_W}x{VIDEO_H}", "-r", str(TICK_FPS), "-i", "-"]
    if audio_wav is not None:
        cmd += ["-i", str(audio_wav)]
    fade = min(1.5, film.duration / 4)   # the video ends with a fade to black
    # starts from pure black (the film's first second is an empty frame) and ends with a fade to black
    cmd += ["-vf", f"fade=t=in:st=0:d=1,fade=t=out:st={film.duration - fade:.3f}:d={fade:.3f}"]
    cmd += ["-c:v", "libx264", "-preset", "medium", "-crf", "18", "-pix_fmt", "yuv420p",
            "-tune", "animation", "-movflags", "+faststart"]
    if audio_wav is not None:
        cmd += ["-c:a", "aac", "-b:a", "192k", "-shortest"]
    cmd.append(str(out))
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    chunk = 24
    jobs = [(t, min(t + chunk, film.n_ticks), glow) for t in range(0, film.n_ticks, chunk)]
    workers = workers or max(1, min(12, (os.cpu_count() or 2) - 2))
    ctx = get_context("fork")
    with ctx.Pool(workers, initializer=_init_worker, initargs=(str(film_path),)) as pool:
        for i, data in enumerate(pool.imap(_render_chunk, jobs)):
            proc.stdin.write(data)
            if progress:
                progress(jobs[i][1], film.n_ticks)
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")


def still(film: Film, seconds: float, out: Path, glow: bool = True) -> None:
    tick = int(seconds * TICK_FPS)
    Image.fromarray(film_image(film, tick, glow)).save(out)


def contact_sheet(film: Film, out: Path, every: float = 2.0, cols: int = 4, width: int = 480,
                  start: float = 0.0, end: Optional[float] = None, glow: bool = False) -> None:
    """Grid of downscaled frames, each labelled with its timestamp and scene."""
    end = film.duration if end is None else min(end, film.duration)
    times = list(np.arange(start, end, every))
    h = int(width * VIDEO_H / VIDEO_W)
    rows = (len(times) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * width, rows * (h + 18)), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(MONO, 13)
    for i, t in enumerate(times):
        tick = int(t * TICK_FPS)
        im = Image.fromarray(film_image(film, tick, glow)).resize((width, h), Image.LANCZOS)
        x, y = (i % cols) * width, (i // cols) * (h + 18)
        sheet.paste(im, (x, y + 18))
        d.text((x + 4, y + 2), f"{t:6.1f}s  {film.scene_at(tick).name}", fill=(200, 200, 200), font=font)
    sheet.save(out)


def export_srt(film: Film, path: Path) -> None:
    from .subtitles import build_srt
    Path(path).write_text(build_srt(film.cues()), encoding="utf-8")
