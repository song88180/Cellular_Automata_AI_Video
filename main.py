#!/usr/bin/env python3
"""Cellular automata, in your terminal -- an ASCII-art science short.

  python3 main.py                        play in the terminal (space, <-/->, esc)
  python3 main.py --chapter chapter4     play one chapter (comma-separate several)
  python3 main.py --render out.mp4       export 1920x1080 mp4 with soundtrack
  python3 main.py --sheet s.png          contact sheet (--every 2 --start 0 --end 30)
  python3 main.py --still 12.5 --out f.png

The first run computes every simulation once (progress is shown) and caches
the result in build/cache/, so later runs start instantly and seeking is free.
"""
import argparse
import sys
import time
from pathlib import Path

from chapters.registry import CHAPTER_ORDER, CHAPTERS
from engine.canvas import COLS, ROWS
from engine.timeline import TICK_FPS, load_or_build_chapters


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chapter", default="all", help=f"one of {CHAPTER_ORDER}, comma-separated, or 'all'")
    p.add_argument("--render", metavar="MP4", help="export video instead of playing")
    p.add_argument("--srt", metavar="SRT", help="also write subtitles (.srt)")
    p.add_argument("--sheet", metavar="PNG", help="write a contact sheet instead of playing")
    p.add_argument("--every", type=float, default=2.0, help="contact sheet interval (s)")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=None)
    p.add_argument("--still", type=float, metavar="SEC", help="write one full-size frame (with --out)")
    p.add_argument("--out", default="still.png")
    p.add_argument("--at", type=float, default=0.0, help="start playback at this time (s)")
    p.add_argument("--mute", action="store_true", help="no soundtrack")
    p.add_argument("--no-glow", action="store_true", help="disable bloom in rendered images")
    p.add_argument("--no-cache", action="store_true", help="recompute even if a cached film exists")
    return p.parse_args(argv)


def _progress_printer():
    t0 = time.time()

    def cb(key: str, secs: float, done: int, total: int) -> None:
        if total == 0:
            return
        msg = (f"首次运行，正在预计算所有模拟 / computing simulations (cached afterwards): "
               f"{done}/{total} chapters")
        if key:
            msg += f"  [{key} {secs:.0f}s]"
        sys.stderr.write(f"\r{msg}  {time.time() - t0:4.0f}s ")
        sys.stderr.flush()
    return cb


def main(argv=None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    keys = CHAPTER_ORDER if args.chapter == "all" else args.chapter.split(",")
    for k in keys:
        if k not in CHAPTERS:
            print(f"unknown chapter {k!r}; available: {CHAPTER_ORDER}", file=sys.stderr)
            return 2

    film, film_path = load_or_build_chapters(keys, CHAPTERS, ROWS, COLS, use_cache=not args.no_cache,
                                             progress=_progress_printer())
    sys.stderr.write("\r" + " " * 110 + "\r")

    if args.srt:
        from engine.export import export_srt
        export_srt(film, Path(args.srt))

    if args.sheet:
        from engine.export import contact_sheet
        contact_sheet(film, Path(args.sheet), every=args.every, start=args.start, end=args.end,
                      glow=not args.no_glow)
        print(f"wrote {args.sheet} ({film.duration:.1f}s film)")
        return 0
    if args.still is not None:
        from engine.export import still
        still(film, args.still, Path(args.out), glow=not args.no_glow)
        print(f"wrote {args.out}")
        return 0

    wav = None
    pcm = None
    if not args.mute:
        from engine.audio import soundtrack_for
        wav, pcm, sr = soundtrack_for(film, film_path)

    if args.render:
        from engine.export import render_mp4

        def prog(done, total):
            sys.stderr.write(f"\rrendering {done}/{total} frames ({100 * done / total:4.1f}%)")
            sys.stderr.flush()
        render_mp4(film_path, Path(args.render), wav, glow=not args.no_glow, progress=prog)
        sys.stderr.write("\n")
        print(f"wrote {args.render} ({film.duration:.1f}s)")
        return 0

    from engine.player import Player
    endless = None
    if keys[-1] == "chapter7":
        from chapters.chapter7_outro import ENDLESS_FPS, endless_frames
        endless = lambda: endless_frames(ROWS, COLS)  # noqa: E731
    Player(film, pcm, endless=endless, endless_fps=ENDLESS_FPS if endless else 12).play(int(args.at * TICK_FPS))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
