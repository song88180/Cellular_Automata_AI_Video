"""Interactive terminal player for a pre-computed Film.

- The 160 x 40 picture (+ subtitles + status line) is centred in whatever
  size the terminal is, and the whole screen is painted in the film's
  background colour, so it looks the same as the video.
- Keys: space = pause/play, <- / -> = back/forward 5 s, Esc or q = quit.
- Optional `endless` frames: after the film ends, keep showing a live
  simulation (the evolving 'thanks' card) until any key is pressed.
- Optional soundtrack via a PCM pipe to aplay / pw-play / paplay.
"""
import os
import select
import shutil
import subprocess
import sys
import termios
import threading
import time
import tty
from typing import Callable, Iterator, List, Optional

import numpy as np

from .canvas import BG, DIM, FAINT
from .frame import Frame, FrameBuffer, char_width
from .subtitles import center_line, display_width, truncate_to_width
from .timeline import TICK_FPS, Film

ESC = "\x1b"
SEEK_SECONDS = 5
EXTRA_ROWS = 4   # gap, zh subtitle, en subtitle, status line


def _move(row: int, col: int) -> str:
    return f"{ESC}[{row + 1};{col + 1}H"


def _fg(rgb) -> str:
    return f"{ESC}[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


def _bg(rgb) -> str:
    return f"{ESC}[48;2;{rgb[0]};{rgb[1]};{rgb[2]}m"


BG_SGR = _bg(BG)
RESET = f"{ESC}[0m"


class AudioOut:
    """Streams a slice of an int16 stereo PCM buffer to a system player."""

    PLAYERS = [
        ["aplay", "-q", "-t", "raw", "-f", "S16_LE", "-r", "{sr}", "-c", "2", "--buffer-time=80000", "-"],
        ["pw-play", "--rate", "{sr}", "--channels", "2", "--format", "s16", "-"],
        ["paplay", "--raw", "--rate={sr}", "--channels=2", "--format=s16le"],
    ]

    def __init__(self, pcm: Optional[np.ndarray], sr: int):
        self.pcm, self.sr = pcm, sr
        self.proc: Optional[subprocess.Popen] = None
        self.cmd = None
        if pcm is not None:
            for c in self.PLAYERS:
                if shutil.which(c[0]):
                    self.cmd = [a.format(sr=sr) for a in c]
                    break

    @property
    def available(self) -> bool:
        return self.cmd is not None

    def start(self, seconds: float) -> None:
        self.stop()
        if not self.available:
            return
        i0 = max(0, int(seconds * self.sr))
        data = self.pcm[i0:]
        try:
            self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.DEVNULL)
        except OSError:
            self.cmd = None
            return
        proc = self.proc

        def feed():
            step = self.sr // 10
            try:
                for j in range(0, len(data), step):
                    if proc.poll() is not None:
                        return
                    proc.stdin.write(data[j: j + step].tobytes())
                proc.stdin.close()
            except (BrokenPipeError, ValueError, OSError):
                pass

        threading.Thread(target=feed, daemon=True).start()

    def stop(self) -> None:
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(timeout=1)
            except Exception:
                pass
            self.proc = None


class Player:
    def __init__(self, film: Film, pcm: Optional[np.ndarray] = None, sr: int = 44100,
                 endless: Optional[Callable[[], Iterator[Frame]]] = None, endless_fps: int = 12):
        self.film = film
        self.endless, self.endless_fps = endless, endless_fps
        self.audio = AudioOut(pcm, sr)
        self.rows_needed = film.rows + EXTRA_ROWS
        self.buf = FrameBuffer(film.rows, film.cols)
        self.term_size = (0, 0)
        self.off = (0, 0)
        self.prev_subs = None
        self.prev_status = None
        self.fd = sys.stdin.fileno()

    # --- terminal geometry ---
    def _check_resize(self) -> bool:
        size = shutil.get_terminal_size((self.film.cols, self.rows_needed))
        if (size.columns, size.lines) == self.term_size:
            return False
        self.term_size = (size.columns, size.lines)
        self.off = (max(0, (size.lines - self.rows_needed) // 2), max(0, (size.columns - self.film.cols) // 2))
        sys.stdout.write(RESET + BG_SGR + f"{ESC}[2J")
        sys.stdout.flush()
        self.buf.reset()
        self.prev_subs = None
        self.prev_status = None
        return True

    def _fits(self) -> bool:
        return self.term_size[0] >= self.film.cols and self.term_size[1] >= self.rows_needed

    # --- drawing ---
    def _draw(self, tick: int, paused: bool) -> None:
        self._paint(self.film.frame(tick), self.film.subtitle(tick), self._status(tick, paused))

    def _paint(self, frame: Frame, subs, status: str) -> None:
        r0, c0 = self.off
        R, C = self.term_size[1], self.term_size[0]
        out: List[str] = []
        cur = None            # where the terminal cursor is now
        cur_fg = cur_bg = None
        for r, c, ch, fg, bg in self.buf.diff(frame):
            rr, cc = r + r0, c + c0
            if rr >= R or cc >= C:
                continue
            if cur != (rr, cc):
                out.append(_move(rr, cc))
            bgc = bg if bg is not None else BG
            if bgc != cur_bg:
                out.append(_bg(bgc))
                cur_bg = bgc
            if fg is not None and fg != cur_fg:
                out.append(_fg(fg))
                cur_fg = fg
            out.append(ch)
            cur = (rr, cc + char_width(ch))
        if subs != self.prev_subs:
            for k, (text, rgb) in enumerate(((subs[0], (240, 240, 245)), (subs[1], (160, 164, 180)))):
                row = r0 + self.film.rows + 1 + k
                if row < R:
                    out.append(_move(row, c0) + BG_SGR + _fg(rgb) + center_line(text, min(self.film.cols, C - c0)))
            self.prev_subs = subs
        if status != self.prev_status:
            row = r0 + self.film.rows + 3
            if row < R:
                out.append(_move(row, c0) + BG_SGR + status)
            self.prev_status = status
        if out:
            sys.stdout.write("".join(out))
            sys.stdout.flush()

    def _status(self, tick: int, paused: bool) -> str:
        W = min(self.film.cols, self.term_size[0] - self.off[1])
        t, T = tick / TICK_FPS, self.film.duration
        left = "space 暂停/播放 · ←/→ 5秒 · esc 退出"
        right = f"{'❚❚' if paused else '▶'} {int(t) // 60:02d}:{int(t) % 60:02d} / {int(T) // 60:02d}:{int(T) % 60:02d}"
        zh, en = self.film.chapter_at(tick)
        if zh:
            right = f"{zh} · {en}   " + right
        if not self.audio.available:
            right = "(no audio) " + right
        bar_w = max(0, W - display_width(left) - display_width(right) - 4)
        filled = int(round(bar_w * min(1.0, tick / max(1, self.film.n_ticks - 1))))
        bar = _fg((95, 100, 125)) + "━" * filled + _fg(FAINT) + "─" * (bar_w - filled)
        return (_fg(FAINT) + left + "  " + bar + "  " + _fg(DIM if paused else FAINT) + right
                + " " * max(0, W - display_width(left) - bar_w - display_width(right) - 4))

    def _endless_status(self) -> str:
        W = min(self.film.cols, self.term_size[0] - self.off[1])
        msg = "按任意键退出 · press any key to exit"
        return _fg(DIM) + msg + " " * max(0, W - display_width(msg))

    def _play_endless(self, interactive: bool) -> None:
        """Keep the last scene alive until any key is pressed."""
        if not interactive:        # no keyboard -> nothing could ever end it
            return
        frames = self.endless()
        dt = 1.0 / self.endless_fps
        t_next = time.monotonic()
        for frame in frames:
            self._check_resize()
            self._paint(frame, (frame.subtitle_zh, frame.subtitle_en), self._endless_status())
            t_next += dt
            wait = max(0.0, t_next - time.monotonic())
            r, _, _ = select.select([self.fd], [], [], wait)
            if r:
                os.read(self.fd, 64)   # any key (including escape sequences)
                return

    def _warn_small(self) -> None:
        C, R = self.term_size
        msg = [f"终端太小：{C}x{R}，需要至少 {self.film.cols}x{self.rows_needed}",
               f"Terminal is {C}x{R}; need {self.film.cols}x{self.rows_needed}.",
               "请全屏或缩小字体 (Ctrl -)；按任意键继续 / press any key to continue anyway"]
        out = [RESET + BG_SGR + f"{ESC}[2J"]
        for i, m in enumerate(msg):
            m = truncate_to_width(m, C)
            out.append(_move(R // 2 - 1 + i, max(0, (C - display_width(m)) // 2)) + _fg((230, 230, 235)) + m)
        sys.stdout.write("".join(out))
        sys.stdout.flush()

    # --- input ---
    def _read_key(self, timeout: float) -> Optional[str]:
        r, _, _ = select.select([self.fd], [], [], None if timeout is None else max(0.0, timeout))
        if not r:
            return None
        b = os.read(self.fd, 1)
        if b == b"\x1b":
            seq = b""
            while True:
                r, _, _ = select.select([self.fd], [], [], 0.03)
                if not r:
                    break
                seq += os.read(self.fd, 1)
                if len(seq) >= 2 and seq[-1:].isalpha() or len(seq) > 6:
                    break
            if seq in (b"[C", b"OC"):
                return "right"
            if seq in (b"[D", b"OD"):
                return "left"
            return "esc" if not seq else None
        if b == b" ":
            return "space"
        if b in (b"q", b"Q", b"\x03"):
            return "esc"
        if b in (b"p", b"P"):
            return "space"
        return None

    # --- main loop ---
    def play(self, start_tick: int = 0) -> None:
        old = termios.tcgetattr(self.fd) if sys.stdin.isatty() else None
        sys.stdout.write(f"{ESC}[?1049h{ESC}[?25l")
        try:
            if old is not None:
                tty.setraw(self.fd)
            self._check_resize()
            if not self._fits():
                self._warn_small()
                if old is not None:
                    self._read_key(None)
                self.term_size = (0, 0)
                self._check_resize()
            tick, paused = start_tick, False
            clock0, tick0 = time.monotonic(), tick
            self.audio.start(tick / TICK_FPS)
            last_drawn = None
            while True:
                if self._check_resize():
                    last_drawn = None
                if not paused:
                    tick = tick0 + int((time.monotonic() - clock0) * TICK_FPS)
                    if tick >= self.film.n_ticks:
                        if self.endless is not None:
                            self._play_endless(old is not None)
                        break
                if (tick, paused) != last_drawn:
                    self._draw(tick, paused)
                    last_drawn = (tick, paused)
                wait = 0.25 if paused else clock0 + (tick - tick0 + 1) / TICK_FPS - time.monotonic()
                key = self._read_key(wait) if old is not None else (time.sleep(max(0, wait)) or None)
                if key == "esc":
                    break
                if key in ("left", "right", "space"):
                    if key == "space":
                        paused = not paused
                    else:
                        d = SEEK_SECONDS * TICK_FPS * (1 if key == "right" else -1)
                        tick = max(0, min(self.film.n_ticks - 1, tick + d))
                    clock0, tick0 = time.monotonic(), tick
                    if paused:
                        self.audio.stop()
                    else:
                        self.audio.start(tick / TICK_FPS)
        finally:
            self.audio.stop()
            if old is not None:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, old)
            sys.stdout.write(RESET + f"{ESC}[?25h{ESC}[?1049l")
            sys.stdout.flush()
