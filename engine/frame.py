"""Core data model: Frame, Scene, and diff-based FrameBuffer."""
import unicodedata
from dataclasses import dataclass, field
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

RGB = Tuple[int, int, int]

# Right half of a double-width (CJK) glyph: the glyph itself lives in the cell
# to its left. Renderers draw nothing for it.
WIDE_CONT = "\ue000"  # private-use codepoint (numpy strips "\\0" from <U1 arrays)


def char_width(ch: str) -> int:
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


@dataclass
class Frame:
    chars: np.ndarray   # ('<U1', (rows, cols))
    colors: np.ndarray  # (uint8, (rows, cols, 3)) foreground RGB
    active: np.ndarray  # (bool, (rows, cols)) False -> plain space
    subtitle_zh: str = ""
    subtitle_en: str = ""
    bg: Optional[np.ndarray] = None         # (uint8, (rows, cols, 3)) background RGB
    bg_active: Optional[np.ndarray] = None  # (bool, (rows, cols)) False -> theme bg
    # Sound cues fired on this frame: (name, param). See engine/audio.py SFX.
    sfx: List[Tuple[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        shape = self.chars.shape
        if self.bg is None:
            self.bg = np.zeros(shape + (3,), dtype=np.uint8)
        if self.bg_active is None:
            self.bg_active = np.zeros(shape, dtype=bool)

    @staticmethod
    def blank(rows: int, cols: int) -> "Frame":
        return Frame(
            chars=np.full((rows, cols), " ", dtype="<U1"),
            colors=np.zeros((rows, cols, 3), dtype=np.uint8),
            active=np.zeros((rows, cols), dtype=bool),
        )

    @staticmethod
    def from_cells(
        live: np.ndarray,
        rgb: np.ndarray,
        rows: int,
        cols: int,
        cell_width: int = 2,
        glyph: str = "█",
    ) -> "Frame":
        """Render a (h, w) cell grid as a (rows, cols) Frame, each cell drawn
        `cell_width` characters wide. Cropped/padded to fit."""
        frame = Frame.blank(rows, cols)
        live_x = np.repeat(live, cell_width, axis=1)[:rows, :cols]
        rgb_x = np.repeat(rgb, cell_width, axis=1)[:rows, :cols]
        h, w = live_x.shape
        frame.active[:h, :w] = live_x
        frame.chars[:h, :w] = np.where(live_x, glyph, " ")
        frame.colors[:h, :w] = np.where(live_x[..., None], rgb_x, 0)
        return frame

    @staticmethod
    def from_pixels(filled: np.ndarray, rgb: np.ndarray, rows: int, cols: int) -> "Frame":
        """Render an (H, W) pixel image at double vertical resolution using
        half-block glyphs: each character cell shows a top pixel (fg of '▀')
        and a bottom pixel (bg), giving (2*rows, cols) square pixels.
        Unfilled pixels show the theme background."""
        H, W = min(filled.shape[0], 2 * rows), min(filled.shape[1], cols)
        f = np.zeros((2 * rows, cols), dtype=bool)
        px = np.zeros((2 * rows, cols, 3), dtype=np.uint8)
        f[:H, :W] = filled[:H, :W]
        px[:H, :W] = rgb[:H, :W]
        top_f, bot_f = f[0::2], f[1::2]
        top_c, bot_c = px[0::2], px[1::2]
        return Frame(
            chars=np.where(top_f, "▀", np.where(bot_f, "▄", " ")),
            colors=np.where(top_f[..., None], top_c, np.where(bot_f[..., None], bot_c, 0)).astype(np.uint8),
            active=top_f | bot_f,
            bg=np.where((top_f & bot_f)[..., None], bot_c, 0).astype(np.uint8),
            bg_active=top_f & bot_f,
        )

    def put_text(self, row: int, col: int, text: str, rgb: RGB, bg: Optional[RGB] = None) -> int:
        """Overlay text onto the grid, clipped to bounds. CJK characters take
        two cells. `bg=None` keeps the theme background. Returns the column
        just past the text."""
        c = col
        R, C = self.chars.shape
        for ch in text:
            w = char_width(ch)
            if 0 <= row < R and 0 <= c and c + w <= C:
                for k in range(w):
                    self.chars[row, c + k] = ch if k == 0 else WIDE_CONT
                    self.colors[row, c + k] = rgb
                    self.active[row, c + k] = ch != " "
                    self.bg_active[row, c + k] = bg is not None
                    if bg is not None:
                        self.bg[row, c + k] = bg
            c += w
        return c

    def copy(self) -> "Frame":
        return Frame(
            chars=self.chars.copy(),
            colors=self.colors.copy(),
            active=self.active.copy(),
            subtitle_zh=self.subtitle_zh,
            subtitle_en=self.subtitle_en,
            bg=self.bg.copy(),
            bg_active=self.bg_active.copy(),
            sfx=list(self.sfx),
        )


@dataclass
class Scene:
    """A stretch of the film. `frame_gen` yields Frames at `fps` (must divide
    engine.timeline.TICK_FPS). `transition` is how this scene enters over the
    previous one (see engine/timeline.py TRANSITIONS). `music` picks the
    soundtrack section (see engine/audio.py MOODS)."""
    name: str
    frame_gen: Callable[[], Iterator[Frame]]
    fps: int = 12
    transition: str = "dissolve"
    transition_ticks: int = 16
    music: str = "calm"


# (row, col, glyph, fg or None for default, bg or None for default)
ChangedCell = Tuple[int, int, str, Optional[RGB], Optional[RGB]]


class FrameBuffer:
    """Tracks the previously rendered frame and yields only changed cells."""

    def __init__(self, rows: int, cols: int):
        self.rows = rows
        self.cols = cols
        self.prev: Optional[Frame] = None

    def diff(self, frame: Frame) -> List[ChangedCell]:
        if frame.chars.shape != (self.rows, self.cols):
            raise ValueError(
                f"frame shape {frame.chars.shape} does not match buffer "
                f"({self.rows}, {self.cols})"
            )
        if self.prev is None:
            mask = np.ones((self.rows, self.cols), dtype=bool)
        else:
            p = self.prev
            mask = (
                (p.chars != frame.chars)
                | np.any(p.colors != frame.colors, axis=-1)
                | (p.active != frame.active)
                | np.any(p.bg != frame.bg, axis=-1)
                | (p.bg_active != frame.bg_active)
            )
            # A changed right half of a wide glyph means the glyph must be redrawn.
            cont = frame.chars[:, 1:] == WIDE_CONT
            mask[:, :-1] |= mask[:, 1:] & cont
        rows, cols = np.where(mask)
        changed: List[ChangedCell] = []
        for r, c in zip(rows.tolist(), cols.tolist()):
            ch = str(frame.chars[r, c])
            if ch == WIDE_CONT:
                continue
            if frame.active[r, c]:
                fg: Optional[RGB] = tuple(int(x) for x in frame.colors[r, c])
            else:
                ch, fg = " ", None
            bg: Optional[RGB] = (
                tuple(int(x) for x in frame.bg[r, c]) if frame.bg_active[r, c] else None
            )
            changed.append((r, c, ch, fg, bg))
        # Snapshot (not a reference): callers may mutate a Frame's arrays in
        # place across yields.
        self.prev = frame.copy()
        return changed

    def reset(self) -> None:
        self.prev = None
