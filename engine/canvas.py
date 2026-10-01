"""Drawing toolkit shared by all chapters.

Coordinates: the film is a fixed COLS x ROWS character grid (160 x 40). Most
drawing happens on a pixel Canvas of (2*ROWS) x COLS = 80 x 160 *square*
pixels (half-block glyphs), which is then converted to a Frame; text is
overlaid afterwards at character resolution with Frame.put_text / put_center
(1 character cell = 1 px wide x 2 px tall).
"""
import math
from functools import lru_cache
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from .frame import RGB, Frame, char_width

COLS, ROWS = 160, 40
PW, PH = COLS, 2 * ROWS

# --- theme -------------------------------------------------------------------
BG = (12, 13, 20)            # film background (terminal + video)
TEXT = (232, 234, 242)
DIM = (120, 124, 142)
FAINT = (58, 61, 76)         # grid lines, dead cells
AMBER = (255, 190, 70)
ORANGE = (255, 130, 60)
CYAN = (80, 210, 255)
BLUE = (100, 140, 255)
PINK = (255, 90, 160)
GREEN = (110, 230, 130)
VIOLET = (175, 125, 255)
RED = (255, 80, 80)
WHITE = (255, 255, 255)

ColorLike = Union[RGB, Sequence[int], np.ndarray]


# --- colour helpers ---------------------------------------------------------

def lerp_rgb(a: ColorLike, b: ColorLike, t) -> np.ndarray:
    """Linear blend; `t` may be a scalar or an array (broadcasts to (..., 3))."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    t = np.asarray(t, dtype=np.float32)
    if t.ndim:
        t = t[..., None]
    return np.clip(a + (b - a) * t, 0, 255).astype(np.uint8)


def scale_rgb(c: ColorLike, k: float) -> np.ndarray:
    return np.clip(np.asarray(c, dtype=np.float32) * k, 0, 255).astype(np.uint8)


def hsv(h, s=0.8, v=1.0) -> np.ndarray:
    """Vectorised HSV -> RGB uint8. h in [0, 1) (scalar or array)."""
    h = np.asarray(h, dtype=np.float32) % 1.0
    s = np.broadcast_to(np.asarray(s, dtype=np.float32), h.shape)
    v = np.broadcast_to(np.asarray(v, dtype=np.float32), h.shape)
    i = np.floor(h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


COLORMAPS = {
    "fire": [(0.0, (40, 10, 40)), (0.35, (200, 40, 60)), (0.7, (255, 170, 40)), (1.0, (255, 250, 200))],
    "ice": [(0.0, (10, 20, 50)), (0.4, (40, 110, 220)), (0.75, (90, 220, 255)), (1.0, (235, 255, 255))],
    "viridis": [(0.0, (68, 1, 84)), (0.33, (49, 104, 142)), (0.66, (53, 183, 121)), (1.0, (253, 231, 37))],
    "magma": [(0.0, (15, 10, 40)), (0.35, (120, 30, 130)), (0.7, (250, 110, 90)), (1.0, (252, 250, 190))],
    "spring": [(0.0, (20, 50, 30)), (0.5, (60, 200, 110)), (1.0, (230, 255, 150))],
    "diverge": [(0.0, (70, 120, 255)), (0.5, (30, 30, 40)), (1.0, (255, 110, 70))],
}


def colormap(values: np.ndarray, name: str = "viridis", vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    stops = COLORMAPS[name]
    t = np.clip((np.asarray(values, dtype=np.float32) - vmin) / max(vmax - vmin, 1e-9), 0, 1)
    xs = [s[0] for s in stops]
    out = np.empty(t.shape + (3,), dtype=np.uint8)
    for ch in range(3):
        out[..., ch] = np.interp(t, xs, [s[1][ch] for s in stops]).astype(np.uint8)
    return out


# --- easing -----------------------------------------------------------------

def clamp01(t: float) -> float:
    return 0.0 if t < 0 else (1.0 if t > 1 else t)


def smoothstep(t: float) -> float:
    t = clamp01(t)
    return t * t * (3 - 2 * t)


def ease_in_out(t: float) -> float:
    t = clamp01(t)
    return 0.5 - 0.5 * math.cos(math.pi * t)


def geom(a: float, b: float, t: float) -> float:
    """Geometric interpolation (for zoom factors / step counts)."""
    t = clamp01(t)
    return a * (b / a) ** t


# --- 5x7 pixel font (ASCII) --------------------------------------------------

_F = {
    "A": ".###.|#...#|#...#|#####|#...#|#...#|#...#", "B": "####.|#...#|#...#|####.|#...#|#...#|####.",
    "C": ".###.|#...#|#....|#....|#....|#...#|.###.", "D": "####.|#...#|#...#|#...#|#...#|#...#|####.",
    "E": "#####|#....|#....|####.|#....|#....|#####", "F": "#####|#....|#....|####.|#....|#....|#....",
    "G": ".###.|#...#|#....|#.###|#...#|#...#|.####", "H": "#...#|#...#|#...#|#####|#...#|#...#|#...#",
    "I": ".###.|..#..|..#..|..#..|..#..|..#..|.###.", "J": "..###|...#.|...#.|...#.|...#.|#..#.|.##..",
    "K": "#...#|#..#.|#.#..|##...|#.#..|#..#.|#...#", "L": "#....|#....|#....|#....|#....|#....|#####",
    "M": "#...#|##.##|#.#.#|#.#.#|#...#|#...#|#...#", "N": "#...#|#...#|##..#|#.#.#|#..##|#...#|#...#",
    "O": ".###.|#...#|#...#|#...#|#...#|#...#|.###.", "P": "####.|#...#|#...#|####.|#....|#....|#....",
    "Q": ".###.|#...#|#...#|#...#|#.#.#|#..#.|.##.#", "R": "####.|#...#|#...#|####.|#.#..|#..#.|#...#",
    "S": ".####|#....|#....|.###.|....#|....#|####.", "T": "#####|..#..|..#..|..#..|..#..|..#..|..#..",
    "U": "#...#|#...#|#...#|#...#|#...#|#...#|.###.", "V": "#...#|#...#|#...#|#...#|#...#|.#.#.|..#..",
    "W": "#...#|#...#|#...#|#.#.#|#.#.#|#.#.#|.#.#.", "X": "#...#|#...#|.#.#.|..#..|.#.#.|#...#|#...#",
    "Y": "#...#|#...#|.#.#.|..#..|..#..|..#..|..#..", "Z": "#####|....#|...#.|..#..|.#...|#....|#####",
    "0": ".###.|#...#|#...#|#...#|#...#|#...#|.###.", "1": "..#..|.##..|..#..|..#..|..#..|..#..|.###.",
    "2": ".###.|#...#|....#|...#.|..#..|.#...|#####", "3": "####.|....#|....#|.###.|....#|....#|####.",
    "4": "...#.|..##.|.#.#.|#..#.|#####|...#.|...#.", "5": "#####|#....|####.|....#|....#|#...#|.###.",
    "6": "..##.|.#...|#....|####.|#...#|#...#|.###.", "7": "#####|....#|...#.|..#..|.#...|.#...|.#...",
    "8": ".###.|#...#|#...#|.###.|#...#|#...#|.###.", "9": ".###.|#...#|#...#|.####|....#|...#.|.##..",
    " ": ".....|.....|.....|.....|.....|.....|.....", ".": ".....|.....|.....|.....|.....|.##..|.##..",
    ",": ".....|.....|.....|.....|.##..|..#..|.#...", ":": ".....|.##..|.##..|.....|.##..|.##..|.....",
    "!": "..#..|..#..|..#..|..#..|..#..|.....|..#..", "?": ".###.|#...#|....#|...#.|..#..|.....|..#..",
    "-": ".....|.....|.....|#####|.....|.....|.....", "+": ".....|..#..|..#..|#####|..#..|..#..|.....",
    "=": ".....|.....|#####|.....|#####|.....|.....", "/": "....#|....#|...#.|..#..|.#...|#....|#....",
    "(": "...#.|..#..|.#...|.#...|.#...|..#..|...#.", ")": ".#...|..#..|...#.|...#.|...#.|..#..|.#...",
    "'": "..#..|..#..|.#...|.....|.....|.....|.....", "#": ".#.#.|.#.#.|#####|.#.#.|#####|.#.#.|.#.#.",
    "%": "##..#|##..#|...#.|..#..|.#...|#..##|#..##", "*": ".....|#.#.#|.###.|#####|.###.|#.#.#|.....",
    "<": "...#.|..#..|.#...|#....|.#...|..#..|...#.", ">": ".#...|..#..|...#.|....#|...#.|..#..|.#...",
    "_": ".....|.....|.....|.....|.....|.....|#####", "x": ".....|.....|#...#|.#.#.|..#..|.#.#.|#...#",
    "^": "..#..|.#.#.|#...#|.....|.....|.....|.....", "&": ".##..|#..#.|#.#..|.#...|#.#.#|#..#.|.##.#",
    "×": ".....|#...#|.#.#.|..#..|.#.#.|#...#|.....",
}
PIXEL_FONT = {k: np.array([[c == "#" for c in row] for row in v.split("|")]) for k, v in _F.items()}


def pixel_text(text: str, scale: int = 1, spacing: int = 1) -> np.ndarray:
    """Bool mask of `text` in the 5x7 pixel font (upper-cased), each font
    pixel drawn scale x scale."""
    glyphs = [PIXEL_FONT.get(ch, PIXEL_FONT.get(ch.upper(), PIXEL_FONT["?"])) for ch in text]
    parts = []
    for i, g in enumerate(glyphs):
        if i:
            parts.append(np.zeros((7, spacing), dtype=bool))
        parts.append(g)
    m = np.concatenate(parts, axis=1) if parts else np.zeros((7, 0), dtype=bool)
    return np.kron(m, np.ones((scale, scale), dtype=bool)).astype(bool)


FONT_CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_CJK_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
FONT_CJK_INDEX = 2  # Simplified Chinese face inside the .ttc


@lru_cache(maxsize=256)
def font_text(text: str, height: int, threshold: float = 0.45, bold: bool = True) -> np.ndarray:
    """Bool mask of `text` rasterised from Noto Sans CJK (Bold, or Regular with
    bold=False) with a cap/ideograph height of about `height` pixels. Good for
    big Chinese titles (height >= 12)."""
    from PIL import Image, ImageDraw, ImageFont

    font = ImageFont.truetype(FONT_CJK if bold else FONT_CJK_REGULAR, size=int(height * 1.15),
                              index=FONT_CJK_INDEX)
    l, t, r, b = font.getbbox(text)
    img = Image.new("L", (r - l + 2, b - t + 2), 0)
    ImageDraw.Draw(img).text((1 - l, 1 - t), text, fill=255, font=font)
    a = np.asarray(img, dtype=np.float32) / 255.0
    m = a > threshold
    ys, xs = np.where(m)
    if len(ys) == 0:
        return m
    return m[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]


# --- pixel canvas --------------------------------------------------------------

class Canvas:
    """(PH, PW) square-pixel drawing surface. `filled` False = theme background."""

    def __init__(self, rows: int = ROWS, cols: int = COLS):
        self.rows, self.cols = rows, cols
        self.H, self.W = 2 * rows, cols
        self.filled = np.zeros((self.H, self.W), dtype=bool)
        self.rgb = np.zeros((self.H, self.W, 3), dtype=np.uint8)

    # primitives
    def rect(self, y: int, x: int, h: int, w: int, rgb: ColorLike) -> None:
        y0, x0, y1, x1 = max(y, 0), max(x, 0), min(y + h, self.H), min(x + w, self.W)
        if y1 > y0 and x1 > x0:
            self.filled[y0:y1, x0:x1] = True
            self.rgb[y0:y1, x0:x1] = rgb

    def outline(self, y: int, x: int, h: int, w: int, rgb: ColorLike, t: int = 1) -> None:
        self.rect(y, x, t, w, rgb)
        self.rect(y + h - t, x, t, w, rgb)
        self.rect(y, x, h, t, rgb)
        self.rect(y, x + w - t, h, t, rgb)

    def clear(self, y: int, x: int, h: int, w: int) -> None:
        y0, x0, y1, x1 = max(y, 0), max(x, 0), min(y + h, self.H), min(x + w, self.W)
        if y1 > y0 and x1 > x0:
            self.filled[y0:y1, x0:x1] = False

    def blit(self, mask: np.ndarray, rgb: ColorLike, y: int, x: int) -> None:
        """Paint `mask` (h, w) at (y, x). `rgb` is one colour or an (h, w, 3) image."""
        h, w = mask.shape
        rgb_arr = np.asarray(rgb, dtype=np.uint8)
        if rgb_arr.ndim == 1:
            rgb_arr = np.broadcast_to(rgb_arr, (h, w, 3))
        sy0, sx0 = max(0, -y), max(0, -x)
        sy1, sx1 = min(h, self.H - y), min(w, self.W - x)
        if sy1 <= sy0 or sx1 <= sx0:
            return
        m = mask[sy0:sy1, sx0:sx1]
        dst_f = self.filled[y + sy0: y + sy1, x + sx0: x + sx1]
        dst_c = self.rgb[y + sy0: y + sy1, x + sx0: x + sx1]
        dst_f |= m
        dst_c[m] = rgb_arr[sy0:sy1, sx0:sx1][m]

    def cells(
        self,
        rgb: np.ndarray,
        filled: np.ndarray,
        y: int,
        x: int,
        cell: int,
        gap: int = 1,
    ) -> None:
        """Draw an (h, w) cell grid with integer `cell` px per cell (including a
        `gap` px dark separator). `rgb` (h, w, 3) is each cell's colour;
        `filled` (h, w) False cells are not drawn."""
        h, w = filled.shape
        inner = max(cell - gap, 1)
        block = np.zeros((cell, cell), dtype=bool)
        block[:inner, :inner] = True
        big_f = np.kron(filled, block).astype(bool)
        big_c = np.repeat(np.repeat(rgb, cell, axis=0), cell, axis=1)
        self.blit(big_f, big_c, y, x)

    def zoom(
        self,
        rgb: np.ndarray,
        filled: np.ndarray,
        scale: float,
        src_center: Tuple[float, float],
        dst_center: Optional[Tuple[float, float]] = None,
        gap: bool = True,
        region: Optional[Tuple[int, int, int, int]] = None,
    ) -> None:
        """Continuously scaled view of an (h, w) cell grid: `scale` px per cell
        (any float > 0). The source point `src_center` (row, col in cells) lands
        on canvas pixel `dst_center` (default: region centre). When scale >= 4
        a 1 px gap separates cells. When scale < 1 cells are max-pooled so thin
        structures don't vanish. `region` = (y, x, h, w) clip rectangle."""
        ry, rx, rh, rw = region if region is not None else (0, 0, self.H, self.W)
        if dst_center is None:
            dst_center = (ry + rh / 2, rx + rw / 2)
        h, w = filled.shape
        if scale < 1:
            k = max(1, int(round(1 / scale)))
            hp, wp = -(-h // k) * k, -(-w // k) * k
            f = np.zeros((hp, wp), dtype=bool)
            f[:h, :w] = filled
            c = np.zeros((hp, wp, 3), dtype=np.uint8)
            c[:h, :w] = rgb
            fb = f.reshape(hp // k, k, wp // k, k)
            cnt = fb.sum(axis=(1, 3))
            csum = (c.astype(np.float32) * f[..., None]).reshape(hp // k, k, wp // k, k, 3).sum(axis=(1, 3))
            rgb = (csum / np.maximum(cnt, 1)[..., None]).astype(np.uint8)
            filled = cnt > 0
            src_center = (src_center[0] / k, src_center[1] / k)
            scale = scale * k
            h, w = filled.shape
        py = np.arange(ry, ry + rh)
        px = np.arange(rx, rx + rw)
        sy = (py + 0.5 - dst_center[0]) / scale + src_center[0]
        sx = (px + 0.5 - dst_center[1]) / scale + src_center[1]
        iy, ix = np.floor(sy).astype(int), np.floor(sx).astype(int)
        vy, vx = (iy >= 0) & (iy < h), (ix >= 0) & (ix < w)
        iyc, ixc = np.clip(iy, 0, h - 1), np.clip(ix, 0, w - 1)
        f = filled[np.ix_(iyc, ixc)] & vy[:, None] & vx[None, :]
        if gap and scale >= 4:
            fy = (sy - iy) * scale
            fx = (sx - ix) * scale
            f &= (fy < scale - 1)[:, None] & (fx < scale - 1)[None, :]
        sub_f = self.filled[ry: ry + rh, rx: rx + rw]
        sub_c = self.rgb[ry: ry + rh, rx: rx + rw]
        sub_f |= f
        sub_c[f] = rgb[np.ix_(iyc, ixc)][f]

    def text_px(self, text: str, y: int, x: int, rgb: ColorLike, scale: int = 1, align: str = "left") -> Tuple[int, int]:
        """5x7 pixel-font text. Returns (height, width) in px. align: left|center|right
        (relative to x)."""
        m = pixel_text(text, scale)
        x0 = x - (m.shape[1] // 2 if align == "center" else (m.shape[1] if align == "right" else 0))
        self.blit(m, rgb, y, x0)
        return m.shape

    def font_px(self, text: str, y: int, x: int, height: int, rgb: ColorLike, align: str = "left") -> Tuple[int, int]:
        """Anti-alias-free rasterised CJK/Latin text of ~`height` px."""
        m = font_text(text, height)
        x0 = x - (m.shape[1] // 2 if align == "center" else (m.shape[1] if align == "right" else 0))
        self.blit(m, rgb, y, x0)
        return m.shape

    def to_frame(self, zh: str = "", en: str = "") -> Frame:
        f = Frame.from_pixels(self.filled, self.rgb, self.rows, self.cols)
        f.subtitle_zh, f.subtitle_en = zh, en
        return f


# --- character-level overlays -------------------------------------------------

def text_width(s: str) -> int:
    return sum(char_width(ch) for ch in s)


def put_center(frame: Frame, row: int, text: str, rgb: ColorLike, col_center: Optional[int] = None,
               bg: Optional[RGB] = None) -> int:
    """Centre `text` on `col_center` (default: screen centre). Returns start col."""
    cc = frame.chars.shape[1] // 2 if col_center is None else col_center
    c0 = cc - text_width(text) // 2
    frame.put_text(row, c0, text, tuple(int(v) for v in rgb), bg)
    return c0


def put_right(frame: Frame, row: int, col_end: int, text: str, rgb: ColorLike, bg: Optional[RGB] = None) -> int:
    c0 = col_end - text_width(text)
    frame.put_text(row, c0, text, tuple(int(v) for v in rgb), bg)
    return c0


def box(frame: Frame, r: int, c: int, h: int, w: int, rgb: ColorLike, title: str = "",
        title_rgb: Optional[ColorLike] = None, style: str = "round", clear: bool = False) -> None:
    """Draw a box-drawing rectangle occupying rows r..r+h-1, cols c..c+w-1.
    `clear` blanks the interior first."""
    rgb = tuple(int(v) for v in rgb)
    tl, tr, bl, br = ("╭", "╮", "╰", "╯") if style == "round" else ("┌", "┐", "└", "┘")
    if clear:
        for rr in range(r + 1, r + h - 1):
            frame.put_text(rr, c + 1, " " * (w - 2), rgb)
    frame.put_text(r, c, tl + "─" * (w - 2) + tr, rgb)
    frame.put_text(r + h - 1, c, bl + "─" * (w - 2) + br, rgb)
    for rr in range(r + 1, r + h - 1):
        frame.put_text(rr, c, "│", rgb)
        frame.put_text(rr, c + w - 1, "│", rgb)
    if title:
        frame.put_text(r, c + 2, f" {title} ", tuple(int(v) for v in (title_rgb or rgb)))


# --- timing helpers -------------------------------------------------------------

class Captions:
    """Caption track indexed by frame number: [(start_frame, zh, en), ...]."""

    def __init__(self, cues: Sequence[Tuple[int, str, str]]):
        self.cues = sorted(cues, key=lambda c: c[0])

    def at(self, i: int) -> Tuple[str, str]:
        zh, en = "", ""
        for start, z, e in self.cues:
            if i >= start:
                zh, en = z, e
        return zh, en

    def apply(self, frame: Frame, i: int) -> Frame:
        frame.subtitle_zh, frame.subtitle_en = self.at(i)
        return frame
