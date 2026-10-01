"""Rasterise Frames to 1920x1080 RGB images (for mp4 export and previews).

The character grid (160 x 40 cells of 12 x 24 px) fills the full 1920 x 960
top of the picture; the bottom 120 px band holds large bilingual subtitles
rendered with a real CJK font (terminal-sized subtitles are unreadable in a
video). Block glyphs are drawn procedurally so half-block pixels are exact
12 x 12 squares; other glyphs come from DejaVu Sans Mono (fallback: Noto Sans
CJK). A light bloom gives bright cells a phosphor glow.
"""
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .canvas import BG
from .frame import WIDE_CONT, Frame

CW, CH = 12, 24
VIDEO_W, VIDEO_H = 1920, 1080
SUB_BAND = 120
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
CJK = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
CJK_SC = 2

SUB_ZH_SIZE, SUB_EN_SIZE = 40, 27
SUB_ZH_RGB, SUB_EN_RGB = (245, 246, 250), (165, 170, 188)
SUB_FADE_TICKS = 5


def _box_glyph(ch: str) -> Optional[np.ndarray]:
    """Procedural block / box-drawing glyphs so they tile seamlessly."""
    m = np.zeros((CH, CW), dtype=np.float32)
    h2, w2 = CH // 2, CW // 2
    t = 2  # line thickness
    blocks = {
        "█": (slice(None), slice(None)), "▀": (slice(0, h2), slice(None)), "▄": (slice(h2, None), slice(None)),
        "▌": (slice(None), slice(0, w2)), "▐": (slice(None), slice(w2, None)),
    }
    if ch in blocks:
        m[blocks[ch]] = 1.0
        return m
    shades = {"░": 0.22, "▒": 0.45, "▓": 0.7}
    if ch in shades:
        m[:] = shades[ch]
        return m
    if ch == "▚":
        m[:h2, :w2] = 1; m[h2:, w2:] = 1
        return m
    if ch == "▞":
        m[:h2, w2:] = 1; m[h2:, :w2] = 1
        return m
    code = ord(ch)
    if 0x2800 <= code <= 0x28FF:  # braille: 2 x 4 dots, drawn as exact squares
        bits = code - 0x2800
        # dot numbering: 1 2 3 7 down the left column, 4 5 6 8 down the right
        order = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (2, 1), (3, 0), (3, 1))
        for b, (r, c) in enumerate(order):
            if bits >> b & 1:
                y, x = 1 + r * 6, 1 + c * 6
                m[y: y + 4, x: x + 4] = 1
        return m
    # box drawing: which arms (up, down, left, right) reach the cell centre
    arms = {
        "─": "lr", "│": "ud", "┌": "dr", "┐": "dl", "└": "ur", "┘": "ul", "╭": "dr", "╮": "dl",
        "╰": "ur", "╯": "ul", "├": "udr", "┤": "udl", "┬": "dlr", "┴": "ulr", "┼": "udlr",
    }
    if ch in arms:
        a = arms[ch]
        cy, cx = h2 - t // 2, w2 - t // 2
        if "u" in a:
            m[: cy + t, cx: cx + t] = 1
        if "d" in a:
            m[cy:, cx: cx + t] = 1
        if "l" in a:
            m[cy: cy + t, : cx + t] = 1
        if "r" in a:
            m[cy: cy + t, cx:] = 1
        return m
    return None


class GlyphAtlas:
    def __init__(self):
        self.mono = ImageFont.truetype(MONO, 20)
        self.cjk = ImageFont.truetype(CJK, 21, index=CJK_SC)
        self.index: Dict[Tuple[str, int], int] = {(" ", 0): 0}
        self.masks = [np.zeros((CH, CW), dtype=np.float32)]
        self._notdef = self._render(self.mono, "￿", CW)

    @staticmethod
    def _render(font, ch: str, width: int) -> np.ndarray:
        img = Image.new("L", (width, CH), 0)
        d = ImageDraw.Draw(img)
        # centre the glyph's advance box in the cell, baseline ~ 78% down
        d.text((width / 2, CH * 0.78), ch, fill=255, font=font, anchor="ms")
        return np.asarray(img, dtype=np.float32) / 255.0

    def _masks_for(self, ch: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        box = _box_glyph(ch)
        if box is not None:
            return box, None
        from .frame import char_width
        if char_width(ch) == 2:
            full = self._render(self.cjk, ch, 2 * CW)
            return full[:, :CW], full[:, CW:]
        m = self._render(self.mono, ch, CW)
        if np.array_equal(m, self._notdef):
            m = self._render(self.cjk, ch, CW)
        return m, None

    def ids(self, chars: np.ndarray) -> np.ndarray:
        """Map a (R, C) char array to atlas indices (wide glyphs -> left/right halves)."""
        R, C = chars.shape
        out = np.zeros((R, C), dtype=np.int32)
        uniq, inv = np.unique(chars, return_inverse=True)
        inv = inv.reshape(R, C)
        for u_i, ch in enumerate(uniq.tolist()):
            if ch == WIDE_CONT:
                continue
            key = (ch, 0)
            if key not in self.index:
                left, right = self._masks_for(ch)
                self.index[key] = len(self.masks)
                self.masks.append(left)
                if right is not None:
                    self.index[(ch, 1)] = len(self.masks)
                    self.masks.append(right)
            out[inv == u_i] = self.index[key]
        cont = chars == WIDE_CONT
        if cont.any():
            rr, cc = np.where(cont)
            for r, c in zip(rr.tolist(), cc.tolist()):
                lead = str(chars[r, c - 1]) if c > 0 else " "
                out[r, c] = self.index.get((lead, 1), 0)
        return out

    def stack(self) -> np.ndarray:
        return np.stack(self.masks)


_ATLAS: Optional[GlyphAtlas] = None


def atlas() -> GlyphAtlas:
    global _ATLAS
    if _ATLAS is None:
        _ATLAS = GlyphAtlas()
    return _ATLAS


def raster_grid(frame: Frame) -> np.ndarray:
    """(R*CH, C*CW, 3) uint8 image of the character grid."""
    at = atlas()
    ids = at.ids(frame.chars)
    masks = at.stack()[ids]                              # (R, C, CH, CW)
    masks = masks * frame.active[..., None, None]
    bg = np.where(frame.bg_active[..., None], frame.bg, np.array(BG, dtype=np.uint8)).astype(np.float32)
    fg = frame.colors.astype(np.float32)
    img = bg[:, :, None, None, :] + (fg - bg)[:, :, None, None, :] * masks[..., None]
    R, C = frame.chars.shape
    return img.transpose(0, 2, 1, 3, 4).reshape(R * CH, C * CW, 3).astype(np.uint8)


def bloom(img: np.ndarray, strength: float = 0.55) -> np.ndarray:
    h, w = img.shape[:2]
    base = np.array(BG, dtype=np.int16)
    lifted = np.clip(img.astype(np.int16) - base - 40, 0, 255).astype(np.uint8)
    small = Image.fromarray(lifted).resize((w // 8, h // 8), Image.BOX)
    small = small.filter(ImageFilter.GaussianBlur(3.0))
    glow = np.asarray(small.resize((w, h), Image.BILINEAR), dtype=np.float32)
    return np.clip(img.astype(np.float32) + strength * glow, 0, 255).astype(np.uint8)


@lru_cache(maxsize=512)
def _sub_image(text: str, size: int, rgb: Tuple[int, int, int]) -> Optional[np.ndarray]:
    """(h, w, 4) float RGBA image of one subtitle line."""
    if not text:
        return None
    font = ImageFont.truetype(CJK, size, index=CJK_SC)
    l, t, r, b = font.getbbox(text)
    pad = 6
    img = Image.new("L", (r - l + 2 * pad, size + 2 * pad + 8), 0)
    ImageDraw.Draw(img).text((pad - l, pad), text, fill=255, font=font)
    a = np.asarray(img, dtype=np.float32) / 255.0
    # soft dark outline for legibility over bright content
    halo = np.asarray(img.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(2)),
                      dtype=np.float32) / 255.0
    out = np.zeros(a.shape + (4,), dtype=np.float32)
    out[..., :3] = np.array(rgb, dtype=np.float32) * a[..., None]
    out[..., 3] = np.maximum(a, 0.7 * halo)
    # scale down if too wide
    if out.shape[1] > VIDEO_W - 60:
        k = (VIDEO_W - 60) / out.shape[1]
        im = Image.fromarray((out * [1, 1, 1, 255]).clip(0, 255).astype(np.uint8), "RGBA")
        im = im.resize((int(out.shape[1] * k), int(out.shape[0] * k)), Image.LANCZOS)
        out = np.asarray(im, dtype=np.float32) / [1, 1, 1, 255]
    return out


def _paste_line(img: np.ndarray, line: Optional[np.ndarray], cy: int, alpha: float) -> None:
    if line is None or alpha <= 0:
        return
    h, w = line.shape[:2]
    x0 = (VIDEO_W - w) // 2
    y0 = cy - h // 2
    y0c, y1c = max(y0, 0), min(y0 + h, img.shape[0])
    seg = line[y0c - y0: y1c - y0]
    a = seg[..., 3:4] * alpha
    dst = img[y0c:y1c, x0: x0 + w].astype(np.float32)
    # premultiplied colour, straight alpha
    img[y0c:y1c, x0: x0 + w] = np.clip(dst * (1 - a) + seg[..., :3] * alpha, 0, 255).astype(np.uint8)


def draw_subtitles(img: np.ndarray, zh: str, en: str, alpha: float = 1.0) -> None:
    top = VIDEO_H - SUB_BAND
    _paste_line(img, _sub_image(zh, SUB_ZH_SIZE, SUB_ZH_RGB), top + 40, alpha)
    _paste_line(img, _sub_image(en, SUB_EN_SIZE, SUB_EN_RGB), top + 92, alpha)


CHAP_ZH_SIZE, CHAP_EN_SIZE = 22, 16
CHAP_ZH_RGB, CHAP_EN_RGB = (150, 154, 175), (100, 104, 124)


def _sub_width(zh: str, en: str) -> int:
    a, b = _sub_image(zh, SUB_ZH_SIZE, SUB_ZH_RGB), _sub_image(en, SUB_EN_SIZE, SUB_EN_RGB)
    return max(a.shape[1] if a is not None else 0, b.shape[1] if b is not None else 0)


def _paste_right(img: np.ndarray, line: Optional[np.ndarray], cy: int, x1: int, alpha: float) -> None:
    if line is None or alpha <= 0:
        return
    h, w = line.shape[:2]
    y0, x0 = cy - h // 2, x1 - w
    a = line[..., 3:4] * alpha
    dst = img[y0: y0 + h, x0: x1].astype(np.float32)
    img[y0: y0 + h, x0: x1] = np.clip(dst * (1 - a) + line[..., :3] * alpha, 0, 255).astype(np.uint8)


def draw_chapter(img: np.ndarray, chapter: Tuple[str, str], subs: Tuple[str, str]) -> None:
    """Small chapter indicator in the bottom-right corner of the subtitle band;
    hidden while a subtitle is too wide to leave room for it."""
    zh, en = chapter
    if not zh:
        return
    lz, le = _sub_image(zh, CHAP_ZH_SIZE, CHAP_ZH_RGB), _sub_image(en, CHAP_EN_SIZE, CHAP_EN_RGB)
    lw = max(lz.shape[1] if lz is not None else 0, le.shape[1] if le is not None else 0)
    if _sub_width(*subs) > VIDEO_W - 2 * (lw + 36):
        return
    top = VIDEO_H - SUB_BAND
    _paste_right(img, lz, top + 44, VIDEO_W - 22, 1.0)
    _paste_right(img, le, top + 78, VIDEO_W - 22, 1.0)


def render_image(frame: Frame, subs_now: Tuple[str, str], subs_prev: Tuple[str, str] = ("", ""),
                 sub_alpha: float = 1.0, glow: bool = True, chapter: Tuple[str, str] = ("", "")) -> np.ndarray:
    """Full 1920x1080 picture. During a subtitle change, `subs_prev` fades out
    while `subs_now` fades in (sub_alpha = progress)."""
    img = np.empty((VIDEO_H, VIDEO_W, 3), dtype=np.uint8)
    img[:] = BG
    grid = raster_grid(frame)
    if glow:
        grid = bloom(grid)
    img[: grid.shape[0], : grid.shape[1]] = grid
    if sub_alpha < 1 and subs_prev != subs_now:
        draw_subtitles(img, *subs_prev, alpha=1 - sub_alpha)
    draw_subtitles(img, *subs_now, alpha=sub_alpha)
    draw_chapter(img, chapter, subs_now if sub_alpha >= 0.5 else subs_prev)
    return img


def film_image(film, tick: int, glow: bool = True) -> np.ndarray:
    """Picture for `tick` of a Film, including subtitle cross-fades."""
    now = film.subtitle(tick)
    k = 0
    while k < SUB_FADE_TICKS and tick - k - 1 >= 0 and film.subtitle(tick - k - 1) == now:
        k += 1
    prev = film.subtitle(tick - k - 1) if tick - k - 1 >= 0 else ("", "")
    alpha = 1.0 if k >= SUB_FADE_TICKS or prev == now else (k + 1) / (SUB_FADE_TICKS + 1)
    return render_image(film.frame(tick), now, prev, alpha, glow, film.chapter_at(tick))
