"""Photo -> ASCII art (structure-aware glyph matching) for portraits etc.

Each character cell is compared with the rendered shapes of ~70 printable
ASCII glyphs (from the same monospace font the video uses) and the glyph
whose coverage pattern best matches the local image patch wins, so edges get
'/', '|', '_' ... and flat areas get density-appropriate glyphs. Brightness is
also carried by the per-character colour.
"""
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter, ImageOps

from .frame import Frame

ASSETS = Path(__file__).resolve().parent.parent / "assets" / "images"
GLYPHS = " .'`,:;-_~\"^!|/\\()[]{}<>+=*ilIjtfrcvxzsunoeaLJT17YVXZCUOQ0?#%&$@8BDHKMNPRSWEFGA469"
PATCH = (8, 4)  # (h, w) sub-samples per character cell


@lru_cache(maxsize=1)
def _glyph_bank() -> Tuple[np.ndarray, np.ndarray]:
    from .raster import atlas
    at = atlas()
    chars = np.array(list(GLYPHS)).reshape(1, -1)
    ids = at.ids(chars)[0]
    masks = at.stack()[ids]                                   # (G, 24, 12)
    g = masks.reshape(len(GLYPHS), PATCH[0], 24 // PATCH[0], PATCH[1], 12 // PATCH[1]).mean(axis=(2, 4))
    return np.array(list(GLYPHS)), g.reshape(len(GLYPHS), -1)  # (G,), (G, 32)


def load(name: str) -> Image.Image:
    return Image.open(ASSETS / f"{name}.jpg").convert("RGB")


# Per-image framing: (face centre y, x as fractions, vignette radius as a
# fraction of the image height, invert). The vignette darkens the backdrop
# so the subject reads clearly as bright glyphs on the dark film background.
FRAMING = {
    "neumann": (0.42, 0.52, 0.52, False),
    "conway": (0.40, 0.47, 0.50, False),
    "wolfram": (0.36, 0.50, 0.46, False),
    "langton": (0.38, 0.50, 0.55, False),
    "conus": (0.50, 0.50, 0.95, False),
}


def ascii_art(
    name: str,
    rows: int,
    cols: int,
    tint: Optional[Tuple[int, int, int]] = None,
    contrast: float = 1.3,
    gamma: float = 1.1,
    keep_aspect: bool = True,
    color_mix: float = 0.6,
    edges: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (chars (r, c) '<U1', colors (r, c, 3) uint8, lum (r, c) float).
    `tint`: colour for monochrome art (None = photo colours). With
    keep_aspect the art is shrunk to the photo's aspect (cells are 1:2), so the
    returned arrays may be smaller than (rows, cols)."""
    img = load(name)
    if keep_aspect:
        ar = img.width / img.height          # w/h in pixels; a cell is 1 wide x 2 tall
        c = min(cols, int(round(rows * 2 * ar)))
        r = min(rows, int(round(c / ar / 2)))
        rows, cols = max(r, 1), max(c, 1)
    gray = ImageOps.autocontrast(img.convert("L"), cutoff=1)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=120, threshold=2))
    ph, pw = PATCH
    g = np.asarray(gray.resize((cols * pw, rows * ph), Image.LANCZOS), dtype=np.float32) / 255.0
    g = np.clip((g - 0.5) * contrast + 0.5, 0, 1)
    cy, cx, rad, invert = FRAMING.get(name, (0.5, 0.5, 0.6, False))
    if invert:
        g = 1 - g
    if edges:
        gy, gx = np.gradient(g)
        e = np.hypot(gx, gy)
        g = np.clip(g + edges * e / (e.max() + 1e-6), 0, 1)
    yy, xx = np.mgrid[: g.shape[0], : g.shape[1]].astype(np.float32)
    H = g.shape[0]
    d = np.hypot((yy - cy * H) / H, (xx - cx * g.shape[1]) / H * (g.shape[1] / H) / (g.shape[1] / H))
    vign = np.clip(1.25 - d / rad, 0, 1) ** 1.5
    g = (g * vign) ** gamma
    patches = g.reshape(rows, ph, cols, pw).transpose(0, 2, 1, 3).reshape(rows, cols, -1)
    glyphs, bank = _glyph_bank()
    # scale glyph coverage so the densest glyph can represent full white
    bank = bank / bank.max()
    d = ((patches[:, :, None, :] - bank[None, None]) ** 2).sum(-1)   # (r, c, G)
    idx = d.argmin(-1)
    chars = glyphs[idx]
    lum = patches.mean(-1)
    if tint is not None:
        base = np.array(tint, dtype=np.float32)
        colors = base * (0.35 + 0.65 * lum[..., None])
    else:
        col = np.asarray(img.resize((cols, rows), Image.LANCZOS), dtype=np.float32)
        mono = 255 * (0.35 + 0.65 * lum[..., None])
        colors = col * color_mix + mono * (1 - color_mix)
        colors = colors * (0.55 + 0.6 * lum[..., None])
    return chars, np.clip(colors, 0, 255).astype(np.uint8), lum


def place(frame: Frame, art: Tuple[np.ndarray, np.ndarray, np.ndarray], r0: int, c0: int,
          show: Optional[np.ndarray] = None) -> None:
    """Paint ASCII art into a frame at (r0, c0). `show` (same shape, bool)
    limits which cells are drawn (for reveals)."""
    chars, colors, _ = art
    h, w = chars.shape
    R, C = frame.chars.shape
    r1, c1 = min(R, r0 + h), min(C, c0 + w)
    if r1 <= r0 or c1 <= c0:
        return
    sub = (slice(r0, r1), slice(c0, c1))
    ch, co = chars[: r1 - r0, : c1 - c0], colors[: r1 - r0, : c1 - c0]
    m = ch != " "
    if show is not None:
        m &= show[: r1 - r0, : c1 - c0]
    frame.chars[sub] = np.where(m, ch, frame.chars[sub])
    frame.colors[sub] = np.where(m[..., None], co, frame.colors[sub])
    frame.active[sub] |= m
    frame.bg_active[sub] &= ~m


# Portrait crops as fractions of the stored image: (x0, y0, x1, y1).
CROPS = {
    "neumann": (0.0, 0.0, 1.0, 1.0),
    "conway": (0.0, 0.0, 1.0, 1.0),
    "wolfram": (0.0, 0.0, 1.0, 1.0),
    "langton": (0.0, 0.0, 1.0, 1.0),
    "conus": (0.0, 0.0, 1.0, 1.0),
}


GAMMA = {"conway": 0.55, "langton": 0.8}


def portrait(name: str, h: int, dark=(20, 14, 10), light=(255, 222, 170),
             crop: Optional[Tuple[float, float, float, float]] = None, color: float = 0.0,
             gamma: Optional[float] = None) -> np.ndarray:
    """(h, w, 3) duotone (or partly photo-coloured, `color` in [0, 1]) portrait
    for the half-block canvas; w follows the crop's aspect ratio."""
    img = load(name)
    x0, y0, x1, y1 = crop or CROPS.get(name, (0, 0, 1, 1))
    img = img.crop((int(x0 * img.width), int(y0 * img.height), int(x1 * img.width), int(y1 * img.height)))
    w = max(1, int(round(h * img.width / img.height)))
    g = np.asarray(ImageOps.autocontrast(img.convert("L"), cutoff=1).resize((w, h), Image.LANCZOS),
                   dtype=np.float32)[..., None] / 255.0
    g = g ** (GAMMA.get(name, 1.0) if gamma is None else gamma)
    duo = np.array(dark, np.float32) + (np.array(light, np.float32) - np.array(dark, np.float32)) * g
    if color:
        col = np.asarray(img.resize((w, h), Image.LANCZOS), dtype=np.float32)
        duo = duo * (1 - color) + col * color
    # soft rounded vignette so the picture melts into the background
    yy, xx = np.mgrid[:h, :w].astype(np.float32)
    d = np.maximum(np.abs(yy / (h - 1) - 0.5) * 2, np.abs(xx / max(w - 1, 1) - 0.5) * 2)
    fadek = np.clip((1.0 - d) / 0.12, 0, 1)[..., None]
    bg = np.array([12, 13, 20], np.float32)
    return np.clip(bg + (duo - bg) * fadek, 0, 255).astype(np.uint8)


def load_mask(name: str) -> Optional[Image.Image]:
    """Person cut-out (L, 255 = subject) made by tools/make_portrait_masks.py."""
    p = ASSETS / f"{name}_mask.png"
    return Image.open(p).convert("L") if p.exists() else None


# Head-and-shoulders crops (x0, y0, x1, y1 fractions) for the character
# portraits: tighter than the photo so the face gets as many dots as possible.
PORTRAIT_CROPS = {
    "neumann": (0.0, 0.0, 1.0, 0.9),
    "conway": (0.16, 0.0, 0.84, 0.6),
    "wolfram": (0.04, 0.0, 0.96, 0.8),
    "langton": (0.04, 0.0, 0.96, 0.8),
    "conus": (0.075, 0.207, 0.87, 0.814),
}
# (gamma, local-contrast blend) per photo
PORTRAIT_TONE = {"neumann": (0.9, 0.15), "conway": (1.0, 0.4), "wolfram": (0.9, 0.0), "langton": (0.85, 0.15),
                 "conus": (1.0, 0.0)}
_BRAILLE_BIT = np.array([[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]])


def _masked_blur(a: np.ndarray, m: np.ndarray, radius: float) -> np.ndarray:
    """Blur of `a` that only averages over the mask (normalised convolution;
    three box passes ~ Gaussian), so the cut-away background never bleeds in."""
    k = max(1, int(round(radius * 0.9)))

    def box(x, axis):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (k + 1, k)
        c = np.cumsum(np.pad(x, pad, mode="edge"), axis=axis, dtype=np.float64)
        hi = np.take(c, np.arange(2 * k + 1, c.shape[axis]), axis=axis)
        lo = np.take(c, np.arange(0, c.shape[axis] - 2 * k - 1), axis=axis)
        return (hi - lo) / (2 * k + 1)

    def blur(x):
        for _ in range(3):
            x = box(box(x, 0), 1)
        return x
    return (blur(a * m) / np.maximum(blur(m), 1e-4)).astype(np.float32)


def _floyd_steinberg(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64).copy()
    h, w = a.shape
    out = np.zeros((h, w), dtype=bool)
    for y in range(h):
        row, nxt = a[y], a[y + 1] if y + 1 < h else None
        for x in range(w):
            o = row[x] >= 0.5
            out[y, x] = o
            e = row[x] - o
            if x + 1 < w:
                row[x + 1] += e * 0.4375
            if nxt is not None:
                if x > 0:
                    nxt[x - 1] += e * 0.1875
                nxt[x] += e * 0.3125
                if x + 1 < w:
                    nxt[x + 1] += e * 0.0625
    return out


@lru_cache(maxsize=16)
def char_portrait(name: str, rows: int, light=(255, 236, 200), floor: float = 0.14, contour: bool = True,
                  color: float = 0.0, threshold: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The person drawn with characters, background cut away (masks from
    tools/make_portrait_masks.py): the photo is Floyd-Steinberg dithered at
    braille resolution (each character = 2 x 4 dots) and packed into braille
    glyphs, so a 40-row portrait keeps 160 rows of detail. Colour brightness
    follows the local tone. `floor` = minimum dot density inside the person
    (so dark hair, ears and suits still show their shape); `contour` adds the
    outline of the cut-out; `color` mixes in the photo's own colours;
    `threshold` draws line-art (light-vs-local-mean) instead of dithering.
    Returns (chars (r, c), colors (r, c, 3), tone (r, c)); cells outside the
    person are ' '."""
    img = load(name)
    mask_img = load_mask(name) or Image.new("L", img.size, 255)
    x0, y0, x1, y1 = PORTRAIT_CROPS.get(name, (0, 0, 1, 1))
    box = (int(x0 * img.width), int(y0 * img.height), int(x1 * img.width), int(y1 * img.height))
    img, mask_img = img.crop(box), mask_img.crop(box)
    cols = max(1, int(round(rows * 2 * img.width / img.height)))
    size = (cols * 2, rows * 4)
    gray = img.convert("L").filter(ImageFilter.UnsharpMask(radius=3, percent=140, threshold=1))
    g = np.asarray(gray.resize(size, Image.LANCZOS), dtype=np.float32) / 255.0
    m = np.asarray(mask_img.resize(size, Image.LANCZOS), dtype=np.float32) / 255.0
    inside = m > 0.5
    lo, hi = np.percentile(g[inside], (2, 98)) if inside.any() else (0.0, 1.0)
    g = np.clip((g - lo) / max(hi - lo, 1e-3), 0, 1)
    gamma, local = PORTRAIT_TONE.get(name, (0.9, 0.0))
    if local:
        r = size[1] / 12
        mu = _masked_blur(g, m, r)
        sd = np.sqrt(np.maximum(_masked_blur((g - mu) ** 2, m, r), 1e-5))
        g = (1 - local) * g + local * np.clip(0.5 + 0.3 * (g - mu) / sd, 0, 1)
    g = g ** gamma
    g = np.where(inside, floor + (1 - floor) * g, 0.0)
    if threshold:   # line-art: a dot wherever the photo is lighter than its surroundings
        dots = (g > _masked_blur(g, m, size[1] / 24) + 0.02) & inside
    else:
        dots = _floyd_steinberg(g * m) & inside
    if contour:
        edge = inside & ~(np.roll(inside, 1, 0) & np.roll(inside, -1, 0) & np.roll(inside, 1, 1) & np.roll(inside, -1, 1))
        dots |= edge
    code = (dots.reshape(rows, 4, cols, 2) * _BRAILLE_BIT[None, :, None, :]).sum(axis=(1, 3))
    cover = m.reshape(rows, 4, cols, 2).mean(axis=(1, 3))
    tone = (g * m).reshape(rows, 4, cols, 2).mean(axis=(1, 3)) / np.maximum(cover, 1e-3)
    glyph = np.vectorize(lambda c: chr(0x2800 + int(c)))(code)
    chars = np.where((cover > 0.3) & (code > 0), glyph, " ")
    colors = np.array(light, np.float32) * (0.55 + 0.45 * np.clip(tone, 0, 1))[..., None]
    if color:
        photo = np.asarray(img.resize((cols, rows), Image.LANCZOS), dtype=np.float32)
        photo = photo * (255.0 / max(float(np.percentile(photo, 98)), 1.0))
        colors = colors * (1 - color) + np.clip(photo * 1.1 + 25, 0, 255) * color
    return chars, np.clip(colors, 0, 255).astype(np.uint8), np.where(chars != " ", tone, 0.0)


def ascii_tone(rgb: np.ndarray, seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """ASCII rendition of a (2r, c, 3) half-block image: (chars (r, c), colors (r, c, 3)).
    Used for the 'decoding' phase of portrait reveals."""
    ramp = np.array(list(" .:-=+*#%@"))
    px = rgb.astype(np.float32)
    cell = (px[0::2] + px[1::2]) / 2
    lum = cell.mean(-1) / 255.0
    idx = np.clip((lum * 1.4 * (len(ramp) - 1)).astype(int), 0, len(ramp) - 1)
    return ramp[idx], np.clip(cell * 1.15 + 20, 0, 255).astype(np.uint8)
