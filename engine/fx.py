"""Reveal / transition effects on whole Frames (character-cell resolution).

A *threshold map* gives every character cell a value in [0, 1]; at progress p
cells with value < p show the new content. A thin band just behind the front
is drawn as bright 'decoding' glyphs, which makes every transition look like
the picture is being computed cell by cell.
"""
from functools import lru_cache
from typing import Optional, Tuple

import numpy as np

from .frame import WIDE_CONT, Frame

EDGE_GLYPHS = np.array(list("░▒▓█▚▞"))
EDGE_WIDTH = 0.07


@lru_cache(maxsize=64)
def threshold_map(kind: str, rows: int, cols: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rr, cc = np.mgrid[:rows, :cols].astype(np.float32)
    if kind == "dissolve":
        m = rng.random((rows, cols))
    elif kind == "wipe":            # left -> right with a ragged front
        m = cc / cols + 0.12 * rng.random((rows, 1)) + 0.05 * rng.random((rows, cols))
    elif kind == "wipe_down":
        m = rr / rows + 0.12 * rng.random((1, cols)) + 0.05 * rng.random((rows, cols))
    elif kind == "iris":            # from the centre outwards (cells are 1:2)
        d = np.hypot((rr - rows / 2) * 2, cc - cols / 2)
        m = d / d.max() + 0.08 * rng.random((rows, cols))
    elif kind == "rain":            # each column falls with its own delay
        m = rr / rows * 0.55 + 0.45 * rng.random((1, cols))
    elif kind == "rule30":          # the front is a Rule 30 triangle
        from sim.elementary_ca import generate
        ca = generate(cols, 30, rows - 1).astype(np.float32)
        d = np.abs(cc - cols / 2) / (cols / 2) * 0.6 + rr / rows * 0.4
        m = d + 0.25 * ca * rng.random((rows, cols))
    else:
        raise ValueError(f"unknown threshold map {kind!r}")
    m = m.astype(np.float32)
    m -= m.min()
    m /= max(float(m.max()), 1e-6)
    return m


def _keep_wide_pairs(show_b: np.ndarray, b: Frame, a: Frame) -> np.ndarray:
    """Never split a double-width glyph between the two frames."""
    for fr in (a, b):
        cont = fr.chars[:, 1:] == WIDE_CONT
        show_b[:, 1:] = np.where(cont, show_b[:, :-1], show_b[:, 1:])
    return show_b


def blend_reveal(
    a: Frame,
    b: Frame,
    p: float,
    tmap: np.ndarray,
    edge_rgb: Optional[Tuple[int, int, int]] = (150, 230, 255),
    tick: int = 0,
) -> Frame:
    """Frame showing `b` where tmap < p, `a` elsewhere, with a glowing edge.
    Subtitles/sfx come from `b`."""
    show_b = _keep_wide_pairs(tmap < p, b, a)
    out = b.copy()
    keep_a = ~show_b
    for name in ("chars", "colors", "active", "bg", "bg_active"):
        getattr(out, name)[keep_a] = getattr(a, name)[keep_a]
    if edge_rgb is not None and 0 < p < 1 + EDGE_WIDTH:
        edge = (tmap < p) & (tmap >= p - EDGE_WIDTH)
        # the edge must not cut through wide glyphs either
        cont = out.chars == WIDE_CONT
        edge &= ~cont
        edge[:, :-1] &= ~cont[:, 1:]
        if edge.any():
            rng = np.random.default_rng(tick)
            k = (p - tmap[edge]) / EDGE_WIDTH          # 0 at the front -> 1 behind
            out.chars[edge] = EDGE_GLYPHS[rng.integers(0, len(EDGE_GLYPHS), int(edge.sum()))]
            base = np.array(edge_rgb, dtype=np.float32)
            glow = (base[None] * (1.0 - 0.55 * k[:, None]) + 255 * 0.35 * (1 - k[:, None]))
            out.colors[edge] = np.clip(glow, 0, 255).astype(np.uint8)
            out.active[edge] = True
            out.bg_active[edge] = False
    return out


def fade(frame: Frame, k: float) -> Frame:
    out = frame.copy()
    k = float(np.clip(k, 0.0, 1.0))
    out.colors = (out.colors.astype(np.float32) * k).astype(np.uint8)
    out.bg = (out.bg.astype(np.float32) * k).astype(np.uint8)
    return out


def transition(kind: str, a: Frame, b: Frame, p: float, tick: int = 0, seed: int = 0) -> Frame:
    """Composite for a scene transition at progress p in [0, 1]."""
    rows, cols = b.chars.shape
    if kind == "cut" or a is None:
        return b
    if kind == "fade":  # through black
        return fade(a, 1 - 2 * p) if p < 0.5 else fade(b, 2 * p - 1)
    if kind == "flash":  # quick white-ish flash then the new scene (for big reveals)
        out = b.copy()
        k = max(0.0, 1.0 - p * 1.6)
        out.colors = np.clip(out.colors.astype(np.float32) + 200 * k, 0, 255).astype(np.uint8)
        return out
    tmap = threshold_map(kind, rows, cols, seed)
    # overshoot so the trailing edge fully clears by p = 1
    return blend_reveal(a, b, p * (1 + EDGE_WIDTH), tmap, tick=tick)
