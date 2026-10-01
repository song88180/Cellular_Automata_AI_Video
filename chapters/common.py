"""Scenes shared by every chapter, so the film has one consistent visual
language: chapter title cards and 'who is this' portrait cards, both of which
reveal their Chinese text by *reverse cellular-automaton evolution*."""
from functools import lru_cache
from typing import Iterator, List, Sequence, Tuple

import numpy as np

from engine.ascii_art import ascii_tone, char_portrait, portrait
from engine.canvas import (BG, DIM, FAINT, PH, PW, TEXT, Canvas, font_text, lerp_rgb, pixel_text,
                           scale_rgb, smoothstep)
from engine.frame import Frame, Scene
from engine.fx import threshold_map
from sim.game_of_life import step as life_step

FPS = 24
# "braille": the person drawn with dithered braille characters, background cut
# away; "pixels": the original duotone half-block photo.
PORTRAIT_STYLE = "braille"
# Chinese chapter titles and portrait names: False = thin (Noto Sans CJK
# Regular, the current choice), True = the original bold face.
TITLE_BOLD = False


def title_font(text: str, height: int) -> np.ndarray:
    return font_text(text, height, 0.45, True) if TITLE_BOLD else font_text(text, height, 0.35, False)


# --- reverse evolution --------------------------------------------------------------

@lru_cache(maxsize=32)
def _life_history(mask_bytes: bytes, shape: Tuple[int, int], gens: int, pad: int) -> Tuple[np.ndarray, ...]:
    mask = np.frombuffer(mask_bytes, dtype=bool).reshape(shape)
    g = np.pad(mask, pad)
    hist = [g]
    for _ in range(gens):
        g = life_step(g)
        hist.append(g)
    return tuple(hist)


def reverse_schedule(gens: int, fps: int, final_hold: float = 1.0, ratio: float = 0.7) -> List[int]:
    """Which history state to show on each frame of a reverse reveal: states
    gens, gens-1, ..., 1 in turn, slowing down geometrically - state 1 (the
    second-to-last) is held `final_hold` seconds, state 2 `ratio` times as
    long, state 3 `ratio`^2 ... (at least one frame each). The final text
    (state 0) follows the returned frames."""
    seq: List[int] = []
    for k in range(gens, 0, -1):
        n = max(1, int(round(final_hold * ratio ** (k - 1) * fps)))
        seq += [k] * n
    return seq


def reverse_life_state(mask: np.ndarray, k: int, gens: int) -> Tuple[np.ndarray, int]:
    """History state k of `mask` under Game of Life (k = 0 is the text itself).
    Returns (padded mask, pad): draw it at (y - pad, x - pad)."""
    pad = gens + 2
    hist = _life_history(np.ascontiguousarray(mask, dtype=bool).tobytes(), mask.shape, gens, pad)
    return hist[k], pad


def _dilate(m: np.ndarray, r: int) -> np.ndarray:
    out = m.copy()
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx <= r * r:
                out |= np.roll(np.roll(m, dy, 0), dx, 1)
    return out


class ReverseReveal:
    """The 'un-evolution' reveal: the text is run forward `gens` generations
    of Game of Life (B3/S23) and that history is shown *backwards*, so debris
    re-assembles into the text - slowing down towards the end - while fading
    in. On landing: the text flashes white inside a glowing halo, shakes, and
    can send out a shockwave ring and sparks; `strong` adds a full-screen flash."""

    IMPACT_SECONDS = 0.8

    def __init__(self, mask: np.ndarray, gens: int = 20, fps: int = FPS, final_hold: float = 1.0,
                 strong: bool = False, ring: bool = False, sparks: bool = False, shake: bool = False):
        """strong = full-screen flash (chapter titles); ring / sparks / shake are
        the shockwave ring, flying sparks and camera shake on landing. The
        white text flash and the glowing halo are always on."""
        self.mask, self.gens, self.fps, self.strong = mask, gens, fps, strong
        self.ring, self.sparks, self.use_shake = ring, sparks, shake
        self.seq = reverse_schedule(gens, fps, final_hold)
        self.n = len(self.seq)          # frames until the text lands
        pad = 4
        self.halo = _dilate(np.pad(mask, pad), 3) & ~np.pad(mask, pad)
        rng = np.random.default_rng(int(mask.sum()))
        ys, xs = np.nonzero(mask)
        k = rng.choice(len(ys), size=min(70, len(ys)), replace=False)
        cy, cx = ys.mean(), xs.mean()
        self.sp_y, self.sp_x = ys[k].astype(float), xs[k].astype(float)
        ang = np.arctan2(self.sp_y - cy, self.sp_x - cx) + rng.normal(0, 0.5, len(k))
        spd = rng.uniform(25, 80, len(k))       # px per second
        self.sp_vy, self.sp_vx = np.sin(ang) * spd * 0.8, np.cos(ang) * spd
        self.centre = (cy, cx)
        self.radius0 = 0.5 * np.hypot(*mask.shape)
        self.shake = rng.normal(0, 1, (int(0.3 * fps) + 1, 2))

    def draw(self, cv: Canvas, y: int, x: int, i: int, rgb, fade_from=BG) -> None:
        """Draw frame i (i may be negative = nothing yet, >= n = arrived)."""
        if i < 0:
            return
        if i < self.n:
            m, pad = reverse_life_state(self.mask, self.seq[i], self.gens)
            k = 0.35 + 0.65 * smoothstep(i / (0.8 * self.n))
            cv.blit(m, lerp_rgb(fade_from, rgb, k), y - pad, x - pad)
            return
        t = (i - self.n) / self.fps
        a = max(0.0, 1.0 - t / self.IMPACT_SECONDS)       # impact strength, 1 -> 0
        if self.strong and t < 0.18:                      # full-screen flash
            cv.rect(0, 0, cv.H, cv.W, scale_rgb((210, 215, 235), (1 - t / 0.18) ** 1.5 * 0.55))
        cy, cx = y + self.centre[0], x + self.centre[1]
        if a > 0 and self.ring:
            # shockwave ring
            r = self.radius0 * (0.6 + 2.2 * (1 - a) ** 0.6)
            yy, xx = np.mgrid[: cv.H, : cv.W]
            d = np.hypot((yy + 0.5 - cy), (xx + 0.5 - cx))
            ring = np.abs(d - r) < 1.2 + 1.5 * a
            cv.filled |= ring
            cv.rgb[ring] = lerp_rgb(cv.rgb[ring], (255, 255, 255), a * 0.85)
        if a > 0 and self.sparks:
            py = np.round(cy - self.centre[0] + self.sp_y + self.sp_vy * t).astype(int)
            px = np.round(cx - self.centre[1] + self.sp_x + self.sp_vx * t).astype(int)
            ok = (py >= 0) & (py < cv.H) & (px >= 0) & (px < cv.W)
            cv.filled[py[ok], px[ok]] = True
            cv.rgb[py[ok], px[ok]] = lerp_rgb(rgb, (255, 255, 255), a)
        # shake for the first 0.3 s
        j = i - self.n
        dy, dx = (np.round(self.shake[j] * 2.5 * a).astype(int)
                  if (self.use_shake and j < len(self.shake)) else (0, 0))
        if a > 0:
            cv.blit(self.halo, lerp_rgb(BG, (255, 250, 230), a ** 3.5), y - 4 + dy, x - 4 + dx)
        white = 1.0 if t < 0.12 else max(0.0, 1 - (t - 0.12) / 0.6)
        cv.blit(self.mask, lerp_rgb(rgb, (255, 255, 255), white), y + dy, x + dx)

    def ticking(self, i: int) -> bool:
        """True on frames where the displayed state changes (for a tick sound)."""
        return 0 < i < self.n and self.seq[i] != self.seq[i - 1]


# --- chapter title cards ----------------------------------------------------------

def title_card_frames(number: int, zh: str, en: str, accent, post: int = 48,
                      rows: int = 40, cols: int = 160, en_label: str = "") -> Iterator[Frame]:
    """'CHAPTER n' card: the Chinese title re-assembles out of Game-of-Life
    debris (reverse evolution, slowing down, fading in), lands with a flash,
    then the number, English title and a rule line appear for `post` frames."""
    zh_m = title_font(zh, 15)
    en_m = pixel_text((en_label or en).upper(), 1)
    num_m = pixel_text(f"{number:02d}", 3)
    W = PW
    zy = 30
    zx = (W - zh_m.shape[1]) // 2
    ny = zy - num_m.shape[0] - 5
    nx = (W - num_m.shape[1]) // 2
    ey = zy + zh_m.shape[0] + 7
    ex = (W - en_m.shape[1]) // 2
    accent = np.array(accent, np.float32)
    rev = ReverseReveal(zh_m, gens=20, fps=FPS, strong=True, ring=True, sparks=True)
    title_rgb = accent * 0.35 + 255 * 0.65
    n_frames = rev.n + post
    for i in range(n_frames):
        cv = Canvas(rows, cols)
        rev.draw(cv, zy, zx, i, title_rgb)
        if i >= rev.n:
            reveal = smoothstep((i - rev.n) / (0.55 * post))
            cv.blit(num_m, scale_rgb(accent, 0.35 + 0.65 * reveal), ny, nx)
            n_en = int(round(reveal * en_m.shape[1]))
            cv.blit(en_m[:, :n_en], DIM, ey, ex)
            half = int(reveal * 50)
            cv.rect(ey + 11, W // 2 - half, 1, 2 * half, scale_rgb(accent, 0.8))
        f = cv.to_frame()
        if i == rev.n:
            f.sfx.append(("hit", 1.0))
            f.sfx.append(("boom", 0.8))
        if rev.ticking(i):
            f.sfx.append(("tick", 0.3))
        fade_start = n_frames - 10   # fade out over the last 10 frames
        if i > fade_start:
            k2 = 1 - (i - fade_start) / 10
            f.colors[:] = scale_rgb(f.colors, k2)
            f.bg[:] = scale_rgb(f.bg, k2)
        yield f


def title_card(number: int, zh: str, en: str, accent, music: str = "calm",
               transition: str = "fade", post: int = 48) -> Scene:
    return Scene(f"ch{number}-title", lambda: title_card_frames(number, zh, en, accent, post),
                 fps=FPS, transition=transition, transition_ticks=12, music=music)


# --- portrait cards -------------------------------------------------------------

def _portrait_art(name: str, style: str, accent, photo_rows: int):
    """(chars, colors) of the finished portrait at character resolution,
    plus its (rows, cols) and the (2r, c, 3) pixel image for the 'pixels' style."""
    light = tuple(int(v) for v in lerp_rgb(accent, (255, 255, 255), 0.35))
    if style == "braille":
        chars, colors, _ = char_portrait(name, photo_rows, light=light)
        return chars, colors, None
    dark = tuple(int(v) for v in scale_rgb(accent, 0.07))
    img = portrait(name, 2 * photo_rows, dark=dark, light=light)
    return None, None, img


def portrait_frames(
    name: str,
    zh_name: str,
    en_name: str,
    years: str,
    facts: Sequence[Tuple[str, str]],
    captions: Sequence[Tuple[int, str, str]],
    n_frames: int,
    accent=(255, 205, 130),
    rows: int = 40,
    cols: int = 160,
    photo_rows: int = 38,
    credit: str = "",
    style: str = "",
) -> Iterator[Frame]:
    """Portrait card at 24 fps: the photo decodes out of falling glyphs on the
    left; the Chinese name re-assembles by reverse Game-of-Life evolution; the
    English name, years and (zh, en) fact lines type out on the right.
    `captions` = [(start_frame, zh, en)]."""
    style = style or PORTRAIT_STYLE
    if style == "pixels":
        photo_rows = min(photo_rows, 33)
    chars, colors, img = _portrait_art(name, style, accent, photo_rows)
    if style == "braille":
        ph2, pw = chars.shape
        r0, c0 = max(0, (rows - ph2) // 2 - 1), 6
        tone_chars, tone_cols = chars, colors
    else:
        ph, pw = img.shape[:2]
        ph2 = ph // 2
        r0, c0 = (PH - ph) // 4 - 1, 10
        tone_chars, tone_cols = ascii_tone(img)
    tmap = threshold_map("rain", ph2, pw, seed=3)
    text_c = c0 + pw + 7
    for name_h in (10, 9, 8, 7):  # long names (约翰·冯·诺依曼) must still fit
        name_m = title_font(zh_name, name_h)
        if text_c + name_m.shape[1] <= cols - 3:
            break
    name_y = 10
    name_start = 4
    rev = ReverseReveal(name_m, gens=20, fps=FPS)
    name_len = rev.n                        # reverse-evolution window (frames)
    n_frames += max(0, name_len - 34)       # the slower reveal lengthens the card
    decode_end = 0.3
    type_start = name_start + name_len - 6
    cap = sorted(captions)
    rng_glyphs = np.array(list("01<>/*#+=:."))
    for i in range(n_frames):
        t = i / max(1, n_frames - 1)
        cv = Canvas(rows, cols)
        p_dec = smoothstep(i / (n_frames * decode_end))
        cell_done = (tmap < p_dec) | (p_dec >= 1)
        if style == "pixels":
            cv.blit(np.repeat(cell_done, 2, axis=0), img, 2 * r0, c0)
        rev.draw(cv, name_y, text_c, i - name_start, TEXT)
        f = cv.to_frame()
        if style == "braille":
            show = cell_done & (chars != " ")
            rr, cc = np.nonzero(show)
            f.chars[r0 + rr, c0 + cc] = chars[rr, cc]
            f.colors[r0 + rr, c0 + cc] = colors[rr, cc]
            f.active[r0 + rr, c0 + cc] = True
        # glyph rain for the band just ahead of the decoded part
        glyph_zone = (~cell_done) & (tmap < p_dec + 0.35) & (tone_chars != " ")
        rr, cc = np.nonzero(glyph_zone)
        rng = np.random.default_rng(i)
        for r, c in zip(rr.tolist(), cc.tolist()):
            ch = tone_chars[r, c] if rng.random() < 0.6 else rng_glyphs[rng.integers(0, len(rng_glyphs))]
            f.put_text(r0 + r, c0 + c, ch, tuple(int(v) for v in lerp_rgb(tone_cols[r, c], accent, 0.3)))
        if credit and t > decode_end:
            f.put_text(min(rows - 1, r0 + ph2), c0, credit, FAINT)
        row = (name_y + name_m.shape[0]) // 2 + 2
        nm = smoothstep((i - name_start - name_len + 8) / 10)
        if nm > 0:
            f.put_text(row, text_c, en_name, tuple(int(v) for v in scale_rgb(accent, nm)))
            f.put_text(row, text_c + len(en_name) + 3, years, tuple(int(v) for v in scale_rgb(DIM, nm)))
        row += 3
        budget = max(0, i - type_start) * 2          # zh chars revealed so far
        for k, (zh_line, en_line) in enumerate(facts):
            n = len(zh_line) + len(en_line) // 3
            shown = min(budget, n)
            if shown <= 0:
                break
            if shown == 2 and i > type_start:
                f.sfx.append(("blip", 0.4 + 0.1 * k))
            f.put_text(row, text_c, "▸ " + zh_line[:shown], TEXT)
            en_n = min(len(en_line), max(0, shown - len(zh_line)) * 3 + (3 if shown == n else 0))
            if en_n:
                f.put_text(row + 1, text_c + 2, en_line[:en_n], DIM)
            budget -= n + 6
            row += 3
        if rev.ticking(i - name_start):
            f.sfx.append(("tick", 0.3))
        if i == name_start + name_len:
            f.sfx.append(("hit", 0.6))
            f.sfx.append(("chime", 0.6))
        zh, en = "", ""
        for start, z, e in cap:
            if i >= start:
                zh, en = z, e
        f.subtitle_zh, f.subtitle_en = zh, en
        yield f


def portrait_scene(scene_name: str, music: str = "calm", transition: str = "dissolve", **kw) -> Scene:
    return Scene(scene_name, lambda: portrait_frames(**kw), fps=FPS, transition=transition,
                 transition_ticks=16, music=music)
