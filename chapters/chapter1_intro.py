"""Chapter 1: film title + where cellular automata come from.

Beats: the pixel title CELLULAR AUTOMATA dissolves in; it then evolves under
Game of Life rules while the camera zooms in, until every block has flown out
past the screen edges -> '01 起源 ORIGIN' card -> John von Neumann card ->
a *running* piece of von Neumann's original 29-state automaton (a signal loop
feeding a construction arm that builds itself) with the definition of a CA
-> "let's start with the simplest case, 1D".
"""
from typing import Iterator, List, Tuple

import numpy as np

from chapters.common import portrait_scene, title_card
from engine.canvas import (AMBER, CYAN, DIM, PH, PW, TEXT, Canvas, clamp01, ease_in_out, geom,
                           lerp_rgb, pixel_text, put_center, scale_rgb, smoothstep, text_width)
from engine.frame import Frame, Scene
from sim import von_neumann as vn
from sim.game_of_life import step as life_step

CHAPTER_LABEL = ("第 1 章 · 起源", "Chapter 1 · Origin")

FPS = 12
TITLE_ZH = "元胞自动机：简单规则如何创造复杂世界"
TITLE_EN = "Cellular Automata: how simple rules create complex worlds"
ACCENT = AMBER


def _title_mask() -> np.ndarray:
    lines = [pixel_text("CELLULAR", 3), pixel_text("AUTOMATA", 3)]
    gap = 9
    total = sum(m.shape[0] for m in lines) + gap
    mask = np.zeros((PH, PW), dtype=bool)
    r = (PH - total) // 2 - 2
    for m in lines:
        c = (PW - m.shape[1]) // 2
        mask[r: r + m.shape[0], c: c + m.shape[1]] = m
        r += m.shape[0] + gap
    return mask


def _rainbow(shift: float, h: int = PH, w: int = PW, x0: int = 0) -> np.ndarray:
    t = ((np.arange(w) - x0) / PW + shift) % 1.0
    rgb = np.stack([0.5 + 0.5 * np.cos(2 * np.pi * (t + o)) for o in (0.0, 0.33, 0.67)], axis=-1)
    rgb = (rgb * 175 + 80).astype(np.uint8)
    return np.broadcast_to(rgb[None], (h, w, 3)).copy()


def _empty_spot(states: List[np.ndarray], near: Tuple[float, float]) -> Tuple[float, float]:
    """A point (in grid coords) that stays far from every live cell in all of
    `states`, preferring points near `near` -- where the final zoom can land
    on an empty screen."""
    live = np.zeros_like(states[0])
    for s in states:
        live |= s
    ys, xs = np.nonzero(live)
    best, best_score = near, -1e9
    for cy in np.arange(near[0] - 30, near[0] + 31, 2.0):
        for cx in np.arange(near[1] - 60, near[1] + 61, 2.0):
            d = np.min(np.hypot((ys - cy) * 2, xs - cx)) if len(ys) else 99.0
            score = min(d, 12.0) - 0.02 * np.hypot((cy - near[0]) * 2, cx - near[1])
            if score > best_score:
                best, best_score = (float(cy), float(cx)), score
    return best


def _title_frames(rows: int, cols: int) -> Iterator[Frame]:
    mask = _title_mask()
    rng = np.random.default_rng(1)
    appear_at = rng.integers(0, 20, size=mask.shape)
    n_in, n_hold, n_evolve = 20, 34, 84
    for _ in range(FPS):          # 1 s of pure black before anything appears
        yield Canvas(rows, cols).to_frame()
    for i in range(n_in + n_hold):
        shown = mask & (appear_at <= i)
        cv = Canvas(rows, cols)
        cv.blit(shown, _rainbow(i * 0.032), 0, 0)
        f = cv.to_frame(TITLE_ZH, TITLE_EN)
        if i == 0:
            f.sfx.append(("chime", 0.8))
        if i == n_in:
            f.sfx.append(("hit", 0.7))
        yield f
    # Game of Life on a padded world (nothing wraps back), camera zooming in
    pad = 70
    grid = np.pad(mask, pad)
    H, W = grid.shape
    states, ages, borns = [], [], []
    age = np.zeros(grid.shape, dtype=np.int32)
    for i in range(n_evolve):
        new = life_step(grid)
        age = np.where(new & grid, age + 1, 0)
        borns.append(new & ~grid)
        grid = new
        states.append(grid)
        ages.append(age)
    centre = (pad + PH / 2, pad + PW / 2)
    final_scale = 60.0
    target = _empty_spot(states[-20:], centre)
    shift0 = (n_in + n_hold) * 0.032
    for i in range(n_evolve):
        t = i / (n_evolve - 1)
        z = ease_in_out(clamp01((t - 0.15) / 0.85))
        scale = geom(1.0, final_scale, z)
        pan = smoothstep(clamp01((t - 0.2) / 0.7))
        src = (centre[0] + (target[0] - centre[0]) * pan, centre[1] + (target[1] - centre[1]) * pan)
        cv = Canvas(rows, cols)
        rgb = lerp_rgb(_rainbow(shift0 + i * 0.032, H, W, x0=pad), (255, 255, 255), np.where(borns[i], 0.75, 0.0))
        cv.zoom(rgb, states[i], scale, src_center=src, dst_center=(PH / 2, PW / 2), gap=scale >= 6)
        f = cv.to_frame(TITLE_ZH, TITLE_EN)
        if i == 0:
            f.sfx.append(("glitch", 0.8))
        if i % 6 == 0:
            f.sfx.append(("pop", 0.3 + 0.5 * t))
        if i == int(n_evolve * 0.25):
            f.sfx.append(("rise", 3.5))
        yield f
        if i > n_evolve // 2 and not f.active.any():
            return   # every block has left the screen: cut straight to the 01 card


# --- von Neumann's 29-state automaton ---------------------------------------------------

CELL = 10                           # px per cell (9 + 1 px gap)
GR, GC = 6, 16                      # visible grid rows x cols (60 x 160 px)
SIM_R, SIM_C = 12, 24               # simulated grid (the arm keeps growing off-screen)
GY = 12                             # grid top in px
MACHINE = dict(r0=0, c0=1, arm=2)
ARM_START = (0, 1 + 3 + 1)          # first arm wire (after the loop's confluent cell)

COL_U = (26, 28, 40)
COL_ORD = ((40, 72, 140), (140, 190, 255), (120, 205, 255), (255, 255, 255))  # fill, glyph; excited fill, glyph
COL_SPEC = ((120, 40, 72), (255, 140, 180), (255, 120, 170), (255, 255, 255))
COL_CONF = ((120, 88, 28), (255, 210, 120), (255, 200, 90), (255, 255, 255))
COL_SENS = ((80, 55, 140), (215, 190, 255))

_ARROW_E = np.array([[c == "#" for c in row] for row in (
    ".......", "....#..", ".....#.", "#######", ".....#.", "....#..", ".......")])
ARROWS = [np.rot90(_ARROW_E, k) for k in range(4)]  # E N W S (rot90 is counter-clockwise)
LETTER_C, LETTER_S = pixel_text("C"), pixel_text("S")

LEGEND = [
    ("■", (95, 100, 124), "空白 ground", "×1"),
    ("→", COL_ORD[1], "普通传输 ordinary", "×8"),
    ("→", COL_SPEC[1], "特殊传输 special", "×8"),
    ("C", COL_CONF[1], "汇合 confluent", "×4"),
    ("S", COL_SENS[1], "敏化 sensitized", "×8"),
]

VN_INTRO = [  # (start frame, zh, en)
    (0, "这是冯·诺依曼最初的元胞自动机：每个格子处于 29 种状态之一",
     "This is von Neumann's original automaton: every cell is in one of 29 states"),
    (48, "带箭头的格子是「传输态」：信号每一步沿箭头前进一格",
     "Cells with arrows are 'transmission' states: a signal moves one cell per step along them"),
    (108, "左上角的环里循环着两条「指令」；汇合态 C 把它们复制一份，送进「构造臂」",
     "The loop at the top left stores two 'instructions'; the confluent cell C copies them into the construction arm"),
    (168, "指令 10000 在臂尖造出向右的箭头，1010 造出向下的箭头：构造臂一边建造、一边转弯",
     "Code 10000 builds a right arrow at the tip, 1010 a down arrow: the arm builds itself and turns as it grows"),
]
VN_OUTRO = [  # (frames after the arm has left the view, zh, en)
    (0, "这就是元胞自动机：一张格子网，每格有限几种状态，所有格子按同一条局部规则同时更新",
     "That is a cellular automaton: a grid of cells, a few states each, one local rule applied to all at once"),
    (54, "不过，让我们先从最简单的一维网格开始",
     "But let's start with the simplest grid: one dimension"),
]
VN_TAIL = 96          # frames shown after the arm leaves the view (incl. 24-frame fade)


def _vn_steps(i: int) -> int:
    """Steps to apply on frame i (slow, then faster)."""
    if i < 48:
        return 0
    if i < 108:
        return int(i % 6 == 0)
    if i < 168:
        return int(i % 4 == 0)
    return int(i % 2 == 0)


def _vn_offscreen_frame() -> int:
    """First frame on which the arm's construction site has left the view."""
    g = vn.construction_stair(SIM_R, SIM_C, **MACHINE)
    i = 0
    while True:
        for _ in range(_vn_steps(i)):
            g = vn.step(g)
        (tr, tc), _ = vn.arm_tip(g, *ARM_START)
        if i > 168 and not (tr < GR and tc < GC):
            return i
        i += 1


def _draw_vn(cv: Canvas, g: np.ndarray, reveal: np.ndarray) -> None:
    for r in range(GR):
        for c in range(GC):
            if not reveal[r, c]:
                continue
            s = int(g[r, c])
            y, x = GY + r * CELL, c * CELL
            glyph, gcol = None, None
            if vn.is_T(s):
                pal = COL_ORD if vn.t_kind(s) == 0 else COL_SPEC
                e = vn.t_exc(s)
                fill, gcol = (pal[2], pal[3]) if e else (pal[0], pal[1])
                glyph = ARROWS[vn.t_dir(s)]
            elif vn.is_C(s):
                out = (s - vn.C0) >> 1
                fill, gcol = (COL_CONF[2], COL_CONF[3]) if out else (COL_CONF[0], COL_CONF[1])
                glyph = LETTER_C
            elif vn.is_S(s):
                fill, gcol = COL_SENS
                glyph = LETTER_S
            else:
                fill = COL_U
            cv.rect(y, x, CELL - 1, CELL - 1, fill)
            if glyph is not None:
                gy = y + (CELL - 1 - glyph.shape[0]) // 2
                gx = x + (CELL - 1 - glyph.shape[1]) // 2
                cv.blit(glyph, gcol, gy, gx)


def _vn_frames(rows: int, cols: int) -> Iterator[Frame]:
    g = vn.construction_stair(SIM_R, SIM_C, **MACHINE)
    c0 = MACHINE["c0"]
    rng = np.random.default_rng(4)
    appear = rng.integers(0, 20, size=g.shape)
    machine = g != vn.U
    appear[machine] = 20 + rng.integers(0, 10, size=int(machine.sum()))
    built_prev = vn.arm_tip(g, *ARM_START)[1]
    i_off = _vn_offscreen_frame()
    n_vn = i_off + VN_TAIL
    zh_en = sorted(VN_INTRO + [(i_off + k, z, e) for k, z, e in VN_OUTRO])
    for i in range(n_vn):
        for _ in range(_vn_steps(i)):
            g = vn.step(g)
        cv = Canvas(rows, cols)
        _draw_vn(cv, g, appear <= i)
        (tr, tc), n_built = vn.arm_tip(g, *ARM_START)
        on_screen = tr < GR and tc < GC
        tip_state = int(g[tr, tc]) if (tr < SIM_R and tc < SIM_C) else vn.U
        if 108 <= i and on_screen:     # highlight the cell under construction
            cv.outline(GY + tr * CELL - 1, tc * CELL - 1, CELL + 1, CELL + 1, CYAN)
        zh, en = "", ""
        for start, z, e in zh_en:
            if i >= start:
                zh, en = z, e
        f = cv.to_frame(zh, en)
        # legend: 1 + 8 + 8 + 4 + 8 = 29
        lk = smoothstep((i - 6) / 30)
        if lk > 0:
            items = [f"{sym} {name} {cnt}" for sym, _, name, cnt in LEGEND]
            total = sum(text_width(s) for s in items) + 4 * (len(items) - 1) + 30
            c = cols // 2 - total // 2
            c = f.put_text(1, c, "29 种状态 29 states：", tuple(int(v) for v in scale_rgb(TEXT, lk))) + 2
            for (sym, col, name, cnt), txt in zip(LEGEND, items):
                c = f.put_text(1, c, sym, tuple(int(v) for v in scale_rgb(col, lk)))
                c = f.put_text(1, c, " " + name + " ", tuple(int(v) for v in scale_rgb(DIM, lk)))
                c = f.put_text(1, c, cnt, tuple(int(v) for v in scale_rgb(TEXT, lk))) + 4
            put_center(f, 3, "1 + 8 + 8 + 4 + 8 = 29", scale_rgb(DIM, lk))
        # labels (in the empty lower-left, clear of the staircase)
        if i >= 108:
            lr = (GY + 2 * CELL) // 2 + 1
            f.put_text(lr, c0 * CELL, "↑ 信号环 loop · 汇合态 C", COL_ORD[1])
            f.put_text(lr + 1, c0 * CELL, "tape: 10000 (→) · 1010 (↓)", DIM)
            f.put_text((GY + 5 * CELL) // 2, c0 * CELL, "构造臂 construction arm ↘", TEXT)
            f.put_text((GY + 5 * CELL) // 2 + 1, c0 * CELL, f"已建造 built: {max(0, n_built - 2)}", DIM)
        if i >= 168 and on_screen:
            name = vn.S_NAMES[tip_state - vn.S_BASE] if vn.is_S(tip_state) else ("U" if tip_state == vn.U else "")
            if name:
                f.put_text((GY + tr * CELL + 4) // 2, (tc + 1) * CELL + 1, name, (215, 190, 255))
        if n_built > built_prev:
            f.sfx.append(("pop", 0.3 + 0.05 * (n_built % 10)))
        elif _vn_steps(i) and i < 168:
            f.sfx.append(("tick", 0.3))
        built_prev = n_built
        if i >= n_vn - 24:  # fade out
            k = (n_vn - 1 - i) / 23
            f.colors[:] = scale_rgb(f.colors, k)
            f.bg[:] = scale_rgb(f.bg, k)
        yield f


def build_scenes(rows: int, cols: int) -> List[Scene]:
    return [
        Scene("ch1-opening", lambda: _title_frames(rows, cols), fps=FPS, transition="cut", music="title"),
        title_card(1, "起源", "Origin", ACCENT, music="calm", transition="cut"),
        portrait_scene(
            "ch1-neumann", music="calm", transition="dissolve",
            name="neumann", zh_name="约翰·冯·诺依曼", en_name="John von Neumann", years="1903 – 1957",
            facts=[
                ("1940 年代与乌拉姆一起发明了元胞自动机", "Invented cellular automata with Stanislaw Ulam, 1940s"),
                ("他想回答：机器能否自我复制？", "His question: can a machine build a copy of itself?"),
                ("并设计出一台 29 种状态的自复制机器", "He designed a 29-state self-reproducing automaton"),
            ],
            captions=[
                (0, "1940 年代，冯·诺依曼和乌拉姆发明了元胞自动机",
                 "In the 1940s, John von Neumann and Stanislaw Ulam invented cellular automata"),
                (100, "最初的问题：一台机器，能不能像生物一样复制自己？",
                 "The original question: can a machine reproduce itself, like a living thing?"),
            ],
            n_frames=204, credit="photo: LANL",
        ),
        Scene("ch1-vonneumann", lambda: _vn_frames(rows, cols), fps=FPS, transition="iris",
              transition_ticks=16, music="curious"),
    ]
