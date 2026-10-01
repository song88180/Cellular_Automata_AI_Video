"""Chapter 3: Conway's Game of Life (B3/S23).

Beats: title card -> Conway portrait -> the 8-cell Moore neighbourhood ->
the four rules on 3x3 examples -> a larger grid showing every cell's live-
neighbour count and who will be born / die -> still lifes & oscillators ->
glider (4 phases) & LWSS -> Gosper glider gun (zooming out) with honest notes
on Turing completeness / Gemini -> a random soup over the whole canvas.
Cells are coloured by age (newborn = bright yellow, old = cool blue); dying
cells leave a fading violet afterglow.
"""
from typing import Iterator, List, Optional, Tuple

import numpy as np
from scipy import ndimage

from chapters.common import portrait_scene, title_card
from engine.canvas import (AMBER, BG, CYAN, DIM, FAINT, GREEN, PH, PW, RED, TEXT, WHITE, Canvas,
                           ease_in_out, geom, lerp_rgb, pixel_text, put_center, put_right, scale_rgb)
from engine.frame import Frame, Scene
from sim.game_of_life import neighbor_count, pattern, place, step

CHAPTER_LABEL = ("第 3 章 · 生命游戏", "Chapter 3 · Game of Life")

ACCENT = AMBER
DEAD_RGB = (34, 36, 50)          # empty cell squares in the explainer grids
TRAIL_RGB = np.array((150, 80, 230), np.float32)

AGE_STOPS = [0, 3, 10, 30]
AGE_RGB = np.array([(255, 245, 170), (255, 170, 60), (230, 70, 140), (90, 110, 255)], dtype=float)


def age_colors(age: np.ndarray) -> np.ndarray:
    out = np.empty(age.shape + (3,), dtype=np.uint8)
    for ch in range(3):
        out[..., ch] = np.interp(age, AGE_STOPS, AGE_RGB[:, ch]).astype(np.uint8)
    return out


def _c(rgb) -> Tuple[int, int, int]:
    return tuple(int(v) for v in rgb)


class LifeRun:
    """A Life grid with per-cell age and a decaying afterglow of dead cells."""

    def __init__(self, grid: np.ndarray, trail_decay: float = 0.72):
        self.grid = grid.astype(bool)
        self.age = np.zeros(grid.shape, dtype=np.int32)
        self.trail = np.zeros(grid.shape, dtype=np.float32)
        self.decay = trail_decay
        self.gen = 0

    def advance(self, n: int = 1) -> None:
        for _ in range(n):
            new = step(self.grid)
            died = self.grid & ~new
            self.trail *= self.decay
            self.trail[died] = 1.0
            self.trail[new] = 0.0
            self.age = np.where(new & self.grid, self.age + 1, 0)
            self.grid = new
            self.gen += 1

    def layer(self) -> Tuple[np.ndarray, np.ndarray]:
        """(rgb, filled) for drawing: live cells by age, dead cells by afterglow."""
        live = age_colors(self.age)
        glow = (np.array(BG, np.float32) + (TRAIL_RGB - np.array(BG, np.float32)) * self.trail[..., None] * 0.4)
        rgb = np.where(self.grid[..., None], live, glow.astype(np.uint8))
        return rgb, self.grid | (self.trail > 0.06)


def _solid(h: int, w: int, rgb) -> np.ndarray:
    return np.broadcast_to(np.array(rgb, np.uint8), (h, w, 3)).copy()


def _cell_char(y0: int, x0: int, cell: int, r: int, c: int) -> Tuple[int, int]:
    """Character (row, col) at the centre of cell (r, c) of a grid drawn at
    pixel (y0, x0) with `cell` px per cell (y0 and cell even)."""
    return (y0 + r * cell + cell // 2 - 2) // 2, x0 + c * cell + (cell - 1) // 2


# --- Scene: the Moore neighbourhood -------------------------------------------------

def _cell_digit(cv: Canvas, y0: int, x0: int, cell: int, r: int, c: int, text: str, rgb) -> None:
    """Big pixel-font digit centred in cell (r, c) of a grid drawn at (y0, x0)."""
    m = pixel_text(text, 2)
    inner = cell - 1
    cv.blit(m, rgb, y0 + r * cell + (inner - m.shape[0]) // 2, x0 + c * cell + (inner - m.shape[1]) // 2)


def _neighbourhood_frames(rows: int, cols: int) -> Iterator[Frame]:
    cell, y0, x0 = 18, 10, 16
    order = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0)]
    example = {(0, 1), (1, 2), (2, 0)}  # live neighbours in the counting example
    n = 90
    for i in range(n):
        cv = Canvas(rows, cols)
        lit = min(8, max(0, (i - 12) // 4 + 1)) if i >= 12 else 0
        counting = i >= 54
        rgb = _solid(3, 3, DEAD_RGB)
        for k, (r, c) in enumerate(order):
            if counting:
                rgb[r, c] = AMBER if (r, c) in example else DEAD_RGB
            elif k < lit:
                rgb[r, c] = scale_rgb(CYAN, 0.75)
        rgb[1, 1] = WHITE if not counting else scale_rgb(CYAN, 0.35)
        cv.cells(rgb, np.ones((3, 3), bool), y0, x0, cell)
        cv.outline(y0 + cell - 1, x0 + cell - 1, cell + 1, cell + 1, CYAN)
        if not counting:
            for k, (r, c) in enumerate(order[:lit]):
                _cell_digit(cv, y0, x0, cell, r, c, str(k + 1), BG)
        else:
            for j, (r, c) in enumerate(sorted(example)[:min(3, (i - 54) // 8)]):
                _cell_digit(cv, y0, x0, cell, r, c, str(j + 1), BG)
        # 1D reminder, top right
        cv.cells(_solid(1, 3, (70, 90, 120)), np.ones((1, 3), bool), 14, 104, 8)
        cv.rect(14, 112, 7, 7, WHITE)
        f = cv.to_frame()
        f.put_text(3, 98, "一维 1D：左右 2 个邻居", DIM)
        f.put_text(4, 98, "left + right = 2 neighbours", FAINT)
        f.put_text(14, 98, "二维 2D：周围 8 个邻居", TEXT)
        f.put_text(15, 98, "8 neighbours (Moore neighbourhood)", DIM)
        f.put_text(16, 98, "上下左右 + 4 个对角", DIM)
        if i == 12 + 4 * 7:
            f.sfx.append(("chime", 0.7))
        if 12 <= i < 12 + 4 * 8 and (i - 12) % 4 == 0:
            f.sfx.append(("blip", 0.3 + 0.08 * ((i - 12) // 4)))
        if counting:
            k = min(3, (i - 54) // 8)
            f.put_text(21, 98, "数一数活邻居：", TEXT)
            f.put_text(22, 98, "count the live neighbours", DIM)
            f.put_text(24, 98, "  ".join(["●"] * k) + ("  = " + str(k) if k else ""), AMBER)
            if (i - 54) % 8 == 0 and (i - 54) // 8 < 4 and i > 54:
                f.sfx.append(("blip", 0.6))
            f.put_text(27, 98, "下一代的状态只取决于这个数字与自身状态", TEXT)
            f.put_text(28, 98, "the next state depends only on this number", DIM)
            f.put_text(29, 98, "and on the cell's own state", DIM)
        put_center(f, y0 // 2 + 3 * cell // 2 + 1, "中心 centre", DIM, col_center=x0 + 3 * cell // 2)
        yield f


# --- Scene: the four rules on 3x3 examples --------------------------------------------

RULES = [
    # (zh title, en title, colour, before 3x3, centre after, zh line, en line)
    ("诞生", "BIRTH", GREEN, ["#..", "..#", ".#."], True, "死格 + 恰好 3 个活邻居", "dead + exactly 3 -> born"),
    ("存活", "SURVIVAL", CYAN, ["#..", ".#.", "..#"], True, "活格 + 2 或 3 个活邻居", "alive + 2 or 3 -> stays"),
    ("孤独", "LONELINESS", (140, 150, 255), ["...", ".#.", "..#"], False, "活格 + 少于 2 个", "alive + fewer than 2 -> dies"),
    ("拥挤", "OVERCROWDING", RED, ["#.#", ".#.", "#.#"], False, "活格 + 多于 3 个", "alive + more than 3 -> dies"),
]
RULE_FRAMES = 44   # per rule while it is being introduced (12 fps)


def _rule_panel(cv: Canvas, f: Optional[Frame], k: int, local: int, active: bool) -> None:
    """Draw panel k. local = frames since this panel started (large = finished)."""
    zh, en, col, before, after_alive, zl, el = RULES[k]
    x0 = k * 40
    cell = 5
    by, bx = 22, x0 + 3
    ax = x0 + 23
    g = np.array([[ch == "#" for ch in row] for row in before])
    nb = int(g.sum() - g[1, 1])
    appear = local >= 0
    if not appear:
        return
    if cv is not None:
        tint = scale_rgb(col, 1.0 if active else 0.55)
        cv.font_px(zh, 5, x0 + 20, 9, tint, align="center")
        rgb = np.where(g[..., None], np.array(AMBER, np.uint8), np.array(DEAD_RGB, np.uint8)).astype(np.uint8)
        counted = min(nb, max(0, (local - 10) // 4))
        # highlight the neighbours counted so far
        seen = 0
        for r in range(3):
            for c in range(3):
                if (r, c) != (1, 1) and g[r, c]:
                    if seen < counted:
                        rgb[r, c] = lerp_rgb(AMBER, WHITE, 0.45)
                    seen += 1
        cv.cells(rgb, np.ones((3, 3), bool), by, bx, cell)
        cv.outline(by + cell - 1, bx + cell - 1, cell + 1, cell + 1, scale_rgb(col, 0.9))
        if local >= 30:
            flash = max(0.0, 1 - (local - 30) / 6)
            argb = np.where(g[..., None], np.array(scale_rgb(AMBER, 0.3), np.uint8),
                            np.array(DEAD_RGB, np.uint8)).astype(np.uint8)
            centre = AMBER if after_alive else DEAD_RGB
            argb[1, 1] = lerp_rgb(centre, WHITE, flash)
            cv.cells(argb, np.ones((3, 3), bool), by, ax, cell)
            cv.outline(by + cell - 1, ax + cell - 1, cell + 1, cell + 1, scale_rgb(col, 0.9))
        if active:
            cv.outline(2, x0 + 1, 76, 38, scale_rgb(col, 0.35))
    if f is not None:
        put_center(f, 9, en, _c(scale_rgb(col, 0.9 if active else 0.5)), col_center=x0 + 20)
        f.put_text(14, x0 + 19, "→", TEXT if local >= 30 else FAINT)
        counted = min(nb, max(0, (local - 10) // 4))
        if local >= 10:
            f.put_text(21, x0 + 3, f"活邻居 = {counted}", AMBER)
            f.put_text(22, x0 + 3, f"live nbrs = {counted}", DIM)
        if local >= 30:
            born = after_alive and not g[1, 1]
            f.put_text(21, x0 + 23, "诞生!" if born else ("存活" if after_alive else "死亡"), _c(col))
            f.put_text(22, x0 + 23, "born!" if born else ("survives" if after_alive else "dies"), DIM)
            f.put_text(24, x0 + 2, zl, TEXT)
            f.put_text(25, x0 + 2, el, DIM)


def _rules_frames(rows: int, cols: int) -> Iterator[Frame]:
    total = 4 * RULE_FRAMES + 60
    for i in range(total):
        cv = Canvas(rows, cols)
        k_active = i // RULE_FRAMES if i < 4 * RULE_FRAMES else -1
        for k in range(4):
            _rule_panel(cv, None, k, i - k * RULE_FRAMES, k == k_active)
        summary = i >= 4 * RULE_FRAMES
        if summary:
            m = pixel_text("B3/S23", 2)
            cv.blit(m, AMBER, 53, (PW - m.shape[1]) // 2)
        f = cv.to_frame()
        for k in range(4):
            _rule_panel(None, f, k, i - k * RULE_FRAMES, k == k_active)
            local = i - k * RULE_FRAMES
            if local == 30:
                f.sfx.append(("pop", 0.7) if RULES[k][4] else ("blip", 0.15))
            if 10 <= local < 30 and (local - 10) % 4 == 0:
                f.sfx.append(("tick", 0.5))
        if summary:
            put_center(f, 35, "B = 诞生：需要 3 个活邻居        S = 存活：需要 2 或 3 个活邻居", TEXT)
            put_center(f, 36, "B = Born: exactly 3 live neighbours        S = Survive: 2 or 3 live neighbours", DIM)
            put_center(f, 38, "其余情况：死亡或保持空白", TEXT)
            put_center(f, 39, "everything else dies or stays empty", DIM)
            if i == 4 * RULE_FRAMES:
                f.sfx.append(("chime", 0.9))
        yield f


# --- Scene: counts on a larger grid ------------------------------------------------------

CG_W, CG_H, CG_CELL, CG_Y, CG_X = 16, 9, 8, 4, 4
CG_PAD = 8


def _count_grid_start() -> np.ndarray:
    g = np.zeros((CG_H + 2 * CG_PAD, CG_W + 2 * CG_PAD), dtype=bool)
    p = CG_PAD
    # placed (by search) so the three never interact over the whole demo
    place(g, pattern("glider"), p + 1, p + 0)
    place(g, pattern("blinker"), p + 1, p + 6)
    g[p + 5, p + 12: p + 15] = True        # T-tetromino -> 'traffic light'
    g[p + 6, p + 13] = True
    return g


def _count_grid_frame(rows, cols, g, show_counts: bool, show_fate: bool, flash: float,
                      changed: Optional[np.ndarray], gen: int) -> Frame:
    p = CG_PAD
    view = (slice(p, p + CG_H), slice(p, p + CG_W))
    alive = g[view]
    cnt = neighbor_count(g)[view]
    new = step(g)[view]
    born, dies = new & ~alive, alive & ~new
    rgb = np.where(alive[..., None], np.array(AMBER, np.uint8), np.array(DEAD_RGB, np.uint8)).astype(np.uint8)
    if show_fate:
        rgb[born] = scale_rgb(GREEN, 0.55)
        rgb[dies] = RED
    if changed is not None and flash > 0:
        rgb[changed] = lerp_rgb(rgb[changed], WHITE, flash)
    cv = Canvas(rows, cols)
    cv.cells(rgb, np.ones_like(alive), CG_Y, CG_X, CG_CELL)
    f = cv.to_frame()
    if show_counts:
        for r in range(CG_H):
            for c in range(CG_W):
                n = int(cnt[r, c])
                if n == 0 and not alive[r, c]:
                    continue
                rr, cc = _cell_char(CG_Y, CG_X, CG_CELL, r, c)
                bgc = _c(rgb[r, c])
                fg = _c(BG) if (alive[r, c] or (show_fate and born[r, c])) else _c(DIM)
                f.put_text(rr, cc, str(n), fg, bg=bgc)
    lx = CG_X + CG_W * CG_CELL + 4
    f.put_text(3, lx, f"第 {gen} 代", TEXT)
    f.put_text(4, lx, f"generation {gen}", DIM)
    f.put_text(8, lx, "数字 = 活邻居数", TEXT)
    f.put_text(9, lx, "number = live nbrs", DIM)
    f.put_text(12, lx, "■", AMBER)
    f.put_text(12, lx + 2, "活", TEXT)
    f.put_text(13, lx + 2, "alive", DIM)
    if show_fate:
        f.put_text(15, lx, "■", _c(scale_rgb(GREEN, 0.8)))
        f.put_text(15, lx + 2, "将诞生 (=3)", TEXT)
        f.put_text(16, lx + 2, "will be born", DIM)
        f.put_text(18, lx, "■", RED)
        f.put_text(18, lx + 2, "将死亡 (<2,>3)", TEXT)
        f.put_text(19, lx + 2, "will die", DIM)
        f.put_text(21, lx, "■", AMBER)
        f.put_text(21, lx + 2, "存活 (2,3)", TEXT)
        f.put_text(22, lx + 2, "survives", DIM)
    return f


def _count_grid_frames(rows: int, cols: int) -> Iterator[Frame]:
    g = _count_grid_start()
    gen = 0
    # slow generations: counts -> fates -> apply (with flash)
    for slow in range(3):
        a, b, c = (26, 26, 12) if slow == 0 else (16, 18, 10)
        for i in range(a):
            f = _count_grid_frame(rows, cols, g, True, False, 0, None, gen)
            if i == 0:
                f.sfx.append(("tick", 0.6))
            yield f
        for i in range(b):
            f = _count_grid_frame(rows, cols, g, True, True, 0, None, gen)
            if i == 0:
                f.sfx.append(("blip", 0.5))
            yield f
        new = step(g)
        changed = (new != g)[CG_PAD: CG_PAD + CG_H, CG_PAD: CG_PAD + CG_W]
        g, gen = new, gen + 1
        for i in range(c):
            f = _count_grid_frame(rows, cols, g, i >= c // 2, False, 1 - i / c, changed, gen)
            if i == 0:
                f.sfx.append(("pop", 0.6))
            yield f
    # faster: 4 frames per generation
    for _ in range(12):
        for i in range(4):
            yield _count_grid_frame(rows, cols, g, True, i >= 2, 0, None, gen)
        g, gen = step(g), gen + 1


# --- Scene: still lifes & oscillators ------------------------------------------------------

ZOO_STILL = [("block", "方块 block"), ("beehive", "蜂巢 beehive"), ("loaf", "面包 loaf"), ("boat", "小船 boat")]
ZOO_OSC = [("blinker", "闪光灯", "blinker", 2), ("toad", "蟾蜍", "toad", 2), ("beacon", "信标", "beacon", 2),
           ("pulsar", "脉冲星", "pulsar", 3), ("pentadecathlon", "十五项", "pentadecathlon", 15)]


class _Specimen:
    SIM_MARGIN = 4  # extra hidden border so the torus never wraps onto the specimen

    def __init__(self, name: str, pad: int):
        pat = pattern(name)
        self.run = LifeRun(np.pad(pat, pad + self.SIM_MARGIN), trail_decay=0.55)
        m = self.SIM_MARGIN
        self.view = (slice(m, -m), slice(m, -m))
        self.shape = self.run.grid[self.view].shape

    def draw(self, cv: Canvas, y: int, x: int, cell: int, colour=None) -> None:
        rgb, filled = self.run.layer()
        rgb, filled = rgb[self.view], filled[self.view]
        if colour is not None:
            rgb = np.where(self.run.grid[self.view][..., None], np.array(colour, np.uint8), rgb).astype(np.uint8)
        cv.cells(rgb, filled, y, x, cell)


def _zoo_frames(rows: int, cols: int) -> Iterator[Frame]:
    still = [_Specimen(n, 1) for n, _ in ZOO_STILL]
    osc = [_Specimen(n, 3 if n == "pentadecathlon" else 1) for n, *_ in ZOO_OSC]
    # layout (pixel x of each specimen); oscillator bounding boxes include all phases
    sx = [16, 52, 90, 124]
    ox = [2, 22, 46, 70, 112]
    ocell = [3, 3, 3, 2, 3]
    n = 60
    for i in range(n):
        cv = Canvas(rows, cols)
        for sp, x in zip(still, sx):
            sp.draw(cv, 6, x, 4, colour=CYAN)
        for k, (sp, x) in enumerate(zip(osc, ox)):
            h, w = sp.shape
            y = 36 + (36 - h * ocell[k]) // 2
            sp.draw(cv, y, x, ocell[k])
        cv.rect(33, 4, 1, PW - 8, FAINT)
        f = cv.to_frame()
        f.put_text(1, 2, "静物 still lifes：永远不变 never change", TEXT)
        for (name, label), x, sp in zip(ZOO_STILL, sx, still):
            w = sp.shape[1] * 4
            put_center(f, 13, label, DIM, col_center=x + w // 2)
        f.put_text(17, 2, "振荡器 oscillators：按固定周期 p 重复 repeat every p generations", TEXT)
        for (name, zh, en, per), x, sp, cell in zip(ZOO_OSC, ox, osc, ocell):
            w = sp.shape[1] * cell
            put_center(f, 37, zh, TEXT, col_center=x + w // 2)
            put_center(f, 38, en, DIM, col_center=x + w // 2)
            phase = sp.run.gen % per
            put_center(f, 39, f"p{per} · {phase + 1}/{per}", AMBER, col_center=x + w // 2)
        yield f
        if i % 2 == 1:
            for sp in osc:
                sp.run.advance()
            if i % 6 == 5:
                f.sfx.append(("tick", 0.3))


# --- Scene: glider & LWSS ---------------------------------------------------------------------

def _glider_frames(rows: int, cols: int) -> Iterator[Frame]:
    base = np.zeros((6, 6), dtype=bool)
    place(base, pattern("glider"), 0, 0)
    phases = [base]
    for _ in range(4):
        phases.append(step(phases[-1]))
    # Bottom strip (14 x 53 cells visible): two gliders and a lightweight
    # spaceship on paths that never come near each other (the LWSS starts
    # ahead and outruns both gliders), inside a hidden margin so nothing wraps
    # around the torus back into view.
    VH, VW, M = 14, 53, 16
    fly = np.zeros((VH + 2 * M, VW + 2 * M), dtype=bool)
    place(fly, pattern("glider"), M + 0, M + 12)             # flies down-right
    place(fly, pattern("glider")[::-1], M + 11, M + 2)       # mirrored: flies up-right
    place(fly, pattern("lwss")[:, ::-1], M + 5, M + 20)      # mirrored: flies right
    run = LifeRun(fly, trail_decay=0.8)
    view = (slice(M, M + VH), slice(M, M + VW))
    cell, py, gap = 4, 6, 8
    n = 40 + 64
    for i in range(n):
        cv = Canvas(rows, cols)
        shown = min(5, i // 6 + 1)
        for k in range(shown):
            x = 6 + k * (6 * cell + gap)
            rgb = _solid(6, 6, DEAD_RGB)
            rgb[phases[k]] = AMBER
            if k == 4:  # ghost of generation 0 for comparison
                ghost = phases[0] & ~phases[4]
                rgb[ghost] = scale_rgb(CYAN, 0.35)
            cv.cells(rgb, np.ones((6, 6), bool), py, x, cell)
            if k == 4:
                cv.outline(py - 1, x - 1, 6 * cell + 1, 6 * cell + 1, AMBER)
        if i >= 40:
            if i > 40:
                run.advance(1)
            rgb, filled = run.layer()
            cv.cells(rgb[view], filled[view], 38, 1, 3)
        f = cv.to_frame()
        for k in range(shown):
            x = 6 + k * (6 * cell + gap)
            put_center(f, 16, f"第 {k} 代 gen {k}", AMBER if k == 4 else DIM, col_center=x + 12)
            if k < 4:
                f.put_text(8, x + 6 * cell + 2, "→", DIM)
        if shown == 5:
            put_center(f, 17, "= 第 0 代 ↘ 移动 1 格", CYAN, col_center=6 + 4 * (6 * cell + gap) + 12)
        if i % 6 == 0 and i // 6 < 5:
            f.sfx.append(("blip", 0.4 + 0.1 * (i // 6)))
        if i == 30:
            f.sfx.append(("chime", 0.8))
        if i >= 40:
            f.put_text(18, 2, "滑翔机 gliders ↘ ↗   ·   轻型飞船 LWSS → （每 4 代前进 2 格 2 cells / 4 gens）", DIM)
            put_right(f, 18, cols - 2, f"第 {run.gen} 代 generation {run.gen}", FAINT)
        yield f


# --- Scene: Gosper glider gun ------------------------------------------------------------------

GUN_BOX = (slice(8, 22), slice(8, 48))   # cells occupied by the gun itself (+ margin)


def _gun_frames(rows: int, cols: int) -> Iterator[Frame]:
    g = np.zeros((160, 220), dtype=bool)
    place(g, pattern("gosper_glider_gun"), 10, 10)
    run = LifeRun(g, trail_decay=0.78)
    n = 12 * 22
    fired = 0
    for i in range(n):
        t = i / (n - 1)
        z = ease_in_out((t - 0.12) / 0.55)
        scale = geom(4.0, 1.8, z)
        src = (14 + (28 - 14) * z, 28 + (48 - 28) * z)
        cv = Canvas(rows, cols)
        rgb, filled = run.layer()
        cv.zoom(rgb, filled, scale, src_center=src, dst_center=(PH / 2, PW / 2))
        f = cv.to_frame()
        # count gliders already clear of the gun
        outside = run.grid.copy()
        outside[GUN_BOX] = False
        lab, k = ndimage.label(outside, structure=np.ones((3, 3)))
        if k > fired:
            f.sfx.append(("pop", 0.55 + 0.05 * (k % 5)))
            fired = k
        f.put_text(34, 2, f"第 {run.gen} 代  gen {run.gen}", DIM)
        f.put_text(36, 2, f"已发射滑翔机 gliders fired: {fired}", AMBER)
        f.put_text(38, 2, f"细胞数 population: {int(run.grid.sum())}", DIM)
        if i == 0:
            f.sfx.append(("hit", 0.8))
        yield f
        run.advance()


# --- Scene: random soup -------------------------------------------------------------------------

def _soup_frames(rows: int, cols: int) -> Iterator[Frame]:
    rng = np.random.default_rng(7)
    h, w = PH, PW
    run = LifeRun(rng.random((h, w)) < 0.32, trail_decay=0.8)
    n = 12 * 18
    for i in range(n):
        t = i / (n - 1)
        scale = geom(3.0, 1.0, ease_in_out(t / 0.55))
        cv = Canvas(rows, cols)
        rgb, filled = run.layer()
        cv.zoom(rgb, filled, scale, src_center=(h / 2, w / 2), dst_center=(PH / 2, PW / 2), gap=True)
        f = cv.to_frame()
        f.put_text(0, 1, f" 第 {run.gen} 代  gen {run.gen} ", TEXT, bg=_c(BG))
        f.put_text(1, 1, f" 活细胞 alive: {int(run.grid.sum())} ", DIM, bg=_c(BG))
        if i == 0:
            f.sfx.append(("hit", 1.0))
        yield f
        run.advance(1 if i < 110 else 2)


# --- captions -------------------------------------------------------------------------------------

def _captioned(frames: Iterator[Frame], cues: List[Tuple[int, str, str]]) -> Iterator[Frame]:
    cues = sorted(cues)
    for i, f in enumerate(frames):
        zh, en = "", ""
        for start, z, e in cues:
            if i >= start:
                zh, en = z, e
        f.subtitle_zh, f.subtitle_en = zh, en
        yield f


def build_scenes(rows: int, cols: int) -> List[Scene]:
    return [
        title_card(3, "生命游戏", "Conway's Game of Life", ACCENT, music="curious", transition="fade"),
        portrait_scene(
            "ch3-conway", music="curious", transition="dissolve",
            name="conway", zh_name="约翰·康威", en_name="John H. Conway", years="1937 – 2020",
            facts=[("剑桥数学家，1970 年设计了「生命游戏」", "Cambridge mathematician; devised the Game of Life in 1970"),
                   ("Martin Gardner 在《科学美国人》专栏介绍后风靡全球",
                    "Made famous by Martin Gardner's Scientific American column (Oct 1970)"),
                   ("他想找：最简单、却无法预测的规则", "He sought the simplest rules whose future can't be predicted")],
            captions=[(0, "1970 年，数学家 John Conway 用纸笔和围棋子反复试验...",
                       "In 1970 mathematician John Conway experimented with paper and Go stones..."),
                      (100, "...找到了一套极其简单的二维规则：生命游戏",
                       "...and found an extremely simple 2D rule set: the Game of Life")],
            n_frames=24 * 8, accent=ACCENT, credit="photo: Denise Applewhite, Office of Communications",
        ),
        Scene("ch3-neighbourhood", lambda: _captioned(_neighbourhood_frames(rows, cols), [
            (0, "从一维到二维：格子铺满整个平面",
                "From 1D to 2D: now the cells tile a whole plane"),
            (40, "每个格子看周围 8 个邻居：上下左右，加上 4 个对角",
                 "Each cell looks at its 8 neighbours: 4 sides plus 4 diagonals"),
            (60, "规则只关心一件事：这 8 个邻居里有几个是活的",
                 "The rule cares about one thing only: how many of those 8 are alive"),
        ]), fps=12, transition="iris", transition_ticks=16, music="curious"),
        Scene("ch3-rules", lambda: _captioned(_rules_frames(rows, cols), [
            (0, "规则 1 · 诞生：死格周围恰好有 3 个活邻居，就会诞生",
                "Rule 1 · Birth: a dead cell with exactly 3 live neighbours comes alive"),
            (RULE_FRAMES, "规则 2 · 存活：活格有 2 或 3 个活邻居，就继续活着",
                          "Rule 2 · Survival: a live cell with 2 or 3 live neighbours stays alive"),
            (2 * RULE_FRAMES, "规则 3 · 孤独：活邻居少于 2 个，它会死去",
                              "Rule 3 · Loneliness: with fewer than 2 live neighbours it dies"),
            (3 * RULE_FRAMES, "规则 4 · 拥挤：活邻居多于 3 个，也会死去",
                              "Rule 4 · Overcrowding: with more than 3 live neighbours it also dies"),
            (4 * RULE_FRAMES, "全部规则就这四条，简写为 B3/S23",
                              "That's the whole rule set, written B3/S23"),
        ]), fps=12, transition="wipe", transition_ticks=16, music="curious"),
        Scene("ch3-counting", lambda: _captioned(_count_grid_frames(rows, cols), [
            (0, "在大一点的网格上：先数出每个格子的活邻居",
                "On a bigger grid: first count every cell's live neighbours"),
            (26, "按规则标出：绿色将诞生，红色将死亡",
                 "Apply the rule: green cells will be born, red cells will die"),
            (64, "所有格子同时更新，得到下一代",
                 "Every cell updates at the same moment, giving the next generation"),
            (126, "一代接一代：滑翔机在走，闪光灯在闪，T 形在长大",
                  "Generation after generation: a glider walks, a blinker blinks, a T grows"),
        ]), fps=12, transition="dissolve", transition_ticks=16, music="build"),
        Scene("ch3-zoo", lambda: _captioned(_zoo_frames(rows, cols), [
            (0, "有些图案一旦形成就再也不变：静物",
                "Some patterns never change once formed: still lifes"),
            (26, "有些图案周而复始：振荡器，周期从 2 到 15",
                 "Others cycle forever: oscillators, with periods from 2 to 15"),
        ]), fps=6, transition="rain", transition_ticks=16, music="calm"),
        Scene("ch3-glider", lambda: _captioned(_glider_frames(rows, cols), [
            (0, "滑翔机：只有 5 个细胞，却会「走路」",
                "The glider: just 5 cells, yet it walks"),
            (30, "每过 4 代，同样的形状出现在右下方 1 格处",
                 "Every 4 generations the same shape reappears 1 cell down and to the right"),
            (56, "还有水平方向飞行的「飞船」——信息可以在网格里传递",
                 "Spaceships fly horizontally too: information can travel across the grid"),
        ]), fps=8, transition="wipe_down", transition_ticks=14, music="wonder"),
        Scene("ch3-gun", lambda: _captioned(_gun_frames(rows, cols), [
            (0, "高斯帕滑翔机枪（Bill Gosper，1970）",
                "The Gosper glider gun (Bill Gosper, 1970)"),
            (40, "Conway 曾悬赏 50 美元：有限的图案能否无限增长？",
                 "Conway offered $50 for a finite pattern that grows forever"),
            (84, "答案是能：它每 30 代发射一架新的滑翔机，永不停歇",
                 "Yes: this one fires a new glider every 30 generations, forever"),
            (144, "滑翔机能搭出逻辑门：生命游戏是图灵完备的",
                  "Gliders can build logic gates: the Game of Life is Turing complete"),
            (204, "还有自我复制结构 Gemini（2010），但复制一次约需 3400 万代",
                  "There is even a self-replicator (Gemini, 2010), but one copy takes ~34 million gens"),
            (232, "它太庞大了，这里放不下；自我复制，留到第 5 章细看",
                  "It is far too big to show here; we'll watch self-replication in chapter 5"),
        ]), fps=12, transition="flash", transition_ticks=12, music="build"),
        Scene("ch3-soup", lambda: _captioned(_soup_frames(rows, cols), [
            (0, "最后：从一片完全随机的初始状态开始",
                "Finally: start from a completely random state"),
            (50, "很快，静物、振荡器和滑翔机从混沌中出现",
                 "Soon still lifes, oscillators and gliders appear out of the chaos"),
            (115, "没有人设计它们：它们从局部规则中自己「涌现」",
                  "Nobody designed them: they emerge from the local rule on their own"),
            (160, "四条规则，就足以造出一个生机勃勃的世界",
                  "Four rules are enough to create a whole bustling world"),
        ]), fps=12, transition="rule30", transition_ticks=18, music="epic"),
    ]
