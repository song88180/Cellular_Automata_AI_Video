"""Chapter 5: Langton's self-reproducing loops (1984).

Beats: title -> history (von Neumann's 29-state constructor -> Codd's 8
states -> Langton's 86-cell loop) -> close-up of one loop, one step per
frame, with a state legend, until it has copied itself (151 steps) -> zoom
out while the colony grows like coral; cells that have stopped changing are
dimmed, showing the dead, boxed-in loops in the interior -> Evoloop
(Sayama 1999): the same idea plus a 'dissolving' state, so loops collide,
mutate and evolve.
Simulation: sim/langton_loops.py, sim/evoloop.py (rule tables from Golly).
"""
from typing import Iterator, List, Tuple

import numpy as np

from chapters.common import title_card
from engine.canvas import (AMBER, DIM, FAINT, PH, PW, TEXT, Canvas, Captions, box, clamp01,
                           geom, hsv, put_center, scale_rgb, smoothstep)
from engine.frame import Frame, Scene
from sim import evoloop
from sim.langton_loops import PALETTE, PERIOD, initial_loop, step

CHAPTER_LABEL = ("第 5 章 · 自复制环", "Chapter 5 · Self-reproducing loops")

FPS = 12
ACCENT = (120, 170, 255)


def _rgb(c) -> Tuple[int, int, int]:
    return tuple(int(v) for v in c)


# --- history ---------------------------------------------------------------------

HIST_BEAT = 48
HIST_CAPS = Captions([
    (0, "机器能否自我复制？冯·诺依曼设计了一个 29 种状态的「通用构造机」",
        "Can a machine copy itself? Von Neumann designed a 29-state 'universal constructor'"),
    (HIST_BEAT, "它能按「说明书」造出任何机器，包括它自己；1966 年由 Burks 整理出版",
                "Following a 'tape' of instructions it can build anything, itself included (Burks, 1966)"),
    (2 * HIST_BEAT, "1968 年 Codd 把状态数降到 8 种，但结构依然庞大复杂",
                    "In 1968 Codd cut it to 8 states - but the machine was still huge"),
    (3 * HIST_BEAT, "1984 年 Langton 放弃「通用」，只要求复制自己：一个 86 格的环",
                    "In 1984 Langton dropped universality and kept only self-copying: an 86-cell loop"),
])
HIST_FRAMES = 4 * HIST_BEAT + 24

NODES = [
    # (appear frame, centre col, year, name, n_states, lines)
    (0, 27, "1940s – 1950s", "冯·诺依曼 von Neumann", 29,
     [("通用构造机", "universal constructor"), ("能造任何机器，包括自己", "builds anything, even itself"),
      ("1966 由 Burks 整理出版", "published by Burks, 1966")]),
    (2 * HIST_BEAT, 80, "1968", "E. F. Codd", 8,
     [("同样是通用构造机", "still a universal constructor"), ("状态更少，结构仍很庞大", "fewer states, still huge")]),
    (3 * HIST_BEAT, 133, "1984", "C. G. Langton", 8,
     [("不再通用：只会复制自己", "not universal: only copies itself"),
      ("86 个格子 · 每 151 步一次", "86 cells · a copy every 151 steps")]),
]


def _history_frames(rows: int, cols: int) -> Iterator[Frame]:
    loop = initial_loop()
    for i in range(HIST_FRAMES):
        cv = Canvas(rows, cols)
        texts = []
        for n, (t0, cc, year, name, n_states, lines) in enumerate(NODES):
            a = smoothstep((i - t0) / 12)
            if a <= 0:
                continue
            # state swatches: n_states squares, 8 per row, 4 px pitch
            per = 8
            x0 = cc - (per * 5 - 1) // 2
            for k in range(n_states):
                y, x = 18 + (k // per) * 5, x0 + (k % per) * 5
                cv.rect(y, x, 4, 4, scale_rgb(hsv(k / n_states, 0.65, 1.0), a))
            if n == 2:   # Langton: the actual loop, 2 px per cell
                img = PALETTE[loop]
                cv.cells(scale_rgb(img, a), loop > 0, 44, cc - 15, 2, gap=0)
            texts.append((a, cc, year, name, n_states, lines, n))
        # connecting arrows grow between nodes
        for n in range(2):
            p = smoothstep((i - NODES[n + 1][0] + 10) / 12)
            if p > 0:
                x0, x1 = NODES[n][1] + 22, NODES[n + 1][1] - 22
                cv.rect(24, x0, 1, int((x1 - x0) * p), scale_rgb(DIM, p))
        f = cv.to_frame()
        for a, cc, year, name, n_states, lines, n in texts:
            col = _rgb(scale_rgb(TEXT, a))
            put_center(f, 3, year, scale_rgb(ACCENT, a), cc)
            put_center(f, 5, name, col, cc)
            put_center(f, 7, f"{n_states} 种状态 · {n_states} states", scale_rgb(AMBER, a), cc)
            base = 34 if n == 2 else 26
            if n == 2:
                put_center(f, 21, "Langton 环 / Langton's loop", scale_rgb(DIM, a), cc)
            for j, (lz, le) in enumerate(lines):
                put_center(f, base + 2 * j, lz, col, cc)
                put_center(f, base + 2 * j + 1, le, scale_rgb(DIM, a), cc)
        for n, node in enumerate(NODES):
            if i == node[0]:
                f.sfx.append(("chime", 0.5 + 0.15 * n))
        HIST_CAPS.apply(f, i)
        yield f


# --- close-up: one loop copies itself ------------------------------------------------

CU_GRID = 60
CU_ORIGIN = (25, 20)
CU_CENTER = (27.5, 33.0)       # centre (cells) of the region used during 0..175 steps
CU_SCALE = 4.0
CU_REGION = (0, 0, PH, 116)
CU_INTRO, CU_STEPS, CU_HOLD = 30, PERIOD, 30   # stop exactly when the daughter is complete
CU_CAPS = Captions([
    (0, "环：红色的「鞘」包着蓝色的「核心」，核心里流动着「基因」信号",
        "The loop: a red 'sheath' wraps a blue 'core', along which 'gene' signals flow"),
    (CU_INTRO + 30, "信号成对前进：7-0 让伸出的臂长一格，4-0 让臂左转",
                    "Signals travel in pairs: 7-0 extends the arm by one cell, 4-0 turns it left"),
    (CU_INTRO + 84, "臂一边伸长一边左转，转满 4 次，绕回来与自己相接……",
                    "The arm keeps growing and turning left - four turns - until it meets itself..."),
    (CU_INTRO + PERIOD - 6, "……然后断开：151 步后，出现了一个完整的子环，它们都继续伸出新的臂",
                            "...and detaches: after 151 steps there is a complete daughter; both start new arms"),
])

LEGEND = [
    (1, "核心 core", "信号通道 signal path"),
    (2, "鞘 sheath", "外壳 wall"),
    (7, "7 + 0", "臂伸长一格 extend arm"),
    (4, "4 + 0", "臂左转 turn arm left"),
    (3, "3 · 5 · 6", "其他控制信号 other control"),
]


def _dst_of(r: float, c: float, dst) -> Tuple[float, float]:
    return dst[0] + (r - CU_CENTER[0]) * CU_SCALE, dst[1] + (c - CU_CENTER[1]) * CU_SCALE


def _closeup_frames(rows: int, cols: int) -> Iterator[Frame]:
    grid = np.zeros((CU_GRID, CU_GRID), dtype=np.uint8)
    L = initial_loop()
    grid[CU_ORIGIN[0]: CU_ORIGIN[0] + L.shape[0], CU_ORIGIN[1]: CU_ORIGIN[1] + L.shape[1]] = L
    dst = (PH / 2 - 2, CU_REGION[3] / 2 + 2)
    n_step = 0
    lx, ly = 120, 6
    for i in range(CU_INTRO + CU_STEPS + CU_HOLD):
        if CU_INTRO <= i < CU_INTRO + CU_STEPS:
            grid = step(grid)
            n_step += 1
        cv = Canvas(rows, cols)
        rgb = PALETTE[grid].copy()
        rgb[grid == 0] = (30, 32, 44)
        cv.zoom(rgb, np.ones(grid.shape, bool), CU_SCALE, CU_CENTER, dst, gap=True, region=CU_REGION)
        for k, (st, _, _) in enumerate(LEGEND):
            cv.rect(2 * (ly + 3 + 3 * k), lx, 2, 4, PALETTE[st])
            if st == 7 or st == 4:
                cv.rect(2 * (ly + 3 + 3 * k), lx + 5, 2, 4, (30, 32, 44))   # the '0' partner
        f = cv.to_frame()
        box(f, ly, lx - 3, 20, 41, FAINT, title="8 种状态 · 8 states", title_rgb=ACCENT)
        f.put_text(ly + 1, lx, "0 = 空 empty (暗格)", DIM)
        for k, (st, name, desc) in enumerate(LEGEND):
            r = ly + 3 + 3 * k
            f.put_text(r, lx + 11, name, TEXT)
            f.put_text(r + 1, lx + 11, desc, DIM)
        f.put_text(1, 2, "Langton's loop", _rgb(ACCENT))
        f.put_text(1, 18, f"step {n_step}", TEXT)
        f.put_text(2, 2, f"一次自我复制需 {PERIOD} 步 / one copy takes {PERIOD} steps", DIM)
        flash_k = 1.0 - (i - (CU_INTRO + PERIOD - 1)) / 5   # short flash as the daughter detaches
        if n_step == PERIOD and 0 < flash_k <= 1.0:
            if i == CU_INTRO + PERIOD - 1:
                f.sfx.append(("hit", 1.0))
            add = int(80 * flash_k)
            f.colors[:] = np.clip(f.colors.astype(np.int16) + add, 0, 255).astype(np.uint8)
            f.bg[:] = np.clip(f.bg.astype(np.int16) + add, 0, 255).astype(np.uint8)   # both half-pixels
        if n_step >= PERIOD:
            a = _rgb(scale_rgb(AMBER, smoothstep((n_step - PERIOD) / 8)))
            y, x = _dst_of(35.5, 25.0, dst)
            put_center(f, int(y // 2) + 1, "母环 mother", a, int(x))
            y, x = _dst_of(35.5, 36.0, dst)
            put_center(f, int(y // 2) + 1, "子环 daughter", a, int(x))
        elif CU_INTRO <= i and i % 6 == 0:
            f.sfx.append(("tick", 0.3))
        CU_CAPS.apply(f, i)
        yield f


# --- colony --------------------------------------------------------------------------

COL_GRID = 420
COL_FRAMES, COL_HOLD = 192, 24
COL_START = PERIOD     # the colony carries on exactly from the close-up's last step
COL_END = 2100
DEAD_AFTER = 160       # steps without change -> drawn as a dim 'dead' shell
COL_CAPS = Captions([
    (0, "每个子环又继续复制，一代接一代……",
        "Each daughter copies itself too, generation after generation..."),
    (42, "群落像珊瑚一样向外生长：只有边缘的环还有空间繁殖",
         "The colony grows outward like coral: only loops at the edge have room to breed"),
    (102, "内部的环被挤住，信号停止，只剩暗淡的空壳",
          "Loops inside are boxed in: their signals stop, leaving dim, dead shells"),
    (156, "没有 DNA，没有化学：只有 8 种状态和 219 条局部规则",
          "No DNA, no chemistry: just 8 states and 219 local rules"),
])


def _colony_frames(rows: int, cols: int) -> Iterator[Frame]:
    grid = np.zeros((COL_GRID, COL_GRID), dtype=np.uint8)
    L = initial_loop()
    o = (COL_GRID // 2 - 5, COL_GRID // 2 - 7)
    grid[o[0]: o[0] + L.shape[0], o[1]: o[1] + L.shape[1]] = L
    last_change = np.zeros(grid.shape, dtype=np.int32)
    n = 0
    for _ in range(COL_START):
        new = step(grid)
        n += 1
        last_change[new != grid] = n
        grid = new
    t = np.linspace(0, 1, COL_FRAMES)
    targets = (COL_START + (COL_END - COL_START) * t ** 1.5).astype(int)
    targets = np.concatenate([targets, np.full(COL_HOLD, COL_END)])
    # start exactly where the close-up left off (same centre, scale and screen anchor)
    center0 = (o[0] + CU_CENTER[0] - CU_ORIGIN[0], o[1] + CU_CENTER[1] - CU_ORIGIN[1])
    dst0 = (PH / 2 - 2, CU_REGION[3] / 2 + 2)
    scale = CU_SCALE
    cy, cx = center0
    for i, tgt in enumerate(targets):
        while n < tgt:
            new = step(grid)
            n += 1
            last_change[new != grid] = n
            grid = new
        ys, xs = np.nonzero(grid)
        by, bx = (ys.min() + ys.max() + 1) / 2, (xs.min() + xs.max() + 1) / 2
        bh, bw = (ys.max() - ys.min() + 1) * 1.15, (xs.max() - xs.min() + 1) * 1.15
        target = float(np.clip(min((PH - 4) / bh, (PW - 4) / bw), 0.3, CU_SCALE))
        k = 0.2 * smoothstep(i / 24)
        scale = float(np.exp(np.log(scale) * (1 - k) + np.log(target) * k))
        cy, cx = cy * (1 - k) + by * k, cx * (1 - k) + bx * k
        rgb = PALETTE[grid].copy()
        # a cell counts as 'alive' if anything within 6 cells changed recently
        # (a living loop's sheath is static, but its core signals keep moving)
        act = (n - last_change) <= DEAD_AFTER
        for axis in (0, 1):
            acc = act.copy()
            for d in range(1, 7):
                acc |= np.roll(act, d, axis) | np.roll(act, -d, axis)
            act = acc
        dead = ~act
        rgb[dead] = (rgb[dead].astype(np.float32) * 0.33).astype(np.uint8)
        filled = grid > 0
        empty_k = 1 - smoothstep(i / 18)       # the close-up's grey empty cells fade away
        if empty_k > 0:
            rgb[~filled] = scale_rgb((30, 32, 44), empty_k)
            filled = np.ones_like(filled)
        cv = Canvas(rows, cols)
        m = smoothstep(i / 36)
        dst = (dst0[0] * (1 - m) + (PH / 2 + 1) * m, dst0[1] * (1 - m) + (PW / 2) * m)
        cv.zoom(rgb, filled, scale, (cy, cx), dst, gap=True)
        f = cv.to_frame()
        f.put_text(1, 2, "Langton's loops", _rgb(ACCENT))
        f.put_text(1, 19, f"step {n:,}", TEXT)
        f.put_text(2, 2, "亮 = 仍在活动 active · 暗 = 已停止 dead", DIM)
        if i == 0:
            f.sfx.append(("rise", 6.0))
        if i == 102:
            f.sfx.append(("chime", 0.4))
        COL_CAPS.apply(f, i)
        yield f


# --- Evoloop --------------------------------------------------------------------------

EVO_H, EVO_W = 152, 320         # torus, like Sayama's; 2 x 2 cells per px when zoomed out
EVO_FRAMES = 276
EVO_CAPS = Captions([
    (0, "1999 年，Sayama 给 Langton 环加了第 9 种状态「溶解」（橙色）：Evoloop",
        "In 1999 Sayama added a 9th, 'dissolving' state (orange) to Langton's loop: the Evoloop"),
    (60, "它同样会自我复制；但环与环相撞时会互相破坏，基因也可能被改写",
         "It copies itself too; but when loops collide they damage each other, and genes get rewritten"),
    (130, "无法再复制的残骸被溶解、腾出空间；复制得快的环占满世界",
          "Debris that can no longer reproduce dissolves and frees space; faster breeders take over"),
    (200, "没有人编程「进化」：变异 + 选择从局部规则中自己出现；久而久之，环变得更小、更快",
          "Nobody programmed evolution: mutation + selection emerge; over time loops get smaller, faster"),
])


def _evo_targets() -> np.ndarray:
    slow = np.arange(0, 60) * 2                       # watch the first copy being made
    u = np.linspace(0, 1, EVO_FRAMES - 60)
    fast = 120 + (16000 - 120) * u ** 1.6
    return np.concatenate([slow, fast]).astype(int)


def _evoloop_frames(rows: int, cols: int) -> Iterator[Frame]:
    g = np.zeros((EVO_H, EVO_W), dtype=np.uint8)
    L = evoloop.initial_loop()
    r0, c0 = EVO_H // 2 - 8, EVO_W // 2 - 8
    g[r0: r0 + 17, c0: c0 + 17] = L
    n = 0
    y_top = 4
    for i, tgt in enumerate(_evo_targets()):
        while n < tgt:
            g = evoloop.step(g)
            n += 1
        z = smoothstep(clamp01((i - 50) / 90))
        scale = geom(4.0, 0.5, z)
        cy = EVO_H / 2 * (1 - z) + EVO_H / 2 * z
        cv = Canvas(rows, cols)
        rgb = evoloop.PALETTE[g].copy()
        filled = g > 0
        if scale >= 2:
            rgb[~filled] = (22, 24, 34)
            filled = np.ones_like(filled)
        cv.zoom(rgb, filled, scale, (cy, EVO_W / 2), (y_top + (PH - y_top) / 2, PW / 2), gap=scale >= 3,
                region=(y_top, 0, PH - y_top, PW))
        f = cv.to_frame()
        f.put_text(0, 2, "Evoloop (Sayama 1999)", _rgb(ACCENT))
        f.put_text(0, 25, f"step {n:,}", TEXT)
        c = f.put_text(1, 2, "9 种状态 states:", DIM) + 1
        for k, name in ((1, "核 core"), (2, "鞘 sheath"), (4, "基因 genes"), (8, "溶解 dissolving")):
            c = f.put_text(1, c, "■", _rgb(evoloop.PALETTE[k]))
            c = f.put_text(1, c + 1, name, DIM) + 2
        if i == 60:
            f.sfx.append(("rise", 4.0))
        if i in (130, 200):
            f.sfx.append(("chime", 0.5))
        EVO_CAPS.apply(f, i)
        yield f


def build_scenes(rows: int, cols: int) -> List[Scene]:
    return [
        title_card(5, "兰顿自复制环", "Langton's loops", ACCENT, music="wonder"),
        Scene("ch5-history", lambda: _history_frames(rows, cols), fps=FPS,
              transition="dissolve", transition_ticks=16, music="calm"),
        Scene("ch5-closeup", lambda: _closeup_frames(rows, cols), fps=FPS,
              transition="iris", transition_ticks=18, music="curious"),
        Scene("ch5-colony", lambda: _colony_frames(rows, cols), fps=FPS,
              transition="cut", music="epic"),
        Scene("ch5-evoloop", lambda: _evoloop_frames(rows, cols), fps=FPS,
              transition="iris", transition_ticks=16, music="wonder"),
    ]
