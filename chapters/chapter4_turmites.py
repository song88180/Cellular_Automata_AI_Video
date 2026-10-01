"""Chapter 4: Langton's ant and turmites.

Rule tables are taken from Greg Turk's reference implementation
(https://github.com/gturk1/turmites, turmites/key_commands.pde):
  - "RL"   Langton's original vant (key 1)
  - "RLLR" bilateral symmetry, boxy (key 3)
  - 7-state dragon-curve turmite (key 7)

Beats: title -> Langton portrait -> the ant's two rules, stepped through one
sub-step at a time on big square cells -> accelerate + zoom out through
~10,000 chaotic steps to the 'highway' -> RLLR (more colours) -> a 7-state
turmite (internal state = 2D Turing machine) with its rule table on screen.
"""

from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from chapters.common import portrait_scene, title_card
from engine.canvas import AMBER, CYAN, DIM, FAINT, PH, PW, TEXT, Canvas, Captions, box, clamp01, scale_rgb, smoothstep
from engine.frame import Frame, Scene
from sim.turmite import Ant, run, turmite, vant

CHAPTER_LABEL = ("第 4 章 · 兰顿蚂蚁", "Chapter 4 · Langton's ant")

FPS = 12
ANT_RGB = (255, 70, 70)
ACCENT = (255, 95, 95)

# colour index -> RGB (0 = empty; outro imports this)
PALETTE = np.array(
    [
        (0, 0, 0),
        (255, 200, 80),
        (80, 200, 255),
        (255, 90, 160),
        (120, 255, 120),
        (200, 120, 255),
        (255, 255, 255),
    ],
    dtype=np.uint8,
)

DRAGON_ROWS = [
    "Fg2 --- --- --- --- --- ---",
    "Rb7 --- --- --- --- --- ---",
    "Bb3 Re3 Lf3 Fg3 --- --- Fd4",
    "Ba3 Lb4 Rc4 Fd4 Bb5 Bc6 Bd7",
    "Bc4 Rb5 Lc5 Fd5 --- --- ---",
    "Bb4 Rb6 Lc6 Fd6 --- --- ---",
    "Bd4 Rb7 Lc7 Fd7 --- --- ---",
]
DRAGON = turmite(DRAGON_ROWS)

_ARROW = np.array([[c == "#" for c in row] for row in
                   ["..#..", "..#..", ".###.", ".###.", "#####"]])


def _rgb(c) -> Tuple[int, int, int]:
    return tuple(int(v) for v in c)


# --- camera -------------------------------------------------------------------

class Camera:
    """Smoothly follows the bounding box of the non-empty cells."""

    def __init__(self, scale: float, center: Tuple[float, float], max_scale: float, min_scale: float = 0.3,
                 margin: float = 1.25, speed: float = 0.25):
        self.scale, self.center = scale, center
        self.max_scale, self.min_scale, self.margin, self.speed = max_scale, min_scale, margin, speed

    def update(self, grid: np.ndarray, extra: Sequence[Tuple[int, int]], region_hw: Tuple[int, int]) -> None:
        ys, xs = np.nonzero(grid)
        pts_y = list(ys) + [p[0] for p in extra]
        pts_x = list(xs) + [p[1] for p in extra]
        y0, y1, x0, x1 = min(pts_y), max(pts_y) + 1, min(pts_x), max(pts_x) + 1
        bh, bw = max(6, (y1 - y0) * self.margin), max(6, (x1 - x0) * self.margin)
        target = float(np.clip(min(region_hw[0] / bh, region_hw[1] / bw), self.min_scale, self.max_scale))
        k = self.speed
        self.scale = float(np.exp(np.log(self.scale) * (1 - k) + np.log(target) * k))
        cy, cx = (y0 + y1) / 2, (x0 + x1) / 2
        self.center = (self.center[0] * (1 - k) + cy * k, self.center[1] * (1 - k) + cx * k)


def _draw_grid(cv: Canvas, grid: np.ndarray, palette: np.ndarray, cam: Camera,
               dst: Tuple[float, float], region: Tuple[int, int, int, int], empty_rgb=FAINT,
               tint: Optional[np.ndarray] = None, tint_rgb=CYAN) -> None:
    """Coloured cells; empty cells are drawn as faint squares only when zoomed in.
    `tint` (bool grid) recolours those non-empty cells with `tint_rgb`."""
    rgb = palette[grid % len(palette)]
    if tint is not None:
        rgb = rgb.copy()
        rgb[tint & (grid > 0)] = tint_rgb
    filled = grid > 0
    show_empty = clamp01((cam.scale - 2.5) / 2.5)
    if show_empty > 0:
        rgb = rgb.copy()
        rgb[~filled] = scale_rgb(empty_rgb, show_empty)
        filled = np.ones_like(filled)
    cv.zoom(rgb, filled, cam.scale, cam.center, dst, gap=True, region=region)


def _cell_px(cam: Camera, dst: Tuple[float, float], r: int, c: int) -> Tuple[float, float]:
    return (dst[0] + (r - cam.center[0]) * cam.scale, dst[1] + (c - cam.center[1]) * cam.scale)


def _draw_ant(cv: Canvas, cam: Camera, dst, ant_r: int, ant_c: int, heading: int) -> None:
    y, x = _cell_px(cam, dst, ant_r, ant_c)
    s = cam.scale
    if s >= 5:
        inner = int(s) - 1
        off = (inner - 5) // 2
        cv.blit(np.rot90(_ARROW, heading), ANT_RGB, int(np.floor(y)) + off, int(np.floor(x)) + off)
    else:
        n = max(1, int(round(s)))
        cv.rect(int(np.floor(y)), int(np.floor(x)), n, n, ANT_RGB)


# --- scene: the ant's rules, step by step ---------------------------------------

RULE_CARDS = [
    # (colour under the ant, title zh, title en, lines (zh, en))
    (0, "规则 A · 脚下是颜色 0（暗）", "on colour 0 (dark)",
     [("1. 右转 90°", "turn right 90°"), ("2. 把这格变成 1", "set the cell to 1"), ("3. 前进一格", "step forward")]),
    (1, "规则 B · 脚下是颜色 1（亮）", "on colour 1 (lit)",
     [("1. 左转 90°", "turn left 90°"), ("2. 把这格变回 0", "set the cell to 0"), ("3. 前进一格", "step forward")]),
]
CARD_COL, CARD_W = 111, 47
CARD_ROWS = (7, 20)
RL_COLORS = np.array([(0, 0, 0), AMBER], dtype=np.uint8)


def _draw_cards(cv: Canvas, f_setter, active: Optional[int], line: Optional[int], fade: float = 1.0):
    """Returns a callback that overlays both rule cards on a Frame."""
    for k, (col, _, _, _) in enumerate(RULE_CARDS):
        r = CARD_ROWS[k]
        sw = RL_COLORS[col] if col else FAINT
        cv.rect(2 * (r + 1), CARD_COL + 2, 2, 4, scale_rgb(sw if col else (95, 100, 120), fade))

    def overlay(f: Frame) -> None:
        for k, (col, tzh, ten, lines) in enumerate(RULE_CARDS):
            r = CARD_ROWS[k]
            on = active == k
            border = scale_rgb(AMBER if on else FAINT, fade)
            box(f, r, CARD_COL, 11, CARD_W, border)
            f.put_text(r + 1, CARD_COL + 8, tzh, _rgb(scale_rgb(TEXT if on else DIM, fade)))
            f.put_text(r + 2, CARD_COL + 8, ten, _rgb(scale_rgb(DIM if on else FAINT, fade)))
            for j, (lz, le) in enumerate(lines):
                hl = on and line is not None and j == line
                done = on and line is not None and j < line
                c = AMBER if hl else (TEXT if (on and (line is None or done)) else (DIM if on else FAINT))
                f.put_text(r + 4 + 2 * j, CARD_COL + 3, ("▶ " if hl else "  ") + lz, _rgb(scale_rgb(c, fade)))
                f.put_text(r + 4 + 2 * j, CARD_COL + 22, le, _rgb(scale_rgb(DIM if on else FAINT, fade)))
    return overlay


def _ant_hud(f: Frame, step: int, rule_name: str, extra: str = "") -> None:
    f.put_text(1, 2, f"{rule_name}", _rgb(ACCENT))
    f.put_text(1, 2 + len(rule_name) + 2, f"step {step:,}", TEXT)
    if extra:
        f.put_text(2, 2, extra, DIM)


ANT_GRID = 300
ANT_START = (150, 150)
RULE_REGION = (0, 0, PH, 108)
N_SLOW, SUB_FRAMES = 5, 7
N_MEDIUM, MEDIUM_FRAMES = 12, 5
INTRO_FRAMES = 36

RULES_CAPS = Captions([
    (0, "Langton 蚂蚁：蚂蚁站在一个格子上，每一步只看脚下这一格",
        "Langton's ant: an ant stands on a grid; each step it looks only at the cell beneath it"),
    (INTRO_FRAMES, "脚下是颜色 0（暗）：右转 90°，把这格变成 1，再前进一格",
                   "On colour 0 (dark): turn right 90°, set the cell to 1, then step forward"),
    (INTRO_FRAMES + 4 * 3 * SUB_FRAMES, "脚下是颜色 1（亮）：左转 90°，把这格变回 0，再前进一格",
                                         "On colour 1 (lit): turn left 90°, set it back to 0, then step forward"),
    (INTRO_FRAMES + 5 * 3 * SUB_FRAMES + 24, "整条规则就这两句。名字「RL」= 颜色 0 上右转，颜色 1 上左转",
                                              "That is the entire rule. Its name 'RL' = Right on colour 0, Left on colour 1"),
])


def _rl_turn(color: int) -> int:
    return 3 if color == 0 else 1   # R = +3, L = +1 (heading 0 up, 1 left, 2 down, 3 right)


def _ant_rules_frames(rows: int, cols: int) -> Iterator[Frame]:
    grid = np.zeros((ANT_GRID, ANT_GRID), dtype=np.int64)
    r, c = ANT_START
    heading = 0
    cam = Camera(6.0, (r + 0.5, c + 0.5), 6.0)
    dst = (PH / 2, RULE_REGION[3] / 2)
    i = 0

    def frame(active, line, head, step, sfx=None):
        nonlocal i
        cv = Canvas(rows, cols)
        _draw_grid(cv, grid, RL_COLORS, cam, dst, RULE_REGION)
        _draw_ant(cv, cam, dst, r, c, head)
        ov = _draw_cards(cv, None, active, line)
        f = cv.to_frame()
        ov(f)
        _ant_hud(f, step, "RL")
        if sfx:
            f.sfx.append(sfx)
        RULES_CAPS.apply(f, i)
        i += 1
        return f

    for _ in range(INTRO_FRAMES):
        yield frame(None, None, heading, 0)
    step = 0
    for n in range(N_SLOW + N_MEDIUM):
        color = int(grid[r, c])
        card = color
        new_heading = (heading + _rl_turn(color)) & 3
        if n < N_SLOW:
            for sub in range(3):
                if sub == 1:
                    grid[r, c] = 1 - color
                for k in range(SUB_FRAMES):
                    if sub == 2 and k == 2:
                        r, c = r + (-1, 0, 1, 0)[new_heading], c + (0, -1, 0, 1)[new_heading]
                        step += 1
                    h = heading if sub == 0 and k < 2 else new_heading
                    yield frame(card, sub, h, step, ("tick", 0.4 + 0.2 * sub) if k == 0 else None)
            heading = new_heading
        else:
            grid[r, c] = 1 - color
            heading = new_heading
            r, c = r + (-1, 0, 1, 0)[heading], c + (0, -1, 0, 1)[heading]
            step += 1
            for k in range(MEDIUM_FRAMES):
                yield frame(card, None, heading, step, ("tick", 0.5) if k == 0 else None)
    for _ in range(12):
        yield frame(None, None, heading, step)


# --- scene: accelerate to the highway -------------------------------------------

N_ACCEL, N_HIGHWAY, N_HOLD = 96, 72, 24
HIGHWAY_AT = 10_000
RUN_CAPS = Captions([
    (0, "加速：一步接一步，最初的一万步看起来毫无规律……",
        "Speeding up: step after step, the first ~10,000 steps look like pure chaos..."),
    (N_ACCEL, "……然后它突然开始修建一条「高速公路」：104 步一个周期，永远延伸",
              "...then it suddenly starts building a 'highway': a 104-step cycle, repeated forever"),
    (N_ACCEL + 42, "至今没人能证明：任何初始图案最终都会出现高速公路——只能运行它看看",
                   "Nobody has proved that every starting pattern ends in a highway - you can only run it"),
])


def _ant_run_frames(rows: int, cols: int) -> Iterator[Frame]:
    grid = np.zeros((ANT_GRID, ANT_GRID), dtype=np.int64)
    ant = Ant(*ANT_START, heading=0)
    rule = vant("RL")
    start = N_SLOW + N_MEDIUM
    done = run(grid, ant, rule, start)
    targets = np.concatenate([
        np.geomspace(start + 1, HIGHWAY_AT, N_ACCEL),
        np.linspace(HIGHWAY_AT, 12_600, N_HIGHWAY),
        np.full(N_HOLD, 12_600),
    ]).astype(int)
    cam = Camera(6.0, (ANT_START[0] + 0.5, ANT_START[1] + 0.5), 6.0, min_scale=1.0, speed=0.12)
    full = (0, 0, PH, PW)
    hit_done = False
    snapshot = None
    for i, tgt in enumerate(targets):
        if tgt > done:
            if done < HIGHWAY_AT <= tgt:
                done += run(grid, ant, rule, HIGHWAY_AT - done)
                snapshot = grid.copy()
            done += run(grid, ant, rule, int(tgt) - done)
        cam.update(grid, [(ant.r, ant.c)], (PH - 6, PW - 8))
        mix = smoothstep(i / 30)
        dst = (PH / 2, RULE_REGION[3] / 2 * (1 - mix) + PW / 2 * mix)
        cv = Canvas(rows, cols)
        tint = (grid != snapshot) if snapshot is not None else None
        _draw_grid(cv, grid, RL_COLORS, cam, dst, full, tint=tint)
        _draw_ant(cv, cam, dst, ant.r, ant.c, ant.heading)
        fade = 1 - smoothstep(i / 12)
        ov = _draw_cards(cv, None, None, None, fade) if fade > 0.02 else None
        f = cv.to_frame()
        if ov:
            ov(f)
        extra = ""
        if done >= HIGHWAY_AT:
            extra = "▸ 青色 = 高速公路：104 步一周期，每周期斜移 2 格 / cyan = highway: 104-step cycle, 2 cells diagonally"
        _ant_hud(f, done, "RL", extra)
        if i == 0:
            f.sfx.append(("rise", N_ACCEL / FPS))
        if done >= HIGHWAY_AT and not hit_done:
            hit_done = True
            f.sfx.append(("hit", 1.0))
            f.colors[:] = np.clip(f.colors.astype(np.int16) + 90, 0, 255).astype(np.uint8)
        RUN_CAPS.apply(f, i)
        yield f


# --- scene: RLLR --------------------------------------------------------------------

RLLR_CAPS = Captions([
    (0, "推广：格子可以有更多颜色。蚂蚁按脚下颜色转向，再把这格颜色 +1",
        "Generalize: more colours. Turn by the colour underfoot, then bump that cell's colour by one"),
    (54, "规则名就是转向序列：RLLR = 在颜色 0/1/2/3 上 右/左/左/右 转",
         "The name is the list of turns: RLLR = right/left/left/right on colours 0/1/2/3"),
    (108, "同样简单的规则，却长出一个左右对称、不断扩大的图案",
          "An equally simple rule grows an ever-larger, bilaterally symmetric pattern"),
])
RLLR_FRAMES, RLLR_HOLD = 144, 18


def _legend_colors(cv: Canvas, f_rows: List, x: int, y_row: int, n: int, palette: np.ndarray):
    for k in range(n):
        rgb = palette[k] if k else (95, 100, 120)
        cv.rect(2 * y_row, x + 5 * k, 2, 4, rgb)


def _rllr_frames(rows: int, cols: int) -> Iterator[Frame]:
    rule = vant("RLLR")
    grid = np.zeros((ANT_GRID, ANT_GRID), dtype=np.int64)
    ant = Ant(ANT_START[0], ANT_START[1], heading=3)
    targets = (60_000 * (np.arange(1, RLLR_FRAMES + 1) / RLLR_FRAMES) ** 1.7).astype(int)
    targets = np.concatenate([targets, np.full(RLLR_HOLD, targets[-1])])
    cam = Camera(6.0, (ANT_START[0] + 0.5, ANT_START[1] + 0.5), 6.0, min_scale=0.8, speed=0.2)
    region = (0, 0, PH, PW)
    dst = (PH / 2 + 2, 64)
    lx, ly = 116, 3          # legend: char col / row
    done = 0
    for i, tgt in enumerate(targets):
        if tgt > done:
            done += run(grid, ant, rule, int(tgt) - done)
        cam.update(grid, [(ant.r, ant.c)], (PH - 8, 110))
        cv = Canvas(rows, cols)
        _draw_grid(cv, grid, PALETTE, cam, dst, region)
        _draw_ant(cv, cam, dst, ant.r, ant.c, ant.heading)
        _legend_colors(cv, [], lx + 14, ly + 2, 4, PALETTE)
        f = cv.to_frame()
        box(f, ly, lx - 2, 10, 43, FAINT, title="RLLR", title_rgb=ACCENT)
        f.put_text(ly + 2, lx, "颜色 colour", DIM)
        f.put_text(ly + 3, lx, "            0    1    2    3", DIM)
        f.put_text(ly + 4, lx, "转向 turn    R    L    L    R", TEXT)
        f.put_text(ly + 6, lx, "颜色循环 0→1→2→3→0", DIM)
        f.put_text(ly + 7, lx, "colours cycle 0→1→2→3→0", FAINT)
        # highlight the column for the colour under the ant
        cur = int(grid[ant.r, ant.c]) % 4
        f.put_text(ly + 4, lx + 13 + 5 * cur, "RLLR"[cur], _rgb(AMBER))
        _ant_hud(f, done, "RLLR")
        RLLR_CAPS.apply(f, i)
        if i == 108:
            f.sfx.append(("chime", 0.7))
        yield f


# --- scene: 7-state dragon turmite ---------------------------------------------------

DRAGON_CAPS = Captions([
    (0, "再进一步：让蚂蚁自己带有「内部状态」——这就是 turmite，一台二维图灵机",
        "One step further: give the ant an internal state - a turmite, i.e. a 2D Turing machine"),
    (54, "规则表：（当前状态, 脚下颜色）→（转向, 写入的颜色, 下一个状态）",
         "Rule table: (state, colour underfoot) -> (turn, colour to write, next state)"),
    (108, "这只 7 状态 turmite（Greg Turk 设计）画出了自相似的龙形曲线",
          "This 7-state turmite (designed by Greg Turk) draws a self-similar dragon curve"),
])
DRAGON_FRAMES, DRAGON_HOLD = 150, 18
TABLE_COL, TABLE_ROW = 106, 3


def _dragon_frames(rows: int, cols: int) -> Iterator[Frame]:
    grid = np.zeros((ANT_GRID, ANT_GRID), dtype=np.int64)
    ant = Ant(ANT_START[0], ANT_START[1], heading=3)
    targets = (2_000_000 * (np.arange(1, DRAGON_FRAMES + 1) / DRAGON_FRAMES) ** 2.2).astype(int)
    targets = np.maximum(targets, np.arange(1, DRAGON_FRAMES + 1) * 3)
    targets = np.concatenate([targets, np.full(DRAGON_HOLD, targets[-1])])
    cam = Camera(6.0, (ANT_START[0] + 0.5, ANT_START[1] + 0.5), 6.0, min_scale=0.8, speed=0.2)
    region = (0, 0, PH, 102)
    dst = (PH / 2 + 2, 51)
    letters = "abcdefg"
    done = 0
    for i, tgt in enumerate(targets):
        if tgt > done:
            done += run(grid, ant, DRAGON, int(tgt) - done)
        cam.update(grid, [(ant.r, ant.c)], (PH - 8, 96))
        cv = Canvas(rows, cols)
        _draw_grid(cv, grid, PALETTE, cam, dst, region)
        _draw_ant(cv, cam, dst, ant.r, ant.c, ant.heading)
        f = cv.to_frame()
        r0, c0 = TABLE_ROW, TABLE_COL
        box(f, r0, c0 - 2, 29, 53, FAINT, title="turmite 规则表 rule table", title_rgb=ACCENT)
        f.put_text(r0 + 2, c0, "状态＼颜色", DIM)
        for k, L in enumerate(letters):
            f.put_text(r0 + 2, c0 + 13 + 5 * k, L, _rgb(PALETTE[k] if k else DIM))
        cur_s, cur_c = ant.state, int(grid[ant.r, ant.c])
        for s, row in enumerate(DRAGON_ROWS):
            f.put_text(r0 + 4 + 2 * s, c0 + 3, str(s + 1), _rgb(ACCENT if s == cur_s else DIM))
            for k, word in enumerate(row.split()):
                hl = s == cur_s and k == cur_c
                rgb = AMBER if hl else (FAINT if word.startswith("-") else TEXT)
                f.put_text(r0 + 4 + 2 * s, c0 + 12 + 5 * k, word, _rgb(rgb))
        lr = r0 + 19
        f.put_text(lr, c0, "F 直行 fwd   L 左转 left", DIM)
        f.put_text(lr + 1, c0, "R 右转 right  B 掉头 U-turn", DIM)
        f.put_text(lr + 3, c0, "Fg2 = 直行，写入颜色 g，转到状态 2", TEXT)
        f.put_text(lr + 4, c0, "      go forward, write g, go to state 2", DIM)
        f.put_text(lr + 6, c0, "--- = 未定义 (never used)", FAINT)
        f.put_text(lr + 8, c0, "rules: Greg Turk, github.com/gturk1/turmites", FAINT)
        _ant_hud(f, done, "7-state turmite", f"state {cur_s + 1}")
        DRAGON_CAPS.apply(f, i)
        if i == 108:
            f.sfx.append(("chime", 0.8))
        yield f


# --- chapter ------------------------------------------------------------------------

def build_scenes(rows: int, cols: int) -> List[Scene]:
    return [
        title_card(4, "兰顿蚂蚁", "Langton's ant & turmites", ACCENT, music="curious"),
        portrait_scene(
            "ch4-langton", music="calm", transition="dissolve",
            name="langton", zh_name="克里斯托弗·兰顿", en_name="Christopher Langton", years="",
            facts=[
                ("「人工生命」领域的开创者之一", "A founder of the field of Artificial Life"),
                ("1987 年组织第一届人工生命会议", "Organised the first Artificial Life conference (1987)"),
                ("1986 年提出「Langton 蚂蚁」", "Introduced 'Langton's ant' (1986)"),
                ("1984 年设计了会自我复制的「Langton 环」", "Designed self-reproducing 'Langton's loops' (1984)"),
            ],
            captions=[
                (0, "克里斯托弗·兰顿：人工生命的开创者", "Christopher Langton: a pioneer of Artificial Life"),
                (84, "他追问：生命的特征能否在计算机里自己「涌现」出来？",
                     "His question: can the hallmarks of life emerge on their own inside a computer?"),
            ],
            n_frames=180, accent=(255, 150, 150), credit="photo: Cary Herz, CC BY-SA 3.0",
        ),
        Scene("ch4-ant-rules", lambda: _ant_rules_frames(rows, cols), fps=FPS,
              transition="wipe", transition_ticks=16, music="curious"),
        Scene("ch4-ant-run", lambda: _ant_run_frames(rows, cols), fps=FPS,
              transition="cut", music="chaos"),
        Scene("ch4-rllr", lambda: _rllr_frames(rows, cols), fps=FPS,
              transition="iris", transition_ticks=18, music="wonder"),
        Scene("ch4-dragon", lambda: _dragon_frames(rows, cols), fps=FPS,
              transition="rain", transition_ticks=18, music="build"),
    ]
