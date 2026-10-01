"""Chapter 2: Wolfram elementary (1D, 2-state, 3-neighbour) cellular automata.

Beats: a row of cells and its 3-cell neighbourhood -> the 8 patterns of
Rule 22 with biological labels -> how the outputs spell the rule number ->
Rule 22 computed with a sliding window, accelerating, then zooming out to its
fractal -> Stephen Wolfram card -> Rule 30 (chaos; next to a Conus textile
shell) -> Rule 110 (interacting particles, Turing complete) -> Wolfram's four classes. Every rule first shows its full 8-entry
rule table. All cells are square pixels on the half-block canvas.
"""
from typing import Callable, Iterator, List, Optional, Tuple

import numpy as np

from chapters.common import portrait_scene, title_card
from engine.ascii_art import char_portrait
from engine.canvas import (AMBER, BG, BLUE, CYAN, DIM, FAINT, GREEN, ORANGE, PH, PINK, PW, RED, TEXT,
                           VIOLET, Canvas, Captions, clamp01, lerp_rgb, pixel_text, put_center,
                           scale_rgb, smoothstep)
from engine.frame import Frame, Scene
from sim.elementary_ca import generate, rule_table, step

CHAPTER_LABEL = ("第 2 章 · 一维元胞自动机", "Chapter 2 · 1D cellular automata")

ACCENT = ORANGE


def bits_of(idx: int) -> Tuple[int, int, int]:
    return (idx >> 2) & 1, (idx >> 1) & 1, idx & 1


# --- rule icons ------------------------------------------------------------------

def icon_geom(cell: int, spacing: int) -> Tuple[int, List[int]]:
    """x of each of the 8 icons (Wolfram order: 111 first), centred."""
    width = 7 * spacing + 3 * cell
    x0 = (PW - width) // 2
    return width, [x0 + k * spacing for k in range(8)]


def draw_icon(cv: Canvas, y: int, x: int, idx: int, out: Optional[int], cell: int, live, dead=FAINT,
              vgap: int = 2) -> None:
    for k, b in enumerate(bits_of(idx)):
        cv.rect(y, x + k * cell, cell - 1, cell - 1, live if b else dead)
    if out is not None:
        cv.rect(y + cell + vgap, x + cell, cell - 1, cell - 1, live if out else dead)


def draw_strip(cv: Canvas, rule: int, y: int, cell: int, spacing: int, live, n_out: int = 8,
               highlight: Optional[int] = None) -> List[int]:
    """Full 8-entry rule table; `n_out` outputs revealed; `highlight` = pattern idx."""
    table = rule_table(rule)
    _, xs = icon_geom(cell, spacing)
    for k, x in enumerate(xs):
        idx = 7 - k
        draw_icon(cv, y, x, idx, int(table[idx]) if k < n_out else None, cell, live)
        if highlight == idx:
            cv.outline(y - 2, x - 2, 2 * cell + 2 + 3, 3 * cell + 3, CYAN)
    return xs


# --- growth / zoom helper ------------------------------------------------------------

class Growth:
    """A single-seed (or given) 1D run, shown top-down with continuous zoom:
    scale = px per cell shrinks as generations accumulate."""

    def __init__(self, rule: int, gens: int, init: Optional[np.ndarray] = None,
                 colfn: Optional[Callable[[np.ndarray], np.ndarray]] = None):
        self.gens = gens
        self.cone = None
        if init is None:
            self.W = 2 * gens + 5
            self.data = generate(self.W, rule, gens, seed_col=self.W // 2)
            rr, cc = np.mgrid[: gens + 1, : self.W]
            self.cone = np.abs(cc - self.W // 2) <= rr + 1   # cells the seed can influence
        else:
            self.W = len(init)
            rows = [init.astype(np.uint8)]
            t = rule_table(rule)
            for _ in range(gens):
                rows.append(step(rows[-1], t))
            self.data = np.array(rows)
        g = np.arange(gens + 1)[:, None] / max(gens, 1)
        self.live_rgb = colfn(np.broadcast_to(g, self.data.shape)) if colfn else \
            np.broadcast_to(np.array(AMBER, np.uint8), self.data.shape + (3,))

    def draw(self, cv: Canvas, n_rows: float, scale: float, y_top: int, partial: Optional[Tuple[int, int]] = None,
             region: Optional[Tuple[int, int, int, int]] = None, center_col: Optional[float] = None,
             dead_rgb=None, top: float = 0.0) -> None:
        """Rows [0, n_rows) are known; `partial` = (row, last_col) of a row being
        written cell by cell. Generation `top` is drawn at pixel row y_top
        (camera panned down)."""
        known = np.zeros(self.data.shape, dtype=bool)
        known[: int(n_rows)] = True
        if partial is not None:
            r, c = partial
            known[r, : c + 1] = True
            known[r, : self.W // 2 - 12] = False  # (off screen anyway)
        if self.cone is not None:
            known &= self.cone
        if dead_rgb is None:
            dead_rgb = lerp_rgb(BG, FAINT, clamp01((scale - 1.5) / 3.5))
        rgb = np.where(self.data[..., None] == 1, self.live_rgb, np.array(dead_rgb, np.uint8)).astype(np.uint8)
        cc = self.W / 2 if center_col is None else center_col
        reg = region if region is not None else (y_top, 0, PH - y_top, PW)
        cv.zoom(rgb, known, scale, src_center=(top, cc), dst_center=(y_top, reg[1] + reg[3] / 2), region=reg)


def grad(a, b):
    return lambda g: lerp_rgb(a, b, g)


# --- scene A: a row of cells -------------------------------------------------------------

def _row_frames(rows: int, cols: int) -> Iterator[Frame]:
    n, cell = 15, 10
    x0, y0 = (PW - n * cell) // 2, 28
    rng = np.random.default_rng(22)
    state = (rng.random(n) < 0.45).astype(int)
    state[6:9] = (0, 1, 1)
    cap = Captions([
        (0, "一维元胞自动机：一排格子，每格只有「活」或「死」两种状态",
         "A 1D cellular automaton: a row of cells, each either 'alive' or 'dead'"),
        (42, "下一代时，每个格子只看三格：左邻、自己、右邻",
         "For the next generation, each cell looks at just three cells: left, itself, right"),
        (96, "每个格子都用同一条规则，所有格子同时更新；下一代的状态取决于具体的规则",
         "Every cell uses the same rule, all at the same time; what the next generation is depends on the rule"),
        (138, "三格各有 2 种状态：2 × 2 × 2 = 8 种可能的组合",
         "Each of the three has 2 states: 2 x 2 x 2 = 8 possible patterns"),
    ])
    for i in range(186):
        cv = Canvas(rows, cols)
        shown = np.arange(n) * 2 <= i
        for k in range(n):
            if shown[k]:
                cv.rect(y0, x0 + k * cell, cell - 1, cell - 1, AMBER if state[k] else FAINT)
        win = None
        if 42 <= i < 96:
            win = 7
        elif 96 <= i < 138:
            win = 1 + min(12, (i - 96) // 3)
        elif i >= 138:
            win = 7
        cy = y0 + cell + 10
        q = pixel_text("?")
        if i >= 96:
            # the whole next generation: every cell still undetermined
            for k in range(n):
                cv.rect(cy, x0 + k * cell, cell - 1, cell - 1, (30, 32, 46))
                if k != win:
                    cv.blit(q, DIM, cy + 1, x0 + k * cell + 2)
        if win is not None:
            cv.outline(y0 - 2, x0 + (win - 1) * cell - 2, cell + 3, 3 * cell + 3, CYAN)
            # the child cell below, with a connector, and a big '?' inside it
            cv.rect(y0 + cell + 2, x0 + win * cell + cell // 2 - 1, 7, 1, scale_rgb(CYAN, 0.6))
            cv.outline(cy - 1, x0 + win * cell - 1, cell + 1, cell + 1, CYAN)
            cv.blit(q, CYAN, cy + 1, x0 + win * cell + 2)
        if i >= 138:
            eq = pixel_text("2×2×2=8", 2, spacing=2)
            cv.blit(eq, AMBER, 64, (PW - eq.shape[1]) // 2)
        f = cv.to_frame(*cap.at(i))
        lrow = y0 // 2 - 2
        if i >= 30:
            ka = int(np.argmax(state[:5]))  # first alive cell among the first five
            f.put_text(lrow, x0 + ka * cell, "活 alive", AMBER)
            kd = 11 + int(np.argmin(state[11:]))
            f.put_text(lrow, x0 + kd * cell, "死 dead", DIM)
        if win is not None:
            brow = (y0 + cell + 20) // 2
            if i < 96 or i >= 138:
                for k, lab in enumerate(("左 left", "自己 self", "右 right")):
                    c = x0 + (win - 1 + k) * cell + cell // 2
                    put_center(f, brow + (k % 2), lab, CYAN if k != 1 else TEXT, col_center=c)
            if 96 <= i < 138 and (i - 96) % 3 == 0:
                f.sfx.append(("tick", 0.4))
        if i == 138:
            f.sfx.append(("chime", 0.6))
        yield f


# --- scene B: the 8 rules of Rule 22 ------------------------------------------------------

LABELS_22 = {  # idx -> (zh, en)
    7: ("拥挤", "crowded"), 6: ("拥挤", "crowded"), 5: ("拥挤", "crowded"),
    4: ("繁衍", "born of left"), 3: ("拥挤", "crowded"), 2: ("独自存活", "lone survivor"),
    1: ("繁衍", "born of right"), 0: ("无亲代", "no parent"),
}


def _rules22_frames(rows: int, cols: int) -> Iterator[Frame]:
    rule, cell, spacing = 22, 5, 20
    table = rule_table(rule)
    _, xs = icon_geom(cell, spacing)
    y = 17
    t_rev, per = 36, 16
    t_bin = t_rev + 8 * per            # 164
    t_sum = t_bin + 60                 # 224
    t_256 = t_sum + 60                 # 284
    total = t_256 + 66
    cap = Captions([
        (0, "三格共有 8 种组合；规则为每一种指定下一代的状态",
         "There are 8 patterns; a rule assigns each one a state for the next generation"),
        (t_rev, "Rule 22：三格中恰好有一个活细胞，下一代才是活的",
         "Rule 22: the new cell is alive only if exactly one of the three was alive"),
        (t_rev + 4 * per, "像生物一样：能独自存活、能繁衍到邻格；太拥挤或无亲代就死亡",
         "Like life: survive alone, reproduce into a neighbour; die if crowded or with no parent"),
        (t_bin, "把 8 个结果从左到右读成一个二进制数：00010110",
         "Read the 8 results from left to right as a binary number: 00010110"),
        (t_sum, "16 + 4 + 2 = 22 —— 这就是「Rule 22」这个名字的由来",
         "16 + 4 + 2 = 22 - that is where the name 'Rule 22' comes from"),
        (t_256, "每个结果可取 0 或 1，所以一共有 2 的 8 次方 = 256 条规则",
         "Each result can be 0 or 1, so there are 2^8 = 256 such rules in total"),
    ])
    for i in range(total):
        cv = Canvas(rows, cols)
        n_out = 0 if i < t_rev else min(8, (i - t_rev) // per + 1)
        appear = min(8, i // 3 + 1)
        for k in range(appear):
            idx = 7 - k
            draw_icon(cv, y, xs[k], idx, int(table[idx]) if k < n_out else None, cell, AMBER)
        if t_rev <= i < t_bin:
            k = (i - t_rev) // per
            cv.outline(y - 2, xs[k] - 2, 2 * cell + 2 + 3, 3 * cell + 3, CYAN)
        title = "RULE 22" if i >= t_sum else "RULE ?"
        cv.text_px(title, 2, PW // 2, TEXT if i >= t_sum else DIM, align="center")
        if i >= t_bin:
            for k in range(8):
                b = int(table[7 - k])
                cv.text_px(str(b), 34, xs[k] + cell + cell // 2 + 1, AMBER if b else DIM, align="center")
        if i >= t_sum:
            k = smoothstep((i - t_sum) / 20)
            eq = pixel_text("16+4+2=22", 2, spacing=2)
            cv.blit(eq, scale_rgb(AMBER, k), 56, (PW - eq.shape[1]) // 2)
        f = cv.to_frame(*cap.at(i))
        if i >= t_bin:
            for k in range(8):
                v = 2 ** (7 - k)
                on = int(table[7 - k]) == 1
                put_center(f, (y - 5) // 2, str(v), AMBER if (on and i >= t_bin + 24) else DIM,
                           col_center=xs[k] + cell + cell // 2)
        for k in range(n_out):
            idx = 7 - k
            zh, en = LABELS_22[idx]
            col = GREEN if table[idx] else RED
            cc = xs[k] + cell + cell // 2
            put_center(f, 23, zh + (" → 活" if table[idx] else " → 死"), col, col_center=cc)
            put_center(f, 24, en, DIM, col_center=cc)
        if i >= t_256:
            put_center(f, 38, "每个结果 0 或 1 → 2×2×2×2×2×2×2×2 = 2^8 = 256 种规则   (Wolfram code)", TEXT)
        if t_rev <= i < t_bin and (i - t_rev) % per == 0:
            idx = 7 - (i - t_rev) // per
            f.sfx.append(("pop", 0.7) if table[idx] else ("tick", 0.5))
        if i in (t_bin, t_sum):
            f.sfx.append(("chime", 0.5 if i == t_bin else 0.8))
        if i == t_256:
            f.sfx.append(("hit", 0.6))
        yield f


# --- scene C: Rule 22 computed step by step, then zoom out --------------------------------

def _evolve22_frames(rows: int, cols: int) -> Iterator[Frame]:
    G = 131
    gr = Growth(22, G, colfn=grad(AMBER, PINK))
    table = rule_table(22)
    seed = gr.W // 2
    y_top = 16
    big = 8
    cc = seed + 0.5
    # schedule of (n_rows_complete, partial (row, col) or None, window_col or None)
    sched: List[Tuple[int, Optional[Tuple[int, int]], Optional[int]]] = []
    for _ in range(24):
        sched.append((1, None, None))
    for r, fpp in zip((1, 2, 3, 4), (8, 4, 2, 1)):
        for c in range(seed - 9, seed + 10):
            for _ in range(fpp):
                sched.append((r, (r, c), c))
    for r in range(5, G + 1):
        n = 6 if r < 12 else (3 if r < 31 else 1)
        for _ in range(n):
            sched.append((r + 1, None, None))
    n_slide_end = next(i for i, s in enumerate(sched) if s[0] >= 6)
    hold = 72
    cap = Captions([
        (0, "从一个活细胞开始，用 Rule 22 一格一格地算出下一代",
         "Start from one live cell and compute the next generation with Rule 22, cell by cell"),
        (24 + 19 * 8, "三格窗口滑过上一代，查表，写下新格子",
         "A 3-cell window slides along the previous row, looks up the rule, writes the new cell"),
        (n_slide_end, "越算越快……每一代都只用同一张 8 格规则表",
         "Faster and faster... every generation uses the same 8-entry rule table"),
        (len(sched) - 60, "拉远看：简单的局部规则，长出了自相似的分形——谢尔宾斯基三角形",
         "Zoom out: a simple local rule has grown a self-similar fractal - the Sierpinski triangle"),
    ])
    for i in range(len(sched) + hold):
        n_rows, partial, win = sched[min(i, len(sched) - 1)]
        cur_rows = n_rows + (1 if partial else 0)
        scale = min(big, (PH - y_top - 1) / max(cur_rows, 1))
        scale = max(scale, 0.5)
        cv = Canvas(rows, cols)
        pat = None
        if win is not None:
            r, c = partial
            pat = int(gr.data[r - 1, c - 1] * 4 + gr.data[r - 1, c] * 2 + gr.data[r - 1, c + 1])
        draw_strip(cv, 22, 3, 3, 12, AMBER, highlight=pat)
        gr.draw(cv, n_rows, scale, y_top, partial=partial, center_col=cc)
        if win is not None:
            r, c = partial
            px = lambda col: int(round(PW / 2 + (col - cc) * scale))
            py = lambda row: int(round(y_top + row * scale))
            cv.outline(py(r - 1) - 1, px(c - 1) - 1, big + 1, 3 * big + 1, CYAN)
            cv.outline(py(r) - 1, px(c) - 1, big + 1, big + 1, AMBER)
        f = cv.to_frame(*cap.at(i))
        f.put_text(1, 2, "Rule 22", TEXT)
        f.put_text(1, PW - 24, f"generation {min(cur_rows - 1, G):>4d}", DIM)
        if win is not None:
            r, c = partial
            l, m, rr = (int(v) for v in gr.data[r - 1, c - 1: c + 2])
            out = int(table[pat])
            put_center(f, 6, f"左 {l} · 自己 {m} · 右 {rr}  →  {out}", CYAN)
            prev = sched[i - 1] if i > 0 else None
            if prev is None or prev[1] != partial:
                if r == 1:
                    f.sfx.append(("pop", 0.6) if out else ("tick", 0.3))
                elif out and r <= 2:
                    f.sfx.append(("pop", 0.4))
        if i == len(sched) - 60:
            f.sfx.append(("rise", 2.0))
        if i == len(sched):
            f.sfx.append(("hit", 0.7))
        yield f


# --- shared strip intro ----------------------------------------------------------------------

def _strip_intro(cv: Canvas, rule: int, i: int, live) -> List[int]:
    """Large 8-rule strip centred on the screen, outputs revealed quickly."""
    return draw_strip(cv, rule, 22, 6, 20, live, n_out=min(8, i // 3 + 1))


def _strip_intro_labels(f: Frame, rule: int, i: int, xs: List[int], live) -> None:
    table = rule_table(rule)
    cell = 6
    for k in range(8):
        cc = xs[k] + cell + cell // 2
        put_center(f, 9, "".join(map(str, bits_of(7 - k))), DIM, col_center=cc)
        if i >= k * 3:
            put_center(f, 22, str(int(table[7 - k])), tuple(int(v) for v in live) if table[7 - k] else DIM,
                       col_center=cc)
    binary = "".join(str(int(table[7 - k])) for k in range(8))
    if i >= 24:
        put_center(f, 26, f"二进制 binary {binary} = {rule}", TEXT)


# --- Rule 30 ------------------------------------------------------------------------------

def _rule30_frames(rows: int, cols: int) -> Iterator[Frame]:
    live = ORANGE
    n_strip = 60
    G = 190
    gr = Growth(30, G, colfn=grad((255, 200, 90), (255, 70, 60)))
    y_top = 16
    grow_sched = []
    for r in range(1, G + 1):
        grow_sched += [r + 1] * (3 if r < 16 else (2 if r < 36 else 1))
    n_grow = len(grow_sched) + 36
    n_shell = 156
    cap = Captions([
        (0, "Rule 30：同样是 8 格规则表，二进制 00011110 = 30",
         "Rule 30: the same kind of 8-entry table; binary 00011110 = 30"),
        (n_strip, "还是从一个活细胞出发，这次却长出了毫无规律的混沌",
         "Again one live cell to start with - but this time it grows into chaos"),
        (n_strip + 150, "它的中间一列如此随机，Mathematica 曾用它来生成随机数",
         "Its centre column is so random that Mathematica used it to generate random numbers"),
        (n_strip + n_grow, "海螺 Conus textile 的花纹，和 Rule 30 惊人地相似",
         "The shell of the textile cone snail looks strikingly similar to Rule 30"),
        (n_strip + n_grow + 84, "贝壳边缘的色素细胞彼此激活、抑制，像一排一维元胞，边长边「计算」花纹",
         "Pigment cells along the growing lip act like a 1D automaton, 'computing' the pattern row by row"),
    ])
    # shell comparison: Rule 30 from a random row in shell colours, scrolling down
    rng = np.random.default_rng(30)
    W2 = 58
    shell_ca = Growth(30, 400, init=(rng.random(W2) < 0.5).astype(np.uint8))
    # the shell as braille line-art (cut out of its photo), in its own colours
    sh_chars, sh_cols, _ = char_portrait("conus", 24, floor=0.0, color=1.0, threshold=True)
    cream, brown = np.array((235, 215, 175), np.uint8), np.array((105, 45, 20), np.uint8)
    shell_rgb = np.where(shell_ca.data[..., None] == 1, brown, cream).astype(np.uint8)
    for i in range(n_strip + n_grow + n_shell):
        cv = Canvas(rows, cols)
        if i < n_strip:
            xs = _strip_intro(cv, 30, i, live)
            cv.text_px("RULE 30", 4, PW // 2, TEXT, align="center")
            f = cv.to_frame(*cap.at(i))
            _strip_intro_labels(f, 30, i, xs, live)
            if i % 3 == 0 and i < 24:
                f.sfx.append(("blip", 0.3 + 0.08 * (i // 3)))
        elif i < n_strip + n_grow:
            j = i - n_strip
            nr = grow_sched[min(j, len(grow_sched) - 1)]
            # zoom out until one cell = one pixel (the finest the screen can
            # show), then keep that resolution and let the camera follow down
            scale = max(1.0, min(4.0, (PH - y_top - 1) / nr))
            top = max(0.0, nr - (PH - y_top - 1) / scale)
            draw_strip(cv, 30, 3, 3, 12, live)
            gr.draw(cv, nr, scale, y_top, top=top)
            f = cv.to_frame(*cap.at(i))
            f.put_text(1, 2, "Rule 30", TEXT)
            f.put_text(1, PW - 24, f"generation {nr - 1:>4d}", DIM)
            if j == len(grow_sched):
                f.sfx.append(("hit", 0.6))
        else:
            j = i - n_strip - n_grow
            n_r = min(400, 1 + j * 2)
            vis = shell_rgb[max(0, n_r - 72): n_r]   # top-aligned; scrolls once full
            cv.blit(np.ones(vis.shape[:2], bool), vis, 4, 2)
            f = cv.to_frame(*cap.at(i))
            sh, sw = sh_chars.shape
            sr, sc = 6, 2 + W2 + 2 + (PW - (2 + W2 + 2) - sw) // 2
            k = smoothstep(j / 20)
            cols_k = (sh_cols.astype(np.float32) * k).astype(np.uint8)
            rr, cc = np.nonzero(sh_chars != " ")
            f.chars[sr + rr, sc + cc] = sh_chars[rr, cc]
            f.colors[sr + rr, sc + cc] = cols_k[rr, cc]
            f.active[sr + rr, sc + cc] = True
            put_center(f, 39, "Rule 30（随机初始 random start）", DIM, col_center=2 + W2 // 2)
            put_center(f, sr + sh + 2, "Conus textile 织锦芋螺", TEXT, col_center=sc + sw // 2)
            put_center(f, sr + sh + 4, "photo: Richard Ling, CC BY-SA 3.0", FAINT, col_center=sc + sw // 2)
            if j == 0:
                f.sfx.append(("chime", 0.7))
        yield f


# --- Rule 110 -----------------------------------------------------------------------------

def _rule110_frames(rows: int, cols: int) -> Iterator[Frame]:
    live = BLUE
    n_strip = 54
    rng = np.random.default_rng(110)
    W = PW          # one cell per pixel across the whole screen
    G = 420
    gr = Growth(110, G, init=(rng.random(W) < 0.5).astype(np.uint8),
                colfn=grad((140, 180, 255), (120, 245, 255)))
    y_top = 16
    n_run = 250
    # The regular background ('ether') repeats every 7 generations; cells that
    # break that repetition belong to moving structures and are drawn bright.
    d = gr.data
    odd = np.zeros(d.shape, bool)
    odd[7:] = d[7:] != d[:-7]
    odd = odd | np.roll(odd, 1, axis=1) | np.roll(odd, -1, axis=1)
    ether = np.array((50, 70, 140), np.uint8)
    rgb110 = np.where(d[..., None] == 1, np.where(odd[..., None], gr.live_rgb, ether),
                      np.array((16, 18, 30), np.uint8)).astype(np.uint8)
    rgb110[odd & (d == 0)] = (30, 40, 70)
    cap = Captions([
        (0, "Rule 110：二进制 01101110 = 110",
         "Rule 110: binary 01101110 = 110"),
        (n_strip, "从随机的一行出发：在规则的背景上，一些局部结构（「粒子」，亮色）移动、碰撞",
         "From a random row: over a regular background, localized structures (bright) move and collide"),
        (n_strip + 110, "Wolfram 猜想它能完成任何计算；2004 年 Matthew Cook 证明 Rule 110 图灵完备",
         "Wolfram conjectured it can compute anything; Matthew Cook proved it Turing-complete (2004)"),
    ])
    for i in range(n_strip + n_run):
        cv = Canvas(rows, cols)
        if i < n_strip:
            xs = _strip_intro(cv, 110, i, live)
            cv.text_px("RULE 110", 4, PW // 2, TEXT, align="center")
            f = cv.to_frame(*cap.at(i))
            _strip_intro_labels(f, 110, i, xs, live)
        else:
            j = i - n_strip
            gen = min(G, int(j * 1.6))
            scale = 1.0                        # always the finest resolution, no zoom
            vis_rows = (PH - y_top) / scale
            top = max(0.0, gen + 1 - vis_rows)  # grows from the top, then scrolls
            known = np.zeros(d.shape, bool)
            known[: gen + 1] = True
            draw_strip(cv, 110, 3, 3, 12, live)
            cv.zoom(rgb110, known, scale, src_center=(top, W / 2), dst_center=(y_top, PW / 2),
                    region=(y_top, 0, PH - y_top, PW))
            f = cv.to_frame(*cap.at(i))
            f.put_text(1, 2, "Rule 110", TEXT)
            f.put_text(1, PW - 24, f"generation {gen:>4d}", DIM)
            if j == 0:
                f.sfx.append(("rise", 1.5))
        yield f


# --- Wolfram's four classes -------------------------------------------------------------------

CLASSES = [  # rule, zh, en, colour
    (160, "1 类 均匀", "class 1 uniform", VIOLET),
    (108, "2 类 周期", "class 2 periodic", GREEN),
    (30, "3 类 混沌", "class 3 chaotic", ORANGE),
    (110, "4 类 复杂", "class 4 complex", BLUE),
]


def _classes_frames(rows: int, cols: int) -> Iterator[Frame]:
    pw, gap = 34, 5
    x0 = (PW - (4 * pw + 3 * gap)) // 2
    y0, H = 8, 66
    rng = np.random.default_rng(4)
    runs = [Growth(rule, H, init=(rng.random(pw) < 0.5).astype(np.uint8)) for rule, *_ in CLASSES]
    cap = Captions([
        (0, "Wolfram 把 256 条规则的行为分成四类：均匀、周期、混沌、复杂",
         "Wolfram sorted the behaviour of all 256 rules into four classes"),
        (66, "最有趣的第 4 类，恰好处在秩序与混沌的边缘",
         "The most interesting, class 4, sits right at the edge between order and chaos"),
    ])
    dead = np.array((24, 26, 40), np.uint8)
    for i in range(132):
        cv = Canvas(rows, cols)
        nr = min(H, 2 + i)
        for k, ((rule, zh, en, col), g) in enumerate(zip(CLASSES, runs)):
            x = x0 + k * (pw + gap)
            data = g.data[:nr]
            rgb = np.where(data[..., None] == 1, np.array(col, np.uint8), dead).astype(np.uint8)
            cv.blit(np.ones(data.shape, bool), rgb, y0 + 4, x)
        if i >= 66:
            k = smoothstep((i - 66) / 12)
            cv.outline(y0 + 2, x0 + 3 * (pw + gap) - 2, H + 4, pw + 4, scale_rgb(TEXT, k))
        f = cv.to_frame(*cap.at(i))
        for k, (rule, zh, en, col) in enumerate(CLASSES):
            cc = x0 + k * (pw + gap) + pw // 2
            put_center(f, 1, zh, col, col_center=cc)
            put_center(f, 2, en, DIM, col_center=cc)
            put_center(f, 39, f"Rule {rule}", DIM, col_center=cc)
        if i == 0:
            f.sfx.append(("chime", 0.5))
        yield f


def build_scenes(rows: int, cols: int) -> List[Scene]:
    return [
        title_card(2, "一维元胞自动机", "1D cellular automata", ACCENT, music="curious"),
        Scene("ch2-row", lambda: _row_frames(rows, cols), fps=12, transition="wipe", music="curious"),
        Scene("ch2-rule22", lambda: _rules22_frames(rows, cols), fps=12, transition="dissolve",
              music="curious"),
        Scene("ch2-evolve22", lambda: _evolve22_frames(rows, cols), fps=24, transition="rain", music="build"),
        portrait_scene(
            "ch2-wolfram", music="calm", transition="dissolve",
            name="wolfram", zh_name="沃尔夫勒姆", en_name="Stephen Wolfram", years="1959 –",
            facts=[
                ("1980 年代系统研究了全部 256 条一维规则", "Studied all 256 elementary rules in the 1980s"),
                ("把它们的行为分为四类：均匀、周期、混沌、复杂", "Sorted them into four classes of behaviour"),
                ("著有《一种新科学》（2002）", "Author of A New Kind of Science (2002)"),
            ],
            captions=[
                (0, "1980 年代，物理学家 Stephen Wolfram 把这 256 条规则逐一跑了一遍",
                 "In the 1980s, physicist Stephen Wolfram ran all 256 rules, one by one"),
                (100, "他发现：极简单的规则里，藏着极复杂的行为",
                 "He found that extremely simple rules can hide extremely complex behaviour"),
            ],
            n_frames=204, credit="photo: S. Faust, CC BY-SA 3.0",
        ),
        Scene("ch2-rule30", lambda: _rule30_frames(rows, cols), fps=24, transition="rule30",
              transition_ticks=18, music="chaos"),
        Scene("ch2-rule110", lambda: _rule110_frames(rows, cols), fps=24, transition="iris", music="tense"),
        Scene("ch2-classes", lambda: _classes_frames(rows, cols), fps=12, transition="dissolve",
              music="wonder"),
    ]
