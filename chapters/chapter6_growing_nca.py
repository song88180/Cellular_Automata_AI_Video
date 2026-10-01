"""Chapter 6: Growing Neural Cellular Automata (Mordvintsev et al., Distill 2020).

Runs a *really trained* NCA (weights: sim/nca/weights/model.npz, produced by
sim/nca/train.py) with the pure-numpy runtime. Beats: title -> a seed grows a
lizard (hook) -> five diagram beats explaining how it works (state vector,
perception, per-cell network, stochastic update + alive mask, training) ->
growth with hidden channels visible -> eraser damage and regrowth -> cut in
half and regrowth. All diagrams show real values from the running model.
"""
from functools import lru_cache
from typing import Iterator, List, Tuple

import numpy as np
from PIL import Image

from chapters.common import title_card
from engine.canvas import (AMBER, BG, CYAN, DIM, FAINT, GREEN, PINK, TEXT, WHITE, Canvas, Captions,
                           box, colormap, ease_in_out, lerp_rgb, scale_rgb, smoothstep)
from engine.frame import Frame, Scene
from sim.nca.model import ALIVE_THRESHOLD, NCA, SOBEL_X, SOBEL_Y, alive_mask, perceive, seed_state, to_rgb

CHAPTER_LABEL = ("第 6 章 · 神经元胞自动机", "Chapter 6 · Neural cellular automata")

GRID = 72                              # training grid: 40x40 target + 16 padding
BG_ARR = np.array(BG, dtype=np.uint8)
ACCENT = (110, 230, 130)
TARGET_PNG = "sim/nca/weights/target_emoji_u1f98e.png"

# Real training curve: log10(loss) every 100 steps, copied from the training
# log of the shipped weights (build/out/nca_train.log, 8000 steps).
LOSS_LOG10 = (
    -1.517, -1.808, -1.915, -2.004, -1.977, -2.216, -2.093, -2.263, -2.225, -2.340, -2.502, -2.500,
    -2.587, -2.413, -2.601, -2.738, -2.604, -2.655, -3.044, -2.757, -2.793, -3.157, -3.225, -3.100,
    -3.231, -3.256, -3.466, -3.352, -3.290, -3.687, -3.628, -3.550, -3.428, -3.525, -3.564, -2.926,
    -3.909, -3.582, -3.617, -3.422, -3.655, -3.380, -4.030, -3.500, -3.345, -3.810, -3.983, -3.750,
    -3.477, -4.066, -3.599, -3.148, -3.885, -3.873, -4.031, -3.951, -3.666, -3.772, -3.536, -3.681,
    -3.975, -3.322, -3.419, -4.173, -3.840, -3.773, -3.622, -3.886, -3.943, -3.395, -4.116, -3.884,
    -3.668, -3.908, -4.116, -3.798, -4.170, -3.753, -3.566, -3.785, -3.895,
)

# Hidden channels shown in the side panels: (channel, colormap, vmin, vmax);
# ranges ~ 2nd..98th percentile of a grown lizard.
HIDDEN = [(5, "magma", -1.4, 1.7), (8, "ice", -0.5, 3.9), (9, "viridis", -0.1, 4.5), (14, "fire", -2.6, 0.3)]


# --- model helpers ---------------------------------------------------------------

@lru_cache(maxsize=1)
def _nca() -> NCA:
    return NCA.load()


@lru_cache(maxsize=4)
def _grown(steps: int = 240, seed: int = 2020) -> np.ndarray:
    nca, rng, x = _nca(), np.random.default_rng(seed), seed_state(GRID, GRID)
    for _ in range(steps):
        x = nca.step(x, rng)
    return x


def _probe_cell(x: np.ndarray) -> Tuple[int, int]:
    """A solidly alive body cell near the centre of the grid."""
    a = x[..., 3]
    ys, xs = np.nonzero(a > 0.9)
    d = (ys - 36) ** 2 + (xs - 36) ** 2
    k = int(np.argmin(d))
    return int(ys[k]), int(xs[k])


def _draw_lizard(cv: Canvas, x: np.ndarray, y0: int, x0: int) -> None:
    cv.blit(x[..., 3] > ALIVE_THRESHOLD, to_rgb(x, BG_ARR), y0, x0)


def _pool2(x: np.ndarray) -> np.ndarray:
    h, w = x.shape[:2]
    return x.reshape(h // 2, 2, w // 2, 2, -1).mean(axis=(1, 3))


def _draw_hidden(cv: Canvas, x: np.ndarray, y0: int, x0: int, ch: int, cmap: str, lo: float, hi: float) -> None:
    p = _pool2(x)
    cv.blit(p[..., 3] > 0.05, colormap(p[..., ch], cmap, lo, hi), y0, x0)


def _line(cv: Canvas, y0: float, x0: float, y1: float, x1: float, rgb, add: bool = False) -> None:
    n = int(max(abs(y1 - y0), abs(x1 - x0))) + 1
    ys = np.round(np.linspace(y0, y1, n)).astype(int)
    xs = np.round(np.linspace(x0, x1, n)).astype(int)
    ok = (ys >= 0) & (ys < cv.H) & (xs >= 0) & (xs < cv.W)
    ys, xs = ys[ok], xs[ok]
    col = np.asarray(rgb, dtype=np.uint8)
    if add:
        cur = np.where(cv.filled[ys, xs, None], cv.rgb[ys, xs], 0).astype(np.int16)
        col = np.clip(cur + col.astype(np.int16), 0, 255).astype(np.uint8)
    cv.filled[ys, xs] = True
    cv.rgb[ys, xs] = col


def _circle(cv: Canvas, cy: float, cx: float, r: float, rgb, fill_rgb=None) -> None:
    yy, xx = np.mgrid[:cv.H, :cv.W]
    d = np.hypot(yy + 0.5 - cy, xx + 0.5 - cx)
    if fill_rgb is not None:
        m = d < r - 0.6
        cv.rgb[m] = np.where(cv.filled[m, None], lerp_rgb(cv.rgb[m], fill_rgb, 0.35), scale_rgb(fill_rgb, 0.25))
        cv.filled[m] = True
    ring = np.abs(d - r) < 0.65
    cv.filled[ring] = True
    cv.rgb[ring] = rgb


def _bite(x: np.ndarray, cy: float, cx: float, r: float) -> np.ndarray:
    yy, xx = np.mgrid[: x.shape[0], : x.shape[1]]
    inside = (yy + 0.5 - cy) ** 2 + (xx + 0.5 - cx) ** 2 < r ** 2
    x = x.copy()
    x[inside] = 0.0
    return x


def _t(color) -> Tuple[int, int, int]:
    return tuple(int(v) for v in color)


def _caption(cap: Captions, f: Frame, i: int) -> Frame:
    return cap.apply(f, i)


# --- scene: hook -------------------------------------------------------------------

def _hook_frames(rows: int, cols: int) -> Iterator[Frame]:
    nca, rng, x = _nca(), np.random.default_rng(7), seed_state(GRID, GRID)
    cap = Captions([
        (0, "上一章的「环」靠人手写规则复制自己。换一种思路：", "The loop copied itself with hand-written rules. A different idea:"),
        (30, "让计算机自己「学」出规则——一个种子，长出一整只蜥蜴", "let a computer *learn* the rule - and one seed grows a whole lizard"),
    ])
    n = 78
    for i in range(n):
        if 8 <= i:
            for _ in range(2):
                x = nca.step(x, rng)
        cv = Canvas(rows, cols)
        # 1.5x: the lizard fills most of the picture height
        cv.zoom(to_rgb(x, BG_ARR), x[..., 3] > ALIVE_THRESHOLD, 1.5, (GRID / 2, GRID / 2), (40, 80), gap=False)
        if i < 8:  # the lone seed cell (its RGB is black) pulses white
            k = 0.5 + 0.5 * np.cos(i * 0.8)
            cv.rect(40, 80, 2, 2, lerp_rgb(ACCENT, WHITE, k))
        f = cv.to_frame()
        steps = max(0, (i - 7) * 2)
        f.put_text(38, 2, f"step {steps}", DIM)
        if i == 8:
            f.sfx.append(("chime", 0.7))
        yield _caption(cap, f, i)


# --- scene: (1) state vector ---------------------------------------------------------

STATE_LABELS = ["R", "G", "B", "α"] + [str(k) for k in range(1, 13)]


def _state_frames(rows: int, cols: int) -> Iterator[Frame]:
    nca, rng = _nca(), np.random.default_rng(11)
    x = _grown()
    py, px = _probe_cell(x)
    cap = Captions([
        (0, "① 状态：每个格子里存着 16 个数", "(1) State: every cell stores 16 numbers"),
        (48, "前 4 个是看得见的颜色 RGB 和「活性」α（α > 0.1 才算活着）",
             "The first 4 are visible: colour RGB and 'aliveness' α (alive if α > 0.1)"),
        (108, "另外 12 个是隐藏通道：细胞之间传递的「化学信号」",
              "The other 12 are hidden channels: 'chemical signals' passed between cells"),
    ])
    y0, x0 = 4, 6
    base_y, bx0, bw = 46, 88, 4
    n = 168
    for i in range(n):
        if i % 2 == 0:
            x = nca.step(x, rng)
        cv = Canvas(rows, cols)
        _draw_lizard(cv, x, y0, x0)
        # probe marker + leader line to the bar chart
        cy, cx = y0 + py, x0 + px
        cv.outline(cy - 2, cx - 2, 5, 5, WHITE)
        grow = smoothstep(i / 24)
        _line(cv, cy, cx + 3, cy, cx + 3 + (bx0 - 6 - cx - 3) * grow, scale_rgb(WHITE, 0.6))
        vals = x[py, px]
        for k in range(16):
            v = float(vals[k]) * grow
            if k < 3:
                col = [(255, 90, 90), (90, 235, 110), (90, 150, 255)][k]
            elif k == 3:
                col = WHITE
            else:
                col = AMBER if v >= 0 else CYAN
            show = (k < 4 and i >= 36) or (k >= 4 and i >= 96) or i < 36
            if not show:
                col = scale_rgb(col, 0.3)
            h = int(np.clip(round(abs(v) * (22 if k < 4 else 5)), 1, 24))
            xk = bx0 + k * bw
            if v >= 0:
                cv.rect(base_y - h, xk, h, bw - 1, col)
            else:
                cv.rect(base_y, xk, h, bw - 1, col)
        cv.rect(base_y, bx0 - 2, 1, 16 * bw + 3, scale_rgb(DIM, 0.6))
        f = cv.to_frame()
        f.put_text(2, 84, "每个格子 = 16 个数  ·  each cell = 16 numbers", TEXT)
        f.put_text(5, 84, "■", (255, 90, 90)); f.put_text(5, 85, "■", (90, 235, 110)); f.put_text(5, 86, "■", (90, 150, 255))
        f.put_text(5, 88, "RGB  颜色 visible colour", TEXT if 36 <= i else DIM)
        f.put_text(6, 84, "■", WHITE)
        f.put_text(6, 88, "α    活性 alive if α > 0.1", TEXT if 36 <= i else DIM)
        f.put_text(7, 84, "■", AMBER); f.put_text(7, 85, "■", CYAN)
        f.put_text(7, 88, "1–12 隐藏通道 hidden: ＋/－", TEXT if i >= 96 else DIM)
        for k, lab in enumerate(STATE_LABELS):
            f.put_text(34, bx0 + k * bw + (1 if len(lab) == 1 else 0), lab, _t(DIM))
        f.put_text(38, x0 + 1, f"格子 cell ({py}, {px})", DIM)
        if i in (36, 96):
            f.sfx.append(("blip", 0.5 if i == 36 else 0.7))
        yield _caption(cap, f, i)


# --- scene: (2) perception -----------------------------------------------------------

KERNELS = [
    ("恒等 identity", np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.float32), TEXT),
    ("Sobel x  ÷8", SOBEL_X * 8, AMBER),
    ("Sobel y  ÷8", SOBEL_Y * 8, CYAN),
]


def _perceive_frames(rows: int, cols: int) -> Iterator[Frame]:
    x = _grown()
    py, px = _probe_cell(x)
    per = perceive(x)[py, px]                     # 48 values
    cap = Captions([
        (0, "② 感知：每个格子只看自己和 8 个邻居（3×3）", "(2) Perception: each cell sees only itself and its 8 neighbours (3x3)"),
        (40, "用三个固定的 3×3 核去「扫」邻域：自身值 + 横向与纵向的变化（Sobel 梯度）",
             "Three fixed 3x3 kernels: its own value + left-right and up-down change (Sobel gradients)"),
        (130, "16 个通道 × 3 个核 = 48 个感知值，这就是它对世界的全部了解",
              "16 channels x 3 kernels = 48 numbers: everything the cell knows about the world"),
    ])
    y0, x0 = 4, 6
    zy, zx, zc = 6, 90, 10                         # 3x3 zoom: 10 px cells
    nb = x[py - 1: py + 2, px - 1: px + 2]
    nb_rgb = to_rgb(nb, BG_ARR)
    n = 192
    for i in range(n):
        phase = min(2, max(0, (i - 40) // 30)) if i < 130 else -1   # which kernel is active
        cv = Canvas(rows, cols)
        _draw_lizard(cv, x, y0, x0)
        cy, cx = y0 + py, x0 + px
        cv.outline(cy - 2, cx - 2, 5, 5, WHITE)
        _line(cv, cy - 2, cx + 3, zy, zx - 2, scale_rgb(WHITE, 0.45))
        _line(cv, cy + 2, cx + 3, zy + 3 * zc, zx - 2, scale_rgb(WHITE, 0.45))
        cv.cells(nb_rgb, np.ones((3, 3), bool), zy, zx, zc, gap=1)
        cv.outline(zy + zc - 1, zx + zc - 1, zc + 1, zc + 1, WHITE)   # centre cell
        # output bars: 3 groups of 16
        reveal = smoothstep((i - 130) / 30)
        gx0, gy = 88, 68
        for g in range(3):
            for k in range(16):
                v = float(per[g * 16 + k]) * reveal
                h = int(np.clip(round(abs(v) * 6), 0, 9))
                col = [TEXT, AMBER, CYAN][g]
                xk = gx0 + g * 22 + k
                if h:
                    if v >= 0:
                        cv.rect(gy - h, xk, h, 1, col)
                    else:
                        cv.rect(gy, xk, h, 1, scale_rgb(col, 0.6))
        if reveal > 0:
            cv.rect(gy, gx0 - 1, 1, 3 * 22, scale_rgb(DIM, 0.5))
        f = cv.to_frame()
        f.put_text(4, 124, "3×3 邻域", TEXT)
        f.put_text(5, 124, "neighbourhood", DIM)
        # weights of the active kernel written on the zoomed cells
        if phase >= 0 and i >= 40:
            _, kern, kc = KERNELS[phase]
            for r in range(3):
                for c in range(3):
                    w = int(kern[r, c])
                    s = f"{w:+d}" if w else "0"
                    rowc = (zy + r * zc + zc // 2) // 2
                    colc = zx + c * zc + (zc - 1) // 2 - len(s) // 2
                    cellbg = _t(scale_rgb(nb_rgb[r, c], 0.18))
                    f.put_text(rowc, colc, s, _t(WHITE if kc == TEXT else kc), cellbg)
        for k, (title, kern, kc) in enumerate(KERNELS):
            c0 = 84 + k * 25
            on = phase == k or phase == -1 or i < 40
            col = kc if on else FAINT
            box(f, 20, c0, 6, 24, col, title=title, title_rgb=col if on else DIM)
            for r in range(3):
                line = "  ".join(f"{int(v):+d}" if v else " 0" for v in kern[r])
                f.put_text(21 + r + 1, c0 + 4, line, _t(TEXT if on else DIM))
        if reveal > 0:
            f.put_text(28, 88, "自身 self", TEXT)
            f.put_text(28, 110, "∂/∂x", AMBER)
            f.put_text(28, 132, "∂/∂y", CYAN)
            f.put_text(36, 88, "→ 48 个感知值 · 48 perception values", _t(scale_rgb(TEXT, reveal)))
        if i in (40, 70, 100, 130):
            f.sfx.append(("blip", 0.4 + 0.1 * ((i - 40) // 30)))
        yield _caption(cap, f, i)


# --- scene: (3) per-cell network ------------------------------------------------------

def _net_frames(rows: int, cols: int) -> Iterator[Frame]:
    nca = _nca()
    x = _grown()
    py, px = _probe_cell(x)
    per = perceive(x)
    cap = Captions([
        (0, "③ 更新：每个格子里运行同一个小神经网络 48 → 128 → 16",
            "(3) Update: every cell runs the same tiny neural net, 48 -> 128 -> 16"),
        (60, "输出 16 个数：状态该怎么变（Δ），加回到原来的状态上",
             "It outputs 16 numbers - how the state should change (Δ) - added back to the state"),
        (120, "整条「生命规则」只有 8,320 个参数，所有格子共用",
              "The whole 'rule of life' is just 8,320 weights, shared by every cell"),
    ])
    # Only a readable slice of the real network is drawn: 8 of the 48 inputs,
    # 9 of the 128 hidden units (the most active ones) and 6 of the 16
    # outputs, joined by their real weights; the rest is marked '+N more'.
    CH = ["R", "G", "B", "α"] + [f"h{k}" for k in range(1, 13)]
    in_idx = [0, 1, 2, 3, 16 + 3, 32 + 3, 4, 16 + 4]
    in_lab = [CH[j % 16] if j < 16 else ("∂x " if j < 32 else "∂y ") + CH[j % 16] for j in in_idx]
    out_idx = [0, 1, 2, 3, 4, 5]
    W1, W2 = nca.W1, nca.W2
    probes = [(py, px), (py - 8, px - 6), (py + 10, px + 6)]
    probes = [(r, c) for r, c in probes if x[r, c, 3] > ALIVE_THRESHOLD] or [(py, px)]
    acts = np.mean([np.maximum(per[r, c] @ W1 + nca.b1, 0) for r, c in probes], axis=0)
    hid_idx = sorted(np.argsort(-acts)[:9].tolist())
    XI, XH, XO = 30, 72, 112
    inp = [(12 + 7 * k, XI) for k in range(len(in_idx))]
    hid = [(9 + 7 * k, XH) for k in range(len(hid_idx))]
    out = [(17 + 8 * k, XO) for k in range(len(out_idx))]
    sub1 = W1[np.ix_(in_idx, hid_idx)]
    sub2 = W2[np.ix_(hid_idx, out_idx)]
    m1, m2 = np.abs(sub1).max(), np.abs(sub2).max()
    R = 2.6
    n = 180
    for i in range(n):
        cyc = (i % 60) / 60
        pr, pc = probes[(i // 60) % len(probes)]
        y = per[pr, pc]
        h = np.maximum(y @ W1 + nca.b1, 0)
        d = h @ W2
        yi, hi_, do = y[in_idx], h[hid_idx], d[out_idx]
        a_in = np.clip(np.abs(yi) / (np.abs(yi).max() + 1e-6), 0, 1)
        a_h = np.clip(hi_ / (hi_.max() + 1e-6), 0, 1)
        a_o = np.clip(np.abs(do) / (np.abs(do).max() + 1e-6), 0, 1)
        l_in, l_h, l_o = smoothstep(cyc / 0.2), smoothstep((cyc - 0.3) / 0.2), smoothstep((cyc - 0.6) / 0.2)
        cv = Canvas(rows, cols)
        for a, (ya, xa) in enumerate(inp):
            for b, (yb, xb) in enumerate(hid):
                w = sub1[a, b] / m1
                if abs(w) < 0.3:
                    continue
                k = 0.2 + 0.8 * l_h * a_in[a]
                _line(cv, ya, xa + R + 1, yb, xb - R - 1, scale_rgb(AMBER if w > 0 else CYAN, k * abs(w)), add=True)
        for a, (ya, xa) in enumerate(hid):
            for b, (yb, xb) in enumerate(out):
                w = sub2[a, b] / m2
                if abs(w) < 0.3:
                    continue
                k = 0.2 + 0.8 * l_o * a_h[a]
                _line(cv, ya, xa + R + 1, yb, xb - R - 1, scale_rgb(AMBER if w > 0 else CYAN, k * abs(w)), add=True)
        for j, (yy, xx) in enumerate(inp):
            base = TEXT if in_idx[j] < 16 else (AMBER if in_idx[j] < 32 else CYAN)
            _circle(cv, yy + 0.5, xx + 0.5, R, lerp_rgb(scale_rgb(base, 0.4), base, 0.4 + 0.6 * l_in * a_in[j]),
                    fill_rgb=lerp_rgb(BG, base, l_in * a_in[j]))
        for j, (yy, xx) in enumerate(hid):
            _circle(cv, yy + 0.5, xx + 0.5, R, lerp_rgb(FAINT, (255, 240, 200), 0.3 + 0.7 * l_h * a_h[j]),
                    fill_rgb=lerp_rgb(BG, (255, 240, 200), l_h * a_h[j]))
        for j, (yy, xx) in enumerate(out):
            _circle(cv, yy + 0.5, xx + 0.5, R, lerp_rgb(FAINT, GREEN, 0.3 + 0.7 * l_o * a_o[j]),
                    fill_rgb=lerp_rgb(BG, GREEN, l_o * a_o[j]))
        for (yy, xx) in ((inp[-1][0] + 7, XI), (hid[-1][0] + 6, XH), (out[-1][0] + 8, XO)):
            for dy in (0, 2, 4):
                cv.rect(yy + dy, xx, 1, 1, DIM)
        f = cv.to_frame()
        for j, (yy, xx) in enumerate(inp):
            lab = in_lab[j]
            f.put_text(yy // 2, xx - 4 - len(lab), lab, _t(DIM))
        for j, (yy, xx) in enumerate(out):
            f.put_text(yy // 2, xx + 5, "Δ" + CH[out_idx[j]], _t(GREEN))
        f.put_text((inp[-1][0] + 7) // 2, XI + 3, "+40", FAINT)
        f.put_text((hid[-1][0] + 6) // 2, XH + 3, "+119", FAINT)
        f.put_text((out[-1][0] + 8) // 2, XO + 3, "+10", FAINT)
        f.put_text(1, XI - 12, "48 个感知值（显示 8 个）", TEXT)
        f.put_text(2, XI - 12, "48 inputs (8 shown)", DIM)
        f.put_text(1, XH - 12, "128 个神经元（显示 9 个）", TEXT)
        f.put_text(2, XH - 12, "128 hidden units (9 shown)", DIM)
        f.put_text(5, XO - 6, "16 个 Δ（显示 6 个）", TEXT)
        f.put_text(6, XO - 6, "16 outputs (6 shown)", DIM)
        box(f, 11, 124, 9, 35, DIM, title="新状态 new state")
        f.put_text(13, 126, "状态 + Δ × 随机掩码", TEXT)
        f.put_text(15, 126, "state + Δ × random mask", DIM)
        f.put_text(17, 126, "(掩码见下一步 see next)", FAINT)
        wk = smoothstep((i - 120) / 20)
        if wk > 0:
            c = _t(scale_rgb(AMBER, wk))
            f.put_text(26, 126, "参数 weights", _t(scale_rgb(TEXT, wk)))
            f.put_text(28, 126, "W1  48 × 128 = 6,144", c)
            f.put_text(29, 126, "b1        128 =   128", c)
            f.put_text(30, 126, "W2 128 ×  16 = 2,048", c)
            f.put_text(31, 126, "─────────────────────", _t(scale_rgb(DIM, wk)))
            f.put_text(32, 126, "合计 total    = 8,320", _t(scale_rgb(WHITE, wk)))
        f.put_text(38, 14, "只画出网络的一小部分 · 线 = 真实权重（橙 +，蓝 −）· a slice of the real net; lines = real weights (orange +, blue −)", FAINT)
        if i % 60 == 18:
            f.sfx.append(("tick", 0.5))
        if i == 120:
            f.sfx.append(("chime", 0.6))
        yield _caption(cap, f, i)


# --- scene: (4) stochastic update + alive mask ------------------------------------------

def _stoch_steps(i: int) -> int:
    """Model steps to run after frame i: very slow at first (watch single
    cells fire), then faster while the lizard grows."""
    if i < 40:
        return int(i % 4 == 3)
    if i < 64:
        return 1
    return 3


def _stochastic_frames(rows: int, cols: int) -> Iterator[Frame]:
    """A lizard *growing* from its seed, every cell's update shown: cells
    drawn at 4 px (3 px + 1 px gap); a cell that fires this step gets a white
    dot in its centre, a waiting cell is drawn as is. Fixed camera: the seed is
    at the bottom centre and the lizard grows up into the frame."""
    nca, rng = _nca(), np.random.default_rng(5)
    x = seed_state(GRID, GRID)
    cap = Captions([
        (0, "④ 没有全局时钟：每一步，每个格子只有 50% 的机会更新",
            "(4) No global clock: each step, every cell updates with only 50% chance"),
        (40, "α 不足 0.1、且邻居也都不足 0.1 的格子被视为死亡，状态清零",
             "A cell whose α and all neighbours' α are below 0.1 is dead: its state is zeroed"),
        (68, "所以生长只能从活细胞的边缘向外发生——就像真实的组织",
             "So growth can only happen at the edge of living tissue - like real tissue"),
    ])
    n, step_no = 100, 0
    CELLP = 4                                   # px per cell incl. 1 px gap
    VIEW_W = 122                                # px; the legend lives to the right
    y0 = 76 - (GRID // 2) * CELLP               # seed cell's top-left at (76, ~59)
    x0 = (VIEW_W - CELLP) // 2 - (GRID // 2) * CELLP
    mask = rng.random((GRID, GRID)) <= nca.fire_rate
    for i in range(n):
        k = _stoch_steps(i)
        alive = alive_mask(x)
        a = x[..., 3] > ALIVE_THRESHOLD
        disp = np.where(a[..., None], to_rgb(x, BG_ARR), np.array(FAINT, np.uint8)).astype(np.uint8)
        if step_no == 0:   # the seed's colour channels are 0 - show it white
            disp[GRID // 2, GRID // 2] = WHITE
        cv = Canvas(rows, cols)
        cv.cells(disp, alive | a, y0, x0, CELLP, gap=1)
        # this step's updates: a white dot in the centre of each firing cell
        rr, cc = np.nonzero(mask & alive)
        py, px = y0 + rr * CELLP + 1, x0 + cc * CELLP + 1
        ok = (py >= 0) & (py < 80) & (px >= 0) & (px < VIEW_W)
        cv.filled[py[ok], px[ok]] = True
        cv.rgb[py[ok], px[ok]] = WHITE
        cv.clear(0, VIEW_W, 80, cv.W - VIEW_W)
        # legend swatches: a firing cell and a waiting cell
        L = 126
        sw = np.array((80, 200, 120), np.uint8)
        cv.rect(12, L, 3, 3, sw)
        cv.rect(13, L + 1, 1, 1, WHITE)
        cv.rect(18, L, 3, 3, sw)
        cv.rect(24, L, 3, 3, FAINT)
        f = cv.to_frame()
        f.put_text(2, L, "随机更新 stochastic update", TEXT)
        f.put_text(3, L, f"step {step_no}", DIM)
        f.put_text(6, L + 5, "本步更新（中心白点）", TEXT)
        f.put_text(7, L + 5, "fires this step (p = 0.5)", DIM)
        f.put_text(9, L + 5, "本步等待", TEXT)
        f.put_text(10, L + 5, "waits this step", DIM)
        f.put_text(12, L + 5, "空但挨着活细胞", TEXT)
        f.put_text(13, L + 5, "empty, next to life:", DIM)
        f.put_text(14, L + 5, "can grow", DIM)
        f.put_text(19, L, "活 = 3×3 内有 α > 0.1", AMBER)
        f.put_text(20, L, "alive = any α > 0.1 in 3x3", DIM)
        f.put_text(24, L, "从一个种子开始生长（镜头固定）", DIM)
        f.put_text(25, L, "growing from one seed (fixed camera)", FAINT)
        if k and i < 40:
            f.sfx.append(("tick", 0.4))
        yield _caption(cap, f, i)
        if k:
            x = nca.step(x, update_mask=mask)   # the mask shown is the one used
            for _ in range(k - 1):
                x = nca.step(x, rng)
            step_no += k
            mask = rng.random((GRID, GRID)) <= nca.fire_rate


# --- scene: (5) training -----------------------------------------------------------------

@lru_cache(maxsize=1)
def _target_state() -> np.ndarray:
    img = Image.open(TARGET_PNG).convert("RGBA")
    img.thumbnail((40, 40), Image.LANCZOS)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a[..., :3] *= a[..., 3:]
    p = (GRID - a.shape[0]) // 2, (GRID - a.shape[1]) // 2
    out = np.zeros((GRID, GRID, 4), np.float32)
    out[p[0]: p[0] + a.shape[0], p[1]: p[1] + a.shape[1]] = a
    return out


def _thumb(cv: Canvas, x: np.ndarray, y0: int, x0: int, bite: float = 0.0) -> None:
    p = _pool2(x[..., :4])[3:33, 3:33]
    if bite:
        p = _bite(p, 16, 20, 6 * bite)
    cv.blit(p[..., 3] > 0.05, to_rgb(p, BG_ARR), y0, x0)


def _train_frames(rows: int, cols: int) -> Iterator[Frame]:
    nca, rng, x = _nca(), np.random.default_rng(3), seed_state(GRID, GRID)
    snaps = {}
    for s in range(1, 97):
        x = nca.step(x, rng)
        if s in (16, 32, 64, 96):
            snaps[s] = x
    target = _target_state()
    cap = Captions([
        (0, "⑤ 训练：从一个种子出发，让网络跑 64～96 步",
            "(5) Training: start from one seed and run the network for 64-96 steps"),
        (42, "把结果和目标图片逐像素比较，误差沿时间反向传播，调整那 8,320 个参数",
             "Compare with the target pixel by pixel; backpropagate the error through time; adjust the 8,320 weights"),
        (108, "还会故意破坏训练样本，逼它学会保持形状和再生——共训练 8,000 轮",
              "Samples are also deliberately damaged, so it learns to persist and regrow - 8,000 rounds in all"),
    ])
    xs = [4, 44, 84, 124]
    labels = [16, 32, 64, 96]
    n = 168
    # loss plot geometry
    gx0, gy0, gw, gh = 12, 46, 90, 26
    lo, hi = -4.4, -1.4
    for i in range(n):
        cv = Canvas(rows, cols)
        for k, (xx, s) in enumerate(zip(xs, labels)):
            if i >= 6 + 8 * k:
                _thumb(cv, snaps[s], 4, xx, bite=smoothstep((i - 120) / 10) if (s == 96 and i >= 120) else 0.0)
                cv.outline(3, xx - 1, 32, 32, scale_rgb(DIM, 0.35))
        tr = smoothstep((i - 40) / 12)
        if tr > 0:
            p = _pool2(target)[3:33, 3:33]
            cv.blit(p[..., 3] > 0.05, lerp_rgb(BG, to_rgb(p, BG_ARR), tr), 46, 124)
            cv.outline(45, 123, 32, 32, scale_rgb(AMBER, 0.8 * tr))
        # the loss curve (real training log), drawn progressively
        pr = smoothstep((i - 60) / 50)
        if pr > 0:
            cv.rect(gy0 + gh, gx0, 1, gw, scale_rgb(DIM, 0.6))
            cv.rect(gy0, gx0, gh + 1, 1, scale_rgb(DIM, 0.6))
            nshow = int(pr * len(LOSS_LOG10))
            for j in range(1, nshow):
                ya = gy0 + (hi - LOSS_LOG10[j - 1]) / (hi - lo) * gh
                yb = gy0 + (hi - LOSS_LOG10[j]) / (hi - lo) * gh
                xa = gx0 + 1 + (j - 1) / (len(LOSS_LOG10) - 1) * (gw - 2)
                xb = gx0 + 1 + j / (len(LOSS_LOG10) - 1) * (gw - 2)
                _line(cv, ya, xa, yb, xb, lerp_rgb(PINK, AMBER, j / len(LOSS_LOG10)))
        f = cv.to_frame()
        for k, (xx, s) in enumerate(zip(xs, labels)):
            if i >= 6 + 8 * k:
                f.put_text(19, xx + 9, f"step {s}", DIM)
                if k < 3:
                    f.put_text(10, xx + 32, "→", TEXT)
        if i >= 120:
            f.put_text(1, 124, "故意破坏 damage", PINK)
        if tr > 0:
            f.put_text(21, 136, "↓", _t(scale_rgb(TEXT, tr)))
            f.put_text(21, 138, "比较 compare", _t(scale_rgb(TEXT, tr)))
            f.put_text(38, 127, "目标 target", _t(scale_rgb(AMBER, tr)))
            f.put_text(29, 104, "误差 error ←", _t(scale_rgb(PINK, tr)))
        if pr > 0:
            f.put_text(22, gx0, "训练误差 loss (log₁₀)  ·  真实训练记录 real training log", TEXT)
            f.put_text(23, gx0 - 3, "-1.5", DIM)
            f.put_text(36, gx0 - 3, "-4.4", DIM)
            f.put_text(37, gx0, "0", DIM)
            f.put_text(37, gx0 + gw - 10, "8000 轮", DIM)
            f.put_text(38, gx0 + 20, "梯度下降 gradient descent ↘", AMBER)
        if i in (6, 14, 22, 30):
            f.sfx.append(("blip", 0.3 + 0.1 * ((i - 6) // 8)))
        if i == 120:
            f.sfx.append(("swish", 0.6))
        yield _caption(cap, f, i)


# --- scene: grow + erase + cut, with hidden channels ---------------------------------------

REGEN_CAPTIONS = [
    (0, "从一个种子开始：左边是看得见的颜色，右边是四个隐藏通道",
        "From one seed: visible colour on the left, four hidden channels on the right"),
    (40, "隐藏通道像胚胎里的「化学梯度」，让细胞知道自己在身体的哪个部位",
         "Hidden channels act like chemical gradients in an embryo: cells learn where in the body they are"),
    (84, "长成后，它会一直保持这个形状", "Once grown, it holds its shape"),
    (114, "用橡皮擦掉一块……", "Now erase a chunk..."),
    (150, "……缺口边缘的细胞「感知」到异常，把它重新长了回来",
          "...cells at the wound's edge sense something is wrong, and grow it back"),
    (228, "再狠一点：把它切成两半", "Harsher: cut it in half"),
    (250, "只剩一半细胞，也能重建整只蜥蜴——这是学来的，不是写出来的",
          "Half the cells rebuild the entire lizard - learned, not programmed"),
]
# grown ~78 -> erase 3 s later; healed ~54 frames after -> cut 2 s later;
# healed ~83 frames after the cut (measured) -> end ~1 s later
ERASE_START, ERASE_PRESS, ERASE_END, ERASE_EXIT = 114, 124, 142, 150
CUT_START, CUT_END = 228, 240
REGEN_FRAMES = 336


def _eraser_path(i: int) -> Tuple[float, float, bool]:
    """Cursor position (grid row, col) and whether it's pressed."""
    a, b, c, d = (40, 84), (27, 47), (45, 27), (8, 70)
    if i < ERASE_PRESS:
        t = ease_in_out((i - ERASE_START) / (ERASE_PRESS - ERASE_START))
        p, q, on = a, b, False
    elif i < ERASE_END:
        t = (i - ERASE_PRESS) / (ERASE_END - ERASE_PRESS - 1)
        p, q, on = b, c, True
    else:
        t = ease_in_out((i - ERASE_END) / (ERASE_EXIT - ERASE_END))
        p, q, on = c, d, False
    return p[0] + (q[0] - p[0]) * t, p[1] + (q[1] - p[1]) * t, on


def _regen_frames(rows: int, cols: int) -> Iterator[Frame]:
    nca, rng, x = _nca(), np.random.default_rng(2020), seed_state(GRID, GRID)
    cap = Captions(REGEN_CAPTIONS)
    my, mx = 4, 6
    panels = [(4, 84), (4, 122), (42, 84), (42, 122)]
    radius = 6.0
    steps = 0
    piece = None          # severed half: (filled, rgb) falling away
    piece_t = 0
    for i in range(REGEN_FRAMES):
        cursor = None
        if ERASE_START <= i < ERASE_EXIT:
            cy, cx, on = _eraser_path(i)
            cursor = (cy, cx, on)
            if on:
                x = _bite(x, cy, cx, radius)
        knife_y = None
        if CUT_START <= i < CUT_END:
            knife_y = int(round((i - CUT_START + 1) / (CUT_END - CUT_START) * GRID))
            half = x[:knife_y, GRID // 2:]
            if piece is None:
                piece = [np.zeros((GRID, GRID // 2), bool), np.zeros((GRID, GRID // 2, 3), np.uint8)]
            cut_rows = slice(0, knife_y)
            alive = half[..., 3] > ALIVE_THRESHOLD
            piece[0][cut_rows] |= alive
            piece[1][cut_rows][alive] = to_rgb(half, BG_ARR)[alive]
            x = x.copy()
            x[:knife_y, GRID // 2:] = 0.0
        if i >= 1:
            x = nca.step(x, rng)
            steps += 1
        cv = Canvas(rows, cols)
        _draw_lizard(cv, x, my, mx)
        # the severed half slides away and fades
        if piece is not None and i >= CUT_END:
            piece_t += 1
        if piece is not None and piece_t < 16:
            fade = 1 - piece_t / 16
            dy, dx = int(piece_t * 1.2), int(piece_t * 0.8)
            col = lerp_rgb(piece[1], (255, 70, 70), 0.5 * (1 - fade) + 0.2).astype(np.float32) * fade
            fm = piece[0].copy()
            # keep it inside the main panel
            w_lim = max(0, 78 - (mx + GRID // 2 + dx))
            fm[:, w_lim:] = False
            cv.blit(fm, col.astype(np.uint8), my + dy, mx + GRID // 2 + dx)
        for (py0, px0), (ch, cmap, lo, hi) in zip(panels, HIDDEN):
            _draw_hidden(cv, x, py0, px0, ch, cmap, lo, hi)
            cv.outline(py0 - 1, px0 - 1, 38, 38, scale_rgb(DIM, 0.3))
        cv.outline(my - 1, mx - 1, GRID + 2, GRID + 2, scale_rgb(DIM, 0.3))
        if cursor is not None:
            cy, cx, on = cursor
            _circle(cv, my + cy, mx + cx, radius, WHITE if on else scale_rgb(WHITE, 0.7),
                    fill_rgb=(255, 120, 120) if on else None)
            for (py0, px0) in panels:
                _circle(cv, py0 + cy / 2, px0 + cx / 2, radius / 2, scale_rgb(WHITE, 0.55))
        if knife_y is not None:
            kx = mx + GRID // 2
            _line(cv, my - 2, kx, my + knife_y, kx, (255, 250, 230))
            _line(cv, my - 2, kx - 1, my + knife_y, kx - 1, scale_rgb(AMBER, 0.5))
            cv.rect(my + knife_y - 1, kx - 1, 3, 3, WHITE)
            for (py0, px0) in panels:
                _line(cv, py0, px0 + 18, py0 + knife_y / 2, px0 + 18, scale_rgb((255, 250, 230), 0.6))
        f = cv.to_frame()
        f.put_text(2, 8, "RGBA · 看得见的颜色 visible", TEXT)
        for (py0, px0), (ch, cmap, lo, hi) in zip(panels, HIDDEN):
            f.put_text(py0 // 2, px0 + 1, f"隐藏通道 hidden #{ch - 3}", DIM)
        f.put_text(39, 8, f"step {steps}", DIM)
        if i == ERASE_PRESS:
            f.sfx.append(("swish", 1.0))
        if i == CUT_START:
            f.sfx.append(("swish", 1.2))
        if i == CUT_END - 1:
            f.sfx.append(("hit", 0.7))
        if i == 204:
            f.sfx.append(("chime", 0.7))
        if i == 330:
            f.sfx.append(("chime", 0.9))
        yield _caption(cap, f, i)


def build_scenes(rows: int, cols: int) -> List[Scene]:
    return [
        title_card(6, "神经元胞自动机", "Growing Neural CA", ACCENT, music="wonder"),
        Scene("ch6-hook", lambda: _hook_frames(rows, cols), fps=12, transition="iris",
              transition_ticks=16, music="wonder"),
        Scene("ch6-state", lambda: _state_frames(rows, cols), fps=24, transition="wipe",
              transition_ticks=16, music="curious"),
        Scene("ch6-perceive", lambda: _perceive_frames(rows, cols), fps=24, transition="cut",
              transition_ticks=14, music="curious"),
        Scene("ch6-network", lambda: _net_frames(rows, cols), fps=24, transition="wipe_down",
              transition_ticks=16, music="curious"),
        Scene("ch6-stochastic", lambda: _stochastic_frames(rows, cols), fps=8, transition="rain",
              transition_ticks=16, music="curious"),
        Scene("ch6-training", lambda: _train_frames(rows, cols), fps=12, transition="dissolve",
              transition_ticks=14, music="build"),
        Scene("ch6-regenerate", lambda: _regen_frames(rows, cols), fps=12, transition="rule30",
              transition_ticks=20, music="wonder"),
    ]
