"""Chapter 7: recap montage (six systems running live side by side), the
closing message, references & credits, the author card, and a 'thanks' card
that - like the opening title - dissolves under the Game of Life. In the
terminal player the thanks card keeps evolving until a key is pressed
(see endless_frames); the video simply ends."""
from itertools import islice
from typing import Iterator, List, Tuple

import numpy as np

from chapters.common import ReverseReveal, title_card
from engine.canvas import (AMBER, BG, CYAN, DIM, PH, PW, TEXT, Canvas, Captions, font_text,
                           hsv, lerp_rgb, pixel_text, put_center, scale_rgb, smoothstep)
from engine.frame import Frame, Scene
from sim import elementary_ca
from sim import langton_loops
from sim.game_of_life import step as life_step
from sim.nca.model import ALIVE_THRESHOLD, NCA, seed_state, to_rgb
from sim.turmite import Ant, run as turmite_run, vant

CHAPTER_LABEL = ("第 7 章 · 总结", "Chapter 7 · Recap")
NO_CHAPTER_LABEL = ("ch7-refs", "ch7-credits", "ch7-author", "ch7-thanks")
GITHUB = "GitHub: https://github.com/song88180/Cellular_Automata_AI_Video"

FPS = 12
BG_ARR = np.array(BG, dtype=np.uint8)
ACCENT = (255, 205, 120)
PW_, PH_ = 52, 38                                     # top-row panel size (px)
BW = 79                                               # bottom-row panels are wider
# (y, x, h, w) of each panel: 3 on top, 2 wide ones below
BOXES = [(2, 0, PH_, PW_), (2, 54, PH_, PW_), (2, 108, PH_, PW_), (42, 0, PH_, BW), (42, 81, PH_, BW)]
LABELS = [
    "一维 Rule 30 · 混沌", "生命游戏 Game of Life", "兰顿蚂蚁 Langton's ant",
    "兰顿环 Langton's loops", "神经元胞自动机 NCA",
]
LABEL_COLORS = [(255, 130, 60), (255, 200, 80), (255, 90, 160), (80, 230, 110), (110, 230, 130)]

AGE_STOPS = [0, 3, 10, 30]
AGE_RGB = np.array([(255, 245, 170), (255, 170, 60), (230, 70, 140), (90, 110, 255)], dtype=float)
ANT_RGB = np.array((255, 200, 80), np.uint8)
NCA_BITES = (110, 158, 206)                           # recap frames: a bite ~every 4 s once grown
ANT_WORLD = (240, 320)
LOOP_WORLD = (160, 420)


def _age_colors(age: np.ndarray) -> np.ndarray:
    out = np.empty(age.shape + (3,), dtype=np.uint8)
    for ch in range(3):
        out[..., ch] = np.interp(age, AGE_STOPS, AGE_RGB[:, ch]).astype(np.uint8)
    return out


Panel = Tuple[np.ndarray, np.ndarray, Tuple[float, float]]  # rgb, filled, camera centre (cells, 1 px/cell)


def _bite(x: np.ndarray, rng) -> np.ndarray:
    ys, xs = np.nonzero(x[..., 3] > 0.5)
    near = np.hypot(ys - ys.mean(), xs - xs.mean()) < 12   # bite the body, not a toe
    ys, xs = ys[near], xs[near]
    k = rng.integers(len(ys))
    cy, cx = ys[k], xs[k]
    yy, xx = np.mgrid[: x.shape[0], : x.shape[1]]
    x = x.copy()
    x[(yy - cy) ** 2 + (xx - cx) ** 2 < 8 ** 2] = 0.0
    return x


class _FollowCam:
    """Camera that stays still while a target is inside a dead zone of the
    view, and glides after it when it gets near the edge."""

    def __init__(self, centre, view_hw, margin: float = 0.3, ease: float = 0.12):
        self.c = np.array(centre, np.float64)
        self.half = np.array(view_hw, np.float64) / 2
        self.margin, self.ease = margin, ease

    def update(self, target) -> Tuple[float, float]:
        t = np.array(target, np.float64)
        lim = self.half * (1 - self.margin)
        want = self.c.copy()
        d = t - self.c
        want = np.where(d > lim, t - lim, np.where(d < -lim, t + lim, want))
        self.c += (want - self.c) * self.ease
        return float(self.c[0]), float(self.c[1])


def _panels() -> Iterator[List[Panel]]:
    """Yields the 5 panels for one frame, advancing every system. Every panel
    stays at 1 px per cell; the ant's and the loops' cameras move instead."""
    # 1D Rule 30: generate wide, show the centre, scroll one row per frame
    r30 = elementary_ca.generate(3 * PW_, 30, steps=PH_ + 420)[:, PW_: 2 * PW_]
    # Life soup
    life = np.random.default_rng(11).random((PH_, PW_)) < 0.35
    age = np.zeros(life.shape, dtype=np.int32)
    # classic Langton's ant, pre-run so its highway starts during the recap
    AH, AW = ANT_WORLD
    ant_grid = np.zeros((AH, AW), dtype=np.int64)
    ant = Ant(AH // 2, AW // 2, heading=0)
    rl = vant("RL")
    turmite_run(ant_grid, ant, rl, 9200)   # highway starts ~5 s into the recap
    ys, xs = np.nonzero(ant_grid)
    ant_cam = _FollowCam(((ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2), (PH_, PW_), margin=0.25)
    # Langton's loops: one loop near the left of a wide world; the camera pans right
    LH, LW = LOOP_WORLD
    loops = np.zeros((LH, LW), dtype=np.uint8)
    L = langton_loops.initial_loop()
    lr, lc = LH // 2 - 5, 150   # far from every edge: nothing wraps around the torus
    loops[lr: lr + L.shape[0], lc: lc + L.shape[1]] = L
    loop_x = lc + L.shape[1] / 2
    # NCA, bitten now and then
    nca = NCA.load()
    nca_x = seed_state(72, 72)
    nca_rng = np.random.default_rng(5)
    bite_rng = np.random.default_rng(7)
    k = 0
    while True:
        w30 = r30[max(0, k - PH_ + 1): k + 1]
        p1 = np.zeros((PH_, PW_), bool)
        p1[: len(w30)] = w30 == 1
        c1 = np.zeros((PH_, PW_, 3), np.uint8)
        c1[: len(w30)] = hsv(0.02 + 0.08 * np.arange(len(w30))[:, None] / PH_ * np.ones((1, PW_)), 0.75, 1.0)

        p3 = ant_grid > 0
        c3 = np.broadcast_to(ANT_RGB, (AH, AW, 3)).copy()
        p3[ant.r, ant.c] = True
        c3[ant.r, ant.c] = (255, 60, 60)
        cam_ant = ant_cam.update((ant.r, ant.c))

        p4, c4 = loops > 0, langton_loops.PALETTE[loops]
        # keep the colony's right-hand growth front in view
        cols_alive = np.nonzero(p4.any(axis=0))[0]
        front = cols_alive.max() if len(cols_alive) else loop_x
        loop_x += (max(loop_x, front - BW * 0.35) - loop_x) * 0.08

        yield [(c1, p1, (PH_ / 2, PW_ / 2)), (_age_colors(age), life, (PH_ / 2, PW_ / 2)),
               (c3, p3, cam_ant), (c4, p4, (LH / 2, loop_x)),
               (to_rgb(nca_x, BG_ARR), nca_x[..., 3] > ALIVE_THRESHOLD, (36.0, 36.0))]

        k += 1
        new = life_step(life)
        age = np.where(new & life, age + 1, 0)
        life = new
        turmite_run(ant_grid, ant, rl, 12)
        loops = langton_loops.run(loops, 3)
        if k in NCA_BITES:
            nca_x = _bite(nca_x, bite_rng)
        for _ in range(2):   # grown after ~5 s
            nca_x = nca.step(nca_x, nca_rng)


RECAP = [
    (0, "回顾：一维规则里的混沌，二维网格上的「生命」", "Recap: chaos from a 1D rule, 'life' on a 2D grid"),
    (36, "修「高速公路」的兰顿蚂蚁，和会自我复制的兰顿环",
         "Langton's ant building its highway, and Langton's loops copying themselves"),
    (72, "还有学会了生长、被咬掉一块也能长回来的神经网络",
         "And a neural net that learned to grow - and regrows when a piece is bitten off"),
    (112, "共同点：只有局部规则，没有总设计师——自然界处处是它们的影子",
          "What they share: only local rules, no master plan - and nature is full of them"),
    (180, "复杂，可以从简单中「涌现」", "Complexity can emerge from simplicity"),
]
RECAP_FRAMES = 252
FINALE = 180


def _recap_frames(rows: int, cols: int) -> Iterator[Frame]:
    panels = _panels()
    cap = Captions(RECAP)
    big = font_text("简单规则 → 复杂世界", 13)
    en = np.zeros((6, 1), bool)   # room for the English line (drawn as text)
    for i in range(RECAP_FRAMES):
        data = next(panels)
        cv = Canvas(rows, cols)
        dim = 1.0 - 0.72 * smoothstep((i - FINALE) / 18)
        # panels light up one by one at the start
        for j, ((y0, x0, h, w), (c, p, src)) in enumerate(zip(BOXES, data)):
            on = smoothstep((i - 4 * j) / 10)
            if on <= 0:
                continue
            cv.zoom(scale_rgb(c, on * dim), p, 1.0, src, (y0 + h / 2, x0 + w / 2), gap=False,
                    region=(y0, x0, h, w))
        k = smoothstep((i - FINALE - 6) / 14) if i >= FINALE else 0.0
        by = (PH - big.shape[0]) // 2 - 6
        plate = (by - 6, 10, big.shape[0] + en.shape[0] + 18, PW - 20)   # (y, x, h, w) px
        if k > 0:
            cv.clear(*plate)      # dark plate behind the message for legibility
            cv.blit(big, lerp_rgb(BG, (255, 245, 225), k), by, (PW - big.shape[1]) // 2)
        f = cv.to_frame()
        if k > 0:
            put_center(f, (by + big.shape[0] + 6) // 2 + 1, "S I M P L E   R U L E S   →   C O M P L E X   W O R L D S",
                       lerp_rgb(BG, ACCENT, k))
        prow0, prow1 = plate[0] // 2, (plate[0] + plate[2] + 1) // 2
        for j, ((y0, x0, h, w), lab) in enumerate(zip(BOXES, LABELS)):
            row = y0 // 2 - 1
            if k > 0 and prow0 <= row <= prow1:
                continue          # the plate covers this label
            if i >= 4 * j + 4:
                f.put_text(row, x0 + 1, lab, tuple(int(v) for v in scale_rgb(LABEL_COLORS[j], dim)))
        if i == FINALE + 6:
            f.sfx.append(("hit", 1.0))
        if i in NCA_BITES:
            f.sfx.append(("swish", 1.0))
        if i == FINALE - 10:
            f.sfx.append(("rise", 0.8))
        yield cap.apply(f, i)


PAGE1 = ("参考文献  REFERENCES", [
    "J. von Neumann & A. W. Burks, Theory of Self-Reproducing Automata. Univ. of Illinois Press (1966)",
    "E. F. Codd, Cellular Automata. Academic Press (1968)",
    "S. Wolfram, \"Statistical mechanics of cellular automata\", Rev. Mod. Phys. 55, 601 (1983)",
    "S. Wolfram, A New Kind of Science. Wolfram Media (2002)",
    "M. Cook, \"Universality in elementary cellular automata\", Complex Systems 15, 1-40 (2004)",
    "M. Gardner, \"Mathematical Games: ... John Conway's new solitaire game 'life'\", Sci. Am. 223 (Oct. 1970)",
    "R. W. Gosper, the glider gun (1970)  ·  A. Wade, Gemini, a self-replicating pattern in Life (2010)",
    "C. G. Langton, \"Self-reproduction in cellular automata\", Physica D 10, 135-144 (1984)",
    "C. G. Langton, \"Studying artificial life with cellular automata\", Physica D 22, 120-149 (1986)",
    "H. Sayama, \"Toward the realization of an evolving ecosystem on cellular automata\", AROB 4th (1999)",
    "G. Turk, Vants and Turmites - github.com/gturk1/turmites (turmite rule tables)",
    "A. Mordvintsev, E. Randazzo, E. Niklasson & M. Levin, \"Growing Neural Cellular Automata\",",
    "    Distill (2020) - distill.pub/2020/growing-ca",
])
PAGE2 = ("致谢  CREDITS", [
    "Langton's loop & Evoloop rule tables and patterns: Golly (golly.sourceforge.net), after Langton / Sayama",
    "Portrait & shell cut-outs: rembg (BiRefNet-portrait), drawn as braille characters",
    "Photos via Wikimedia Commons, shown as pixel/ASCII art:",
    "    J. von Neumann - LANL  ·  J. H. Conway - Denise Applewhite, Office of Communications",
    "    S. Wolfram - Stephen Faust, CC BY-SA 3.0  ·  C. Langton - Cary Herz, CC BY-SA 3.0",
    "    Conus textile shell - Richard Ling, CC BY-SA 3.0",
    "Lizard target image: Google Noto Emoji (Apache-2.0)",
    "Music & sound effects: procedurally generated with numpy (no samples)",
    "",
    "所有模拟都是实时计算的真实结果，不是预录动画。",
    "Every simulation is computed for real - nothing is pre-recorded.",
    "Made with Python + numpy, raw ANSI escape codes, Pillow and ffmpeg.",
])


def _page_frames(rows: int, cols: int, page, n_frames: int, seed: int) -> Iterator[Frame]:
    title, lines = page
    strip = elementary_ca.generate(PW, 30, steps=400)
    left = 14
    top = (rows - 2 * len(lines) - 4) // 2
    for i in range(n_frames):
        cv = Canvas(rows, cols)
        # a thin Rule 30 strip scrolling along the top and bottom edges
        s = strip[i % 380: i % 380 + 3] == 1
        cv.blit(s, scale_rgb(AMBER, 0.35), 0, 0)
        cv.blit(s[::-1], scale_rgb(CYAN, 0.35), PH - 3, 0)
        f = cv.to_frame()
        put_center(f, top, title, AMBER)
        for j, line in enumerate(lines):
            k = smoothstep((i - 6 - 3 * j) / 8)
            if k <= 0:
                continue
            is_zh = any(ord(ch) > 0x2E80 for ch in line)
            col = TEXT if (is_zh or not line.startswith("    ")) else DIM
            f.put_text(top + 3 + 2 * j, left, line, tuple(int(v) for v in scale_rgb(col, k)))
        if i == 6:
            f.sfx.append(("chime", 0.5))
        yield f


AUTHOR_HOLD = 72     # frames after the name has landed


def _author_frames(rows: int, cols: int) -> Iterator[Frame]:
    """Author card: 'CLAUDE OPUS 5.5' re-assembles by reverse Game-of-Life
    evolution; below it, who did what, the human assistant, and the repo."""
    name = pixel_text("CLAUDE OPUS 5.5", 1, spacing=1)          # 7 px tall, like SSL
    ny, nx = 16, (PW - name.shape[1]) // 2
    ssl = pixel_text("SSL", 1, spacing=1)
    sy, sx = 50, (PW - ssl.shape[1]) // 2
    rev = ReverseReveal(name, gens=20, fps=FPS)
    for i in range(rev.n + AUTHOR_HOLD):
        cv = Canvas(rows, cols)
        rev.draw(cv, ny, nx, i, AMBER)
        k2 = smoothstep((i - rev.n - 14) / 12)
        if k2 > 0:
            cv.blit(ssl, scale_rgb(AMBER, k2), sy, sx)
        f = cv.to_frame("作者：Claude Opus 5.5 · 借助人类 SSL 完成",
                        "Author: Claude Opus 5.5 - made with the help of a human, SSL")
        put_center(f, 4, "作者  AUTHOR", (255, 255, 255))
        k = smoothstep((i - rev.n + 4) / 12)
        if k > 0:
            put_center(f, 14, "负责：资料研究、脚本与字幕、全部模拟、动画与配乐的程序", scale_rgb(TEXT, k))
            put_center(f, 15, "research, script & subtitles, and the code for every simulation, animation and the music",
                       scale_rgb(DIM, k))
        if k2 > 0:
            put_center(f, 21, "人类助手  HUMAN ASSISTANT", scale_rgb((255, 255, 255), k2))
            put_center(f, 30, "负责提供指导与反馈", scale_rgb((255, 255, 255), k2))
            put_center(f, 31, "guidance and feedback", scale_rgb(DIM, k2))
        k3 = smoothstep((i - rev.n - 30) / 12)
        if k3 > 0:
            put_center(f, 36, GITHUB, scale_rgb((255, 255, 255), k3))
        if rev.ticking(i):
            f.sfx.append(("tick", 0.3))
        if i == rev.n:
            f.sfx.append(("hit", 0.8))
        yield f


THANKS_FPS = 24
THANKS_HOLD = 72                         # 3 s still
THANKS_FILM_FRAMES = THANKS_HOLD + 120   # what the video shows; the player keeps going
THANKS_EVERY = 3                         # one generation every 3 frames = 8 gens/s, perfectly even


def _thanks_gen(rows: int, cols: int) -> Iterator[Frame]:
    """Endless: hold the text, then let it evolve under Life forever at an
    even 8 generations per second. Deterministic, so the terminal player can
    continue exactly where the film's copy stops."""
    zh = font_text("感谢观看", 20)
    en = pixel_text("THANKS FOR WATCHING", 1)
    mask = np.zeros((PH, PW), bool)
    zy = (PH - zh.shape[0]) // 2 - 6
    mask[zy: zy + zh.shape[0], (PW - zh.shape[1]) // 2: (PW - zh.shape[1]) // 2 + zh.shape[1]] = zh
    ey = zy + zh.shape[0] + 7
    mask[ey: ey + en.shape[0], (PW - en.shape[1]) // 2: (PW - en.shape[1]) // 2 + en.shape[1]] = en
    xs = np.arange(PW)
    grad = np.broadcast_to(hsv(0.08 + 0.5 * xs / PW, 0.55, 1.0)[None], (PH, PW, 3))
    g = mask.copy()
    born = np.zeros_like(g)
    i = 0
    while True:
        if i >= THANKS_HOLD and (i - THANKS_HOLD) % THANKS_EVERY == 0:
            new = life_step(g)
            born = new & ~g
            g = new
        cv = Canvas(rows, cols)
        cv.blit(g, lerp_rgb(grad, (255, 255, 255), np.where(born, 0.6, 0.0)), 0, 0)
        yield cv.to_frame("感谢观看", "Thanks for watching")
        i += 1


def _thanks_frames(rows: int, cols: int) -> Iterator[Frame]:
    return islice(_thanks_gen(rows, cols), THANKS_FILM_FRAMES)


def endless_frames(rows: int, cols: int) -> Iterator[Frame]:
    """What the terminal player shows after the film ends, until a key press."""
    return islice(_thanks_gen(rows, cols), THANKS_FILM_FRAMES, None)


ENDLESS_FPS = THANKS_FPS


def build_scenes(rows: int, cols: int) -> List[Scene]:
    return [
        title_card(7, "总结", "Recap", ACCENT, music="outro", post=36),
        Scene("ch7-recap", lambda: _recap_frames(rows, cols), fps=FPS, transition="iris",
              transition_ticks=18, music="outro"),
        Scene("ch7-refs", lambda: _page_frames(rows, cols, PAGE1, 72, 1), fps=FPS, transition="fade",
              transition_ticks=16, music="calm"),
        Scene("ch7-credits", lambda: _page_frames(rows, cols, PAGE2, 64, 2), fps=FPS, transition="wipe",
              transition_ticks=14, music="calm"),
        Scene("ch7-author", lambda: _author_frames(rows, cols), fps=FPS, transition="fade",
              transition_ticks=14, music="outro"),
        Scene("ch7-thanks", lambda: _thanks_frames(rows, cols), fps=THANKS_FPS, transition="fade",
              transition_ticks=14, music="outro"),
    ]
