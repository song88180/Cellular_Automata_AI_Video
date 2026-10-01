"""John von Neumann's 29-state cellular automaton (von Neumann neighbourhood).

States (int codes):
  0            U     ground (blank)
  1  .. 16     T     transmission: 1 + kind*8 + dir*2 + excited
                     kind 0 = ordinary, 1 = special; dir 0=E 1=N 2=W 3=S
  17 .. 20     C     confluent: 17 + 2*now + next   (now = output this step)
  21 .. 28     S     sensitized: S, S0, S1, S00, S01, S10, S11, S000

Rules (as in Burks' edition of von Neumann; see Wikipedia "Von Neumann
cellular automaton"):
- A transmission cell is excited next step if any same-kind transmission
  cell pointing at it is excited (unless the two point at each other), or an
  adjacent confluent cell is outputting and the cell doesn't point into it (OR).
- An excited ordinary cell pointing at a special cell (or vice versa) destroys
  it -> U. Excited special cells also destroy confluent cells.
- A confluent cell's inputs are the ordinary cells pointing into it; it
  outputs their AND two steps later (one step more delay than a wire).
- Construction: an excited transmission cell pointing at U turns it into S;
  each following step the sensitized cell reads input 1 / 0 (excited cell
  pointing at it or not) and walks the binary tree below until it becomes
  one of the 9 target states (quiescent).
"""
from typing import Dict, Tuple

import numpy as np

U = 0
DIRS = ((0, 1), (-1, 0), (0, -1), (1, 0))  # E N W S as (drow, dcol)
C0 = 17
S_BASE = 21
S_NAMES = ("S", "S0", "S1", "S00", "S01", "S10", "S11", "S000")


def T(kind: int, d: int, excited: int = 0) -> int:
    return 1 + kind * 8 + d * 2 + excited


def C(now: int = 0, nxt: int = 0) -> int:
    return C0 + 2 * now + nxt


def S(name: str) -> int:
    return S_BASE + S_NAMES.index(name)


def is_T(s: int) -> bool:
    return 1 <= s <= 16


def t_kind(s: int) -> int:
    return (s - 1) // 8


def t_dir(s: int) -> int:
    return ((s - 1) % 8) // 2


def t_exc(s: int) -> int:
    return (s - 1) % 2


def is_C(s: int) -> bool:
    return C0 <= s < C0 + 4


def is_S(s: int) -> bool:
    return S_BASE <= s < S_BASE + 8


# sensitized tree: state -> (next on 0, next on 1)
E_, N_, W_, S_ = range(4)
TREE: Dict[str, Tuple[int, int]] = {
    "S": (S("S0"), S("S1")),
    "S0": (S("S00"), S("S01")),
    "S00": (S("S000"), T(0, W_)),
    "S000": (T(0, E_), T(0, N_)),
    "S01": (T(0, S_), T(1, E_)),
    "S1": (S("S10"), S("S11")),
    "S10": (T(1, N_), T(1, W_)),
    "S11": (T(1, S_), C(0, 0)),
}


def _pointing_in(g: np.ndarray, r: int, c: int):
    """Yield (state, drow, dcol) of transmission neighbours pointing at (r, c)."""
    H, W = g.shape
    for d, (dr, dc) in enumerate(DIRS):
        rr, cc = r - dr, c - dc  # neighbour whose direction d points at us
        if 0 <= rr < H and 0 <= cc < W:
            s = int(g[rr, cc])
            if is_T(s) and t_dir(s) == d:
                yield s, -dr, -dc


def step(g: np.ndarray) -> np.ndarray:
    H, W = g.shape
    out = g.copy()
    for r in range(H):
        for c in range(W):
            s = int(g[r, c])
            ins = list(_pointing_in(g, r, c))
            exc_ord = any(t_exc(v) and t_kind(v) == 0 for v, _, _ in ins)
            exc_spec = any(t_exc(v) and t_kind(v) == 1 for v, _, _ in ins)
            if is_T(s):
                k, d = t_kind(s), t_dir(s)
                if (k == 0 and exc_spec) or (k == 1 and exc_ord):
                    out[r, c] = U
                    continue
                e = False
                for v, dr, dc in ins:  # (dr, dc) = offset to that neighbour
                    if t_kind(v) == k and t_exc(v) and DIRS[d] != (dr, dc):
                        e = True
                for d2, (dr, dc) in enumerate(DIRS):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W and is_C(int(g[rr, cc])) and d != d2:
                        if (int(g[rr, cc]) - C0) >> 1:  # confluent outputting now
                            e = True
                out[r, c] = T(k, d, int(e))
            elif is_C(s):
                if exc_spec:
                    out[r, c] = U
                    continue
                ords = [v for v, _, _ in ins if t_kind(v) == 0]
                nxt_in = int(bool(ords) and all(t_exc(v) for v in ords))
                out[r, c] = C((s - C0) & 1, nxt_in)
            elif s == U:
                if exc_ord or exc_spec:
                    out[r, c] = S("S")
            elif is_S(s):
                bit = int(exc_ord or exc_spec)
                out[r, c] = TREE[S_NAMES[s - S_BASE]][bit]
    return out


def construction_arm(rows: int, cols: int, r0: int = 3, c0: int = 1, arm: int = 2) -> np.ndarray:
    """A tiny self-extending machine: a clockwise signal loop of 5 ordinary
    wires + 1 confluent cell (period 7) holding one pulse, i.e. the code
    1 0 0 0 0 0 0. The confluent copies every pulse into an east-pointing
    construction arm; at the arm's tip each code sensitizes the blank cell
    (1) and the four 0s that follow turn it into a new east-pointing wire,
    so the arm builds itself one cell per period."""
    g = np.zeros((rows, cols), dtype=np.int16)
    g[r0, c0] = T(0, E_, 1)          # the stored pulse
    g[r0, c0 + 1] = T(0, E_)
    g[r0, c0 + 2] = C()
    g[r0 + 1, c0 + 2] = T(0, W_)
    g[r0 + 1, c0 + 1] = T(0, W_)
    g[r0 + 1, c0] = T(0, N_)
    for k in range(arm):
        g[r0, c0 + 3 + k] = T(0, E_)
    return g


STAIR_PROGRAM = (1, 0, 0, 0, 0, 1, 0, 1, 0)   # 10000 = build '->' , 1010 = build 'v'


def construction_stair(rows: int, cols: int, r0: int = 1, c0: int = 1, arm: int = 2) -> np.ndarray:
    """A self-extending machine that turns: a period-9 signal loop (3 wires,
    a confluent cell C, 4 wires back) holds the tape 10000 1010. C copies it
    into the construction arm; at the tip the two codes alternately build an
    east- and a south-pointing wire, so the arm grows down a diagonal
    staircase, one cell per 9 steps.
    Loop cells in travel order after C: bottom row right->left, top row left->right."""
    a = 3
    g = np.zeros((rows, cols), dtype=np.int16)
    top = [(r0, c0 + j) for j in range(a)]
    bottom = [(r0 + 1, c0 + a - j) for j in range(a + 1)]
    for r, c in top:
        g[r, c] = T(0, E_)
    g[r0, c0 + a] = C()
    for k, (r, c) in enumerate(bottom):
        g[r, c] = T(0, N_ if k == a else W_)
    order = bottom + top
    bits = STAIR_PROGRAM[3:] + STAIR_PROGRAM[:3]   # phase so the first code reaches the tip intact
    for (r, c), b in zip(order, bits):
        if b:
            g[r, c] += 1
    for k in range(arm):
        g[r0, c0 + a + 1 + k] = T(0, E_)
    return g


def arm_tip(g: np.ndarray, r: int, c: int):
    """Follow wires from (r, c) along their arrows; return the first non-wire
    cell reached (the construction site) and the path length."""
    n = 0
    H, W = g.shape
    while 0 <= r < H and 0 <= c < W and is_T(int(g[r, c])):
        dr, dc = DIRS[t_dir(int(g[r, c]))]
        r, c, n = r + dr, c + dc, n + 1
        if n > H * W:
            break
    return (r, c), n
