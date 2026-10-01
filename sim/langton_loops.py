"""Langton's self-reproducing loops (C. G. Langton, Physica D 10, 1984).

8 states, von Neumann neighbourhood (C, N, E, S, W), rotationally symmetric
transition table of 219 rules, taken from Golly's Langtons-Loops.rule
(assets/rules/). Transitions missing from the table leave the cell
unchanged (Golly's convention).

States (Langton's roles): 0 background, 1 core (the 'tube' contents),
2 sheath, 3-7 signals; 7+0 = 'extend arm' gene, 4+0 = 'turn left' gene
(signals travel as (gene, 0) pairs through the core).
"""
from functools import lru_cache
from pathlib import Path

import numpy as np

RULE_DIR = Path(__file__).resolve().parent.parent / "assets" / "rules"
N_STATES = 8
PERIOD = 151  # steps for a loop to produce a complete daughter


def _index(c, n, e, s, w):
    return (((c * 8 + n) * 8 + e) * 8 + s) * 8 + w


@lru_cache(maxsize=1)
def rule_table() -> np.ndarray:
    """(8**5,) uint8 lookup: index(C,N,E,S,W) -> new C."""
    lut = np.arange(8 ** 5, dtype=np.int64)
    lut = (lut // 8 ** 4).astype(np.uint8)   # default: unchanged
    lines = (RULE_DIR / "Langtons-Loops.rule").read_text().splitlines()
    start = lines.index("@TABLE")
    for line in lines[start + 1:]:
        line = line.strip()
        if line.startswith("@"):
            break
        if len(line) == 6 and line.isdigit():
            c, n, e, s, w, new = (int(ch) for ch in line)
            nb = [n, e, s, w]
            for r in range(4):  # rotate4 symmetry
                rn, re_, rs, rw = nb[r:] + nb[:r]
                lut[_index(c, rn, re_, rs, rw)] = new
    return lut


def initial_loop() -> np.ndarray:
    """(10, 15) uint8: Langton's original 86-cell loop (from Golly's .rle)."""
    return parse_rle((RULE_DIR / "Langtons-Loops.rle").read_text())


def parse_rle(rle: str) -> np.ndarray:
    """Golly multi-state RLE ('.' = 0, 'A'.. = 1..) -> uint8 array."""
    text = rle.splitlines()
    body = "".join(l.strip() for l in text if l and not l.startswith("#") and not l.startswith("x"))
    rows, row, num = [], [], ""
    for ch in body:
        if ch.isdigit():
            num += ch
            continue
        k = int(num) if num else 1
        num = ""
        if ch == "$" or ch == "!":
            rows.append(row)
            for _ in range(k - 1):
                rows.append([])
            row = []
            if ch == "!":
                break
        else:
            v = 0 if ch == "." else ord(ch) - ord("A") + 1
            row.extend([v] * k)
    w = max(len(r) for r in rows)
    out = np.zeros((len(rows), w), dtype=np.uint8)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def step(grid: np.ndarray) -> np.ndarray:
    """One synchronous update on a toroidal grid of states 0..7."""
    g = grid.astype(np.int64)
    n = np.roll(g, 1, axis=0)
    s = np.roll(g, -1, axis=0)
    w = np.roll(g, 1, axis=1)
    e = np.roll(g, -1, axis=1)
    return rule_table()[(((g * 8 + n) * 8 + e) * 8 + s) * 8 + w]


def run(grid: np.ndarray, steps: int) -> np.ndarray:
    for _ in range(steps):
        grid = step(grid)
    return grid


# Display palette (Golly's / Sayama's colours, slightly softened for a dark
# background). Index = state.
PALETTE = np.array([
    (0, 0, 0),         # 0 background
    (70, 110, 255),    # 1 core
    (255, 70, 70),     # 2 sheath
    (80, 230, 110),    # 3
    (255, 230, 70),    # 4 'turn' gene
    (255, 80, 230),    # 5
    (240, 240, 240),   # 6
    (70, 235, 235),    # 7 'extend' gene
], dtype=np.uint8)
