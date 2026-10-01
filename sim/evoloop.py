"""Evoloop (H. Sayama, 1999): self-reproducing loops that also *evolve*.

9 states, von Neumann neighbourhood. Like Langton's loops (states 0-7 play
the same roles: 1 core, 2 sheath, 3-7 genes), plus state 8, a 'dissolving'
state: structures that can no longer reproduce are dissolved and their
space is freed. Loops collide, mutate through those collisions and are
selected; the population evolves towards smaller, faster loops.

The transition function is read from Golly's Evoloop.rule (assets/rules/),
whose @TREE section encodes the complete 9^5 table (Golly's rule-tree
format; von Neumann variables in the order N, W, E, S, C).
"""
from functools import lru_cache

import numpy as np

from .langton_loops import RULE_DIR

N_STATES = 9


def _index(c, n, e, s, w):
    return (((c * 9 + n) * 9 + e) * 9 + s) * 9 + w


def _read_section(name: str) -> list:
    lines = (RULE_DIR / "Evoloop.rule").read_text().splitlines()
    start = lines.index(name) + 1
    out = []
    for line in lines[start:]:
        if line.startswith("@"):
            break
        out.append(line.strip())
    return out


@lru_cache(maxsize=1)
def rule_table() -> np.ndarray:
    """(9**5,) uint8 lookup: index(C,N,E,S,W) -> new C, from the @TREE."""
    nodes = []
    for line in _read_section("@TREE"):
        if not line or "=" in line or line.startswith("#"):
            continue
        parts = [int(v) for v in line.split()]
        nodes.append((parts[0], parts[1:]))
    root = len(nodes) - 1
    lut = np.zeros(9 ** 5, dtype=np.uint8)
    for c in range(9):
        for n in range(9):
            for e in range(9):
                for s in range(9):
                    for w in range(9):
                        node = root
                        for v in (n, w, e, s, c):  # Golly tree order for vonNeumann
                            depth, kids = nodes[node]
                            node = kids[v]
                        lut[_index(c, n, e, s, w)] = node  # depth-1 nodes hold states
    return lut


def table_transitions():
    """Explicit (C, N, E, S, W, C') transitions from the @TABLE section, with
    variables expanded (a variable name binds the same value everywhere in
    one transition) and rotate4 symmetry applied. Used to cross-check the
    tree decoding; yields in file order."""
    import itertools
    variables = {}
    rows = []
    for line in _read_section("@TABLE"):
        if not line or line.startswith("#") or ":" in line:
            continue
        if line.startswith("var "):
            name, vals = line[4:].split("=")
            variables[name.strip()] = [int(v) for v in vals.strip("{} ").split(",")]
            continue
        rows.append([t.strip() for t in line.split(",")])
    for fields in rows:
        names = sorted({f for f in fields if not f.isdigit()})
        for combo in itertools.product(*[variables[n] for n in names]):
            bind = dict(zip(names, combo))
            c, n, e, s, w, new = (int(f) if f.isdigit() else bind[f] for f in fields)
            nb = [n, e, s, w]
            for r in range(4):
                rn, re_, rs, rw = nb[r:] + nb[:r]
                yield c, rn, re_, rs, rw, new


def initial_loop() -> np.ndarray:
    """(17, 17) uint8: Sayama's ancestor loop (Golly's Evoloop.rle)."""
    from .langton_loops import parse_rle
    return parse_rle((RULE_DIR / "Evoloop.rle").read_text())


def step(grid: np.ndarray) -> np.ndarray:
    """One synchronous update on a toroidal grid of states 0..8."""
    g = grid.astype(np.int64)
    n = np.roll(g, 1, axis=0)
    s = np.roll(g, -1, axis=0)
    w = np.roll(g, 1, axis=1)
    e = np.roll(g, -1, axis=1)
    return rule_table()[(((g * 9 + n) * 9 + e) * 9 + s) * 9 + w]


# Sayama's colours (loops.java), softened for the dark film background.
PALETTE = np.array([
    (0, 0, 0),         # 0 background
    (70, 110, 255),    # 1 core
    (255, 70, 70),     # 2 sheath
    (80, 230, 110),    # 3
    (255, 230, 70),    # 4
    (255, 80, 230),    # 5
    (240, 240, 240),   # 6
    (70, 235, 235),    # 7
    (255, 150, 40),    # 8 dissolving
], dtype=np.uint8)
