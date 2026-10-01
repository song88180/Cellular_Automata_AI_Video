"""Wolfram elementary (1D, 2-state, 3-neighbor) cellular automata.

Neighborhood (left, center, right) is packed into a 3-bit index
`idx = left*4 + center*2 + right` (idx=7 <-> pattern 111). `rule_table(rule)`
returns an 8-entry lookup table where table[idx] is bit `idx` of the rule
number, per Wolfram's standard numbering convention.
"""
from typing import Optional

import numpy as np


def rule_table(rule: int) -> np.ndarray:
    if not 0 <= rule <= 255:
        raise ValueError(f"rule must be in [0, 255], got {rule}")
    return np.array([(rule >> i) & 1 for i in range(8)], dtype=np.uint8)


def step(row: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Advance one generation with circular (wrap-around) boundaries."""
    left = np.roll(row, 1)
    right = np.roll(row, -1)
    idx = (left << 2) | (row << 1) | right
    return table[idx]


def generate(
    width: int, rule: int, steps: int, seed: str = "single", seed_col: Optional[int] = None
) -> np.ndarray:
    """Return an array of shape (steps + 1, width) of 0/1 rows, row 0 = the
    initial condition. seed='single' = one live cell at `seed_col` (default:
    centered -- the classic Wolfram figure); seed='random' = uniform random init."""
    table = rule_table(rule)
    row = np.zeros(width, dtype=np.uint8)
    if seed == "single":
        row[width // 2 if seed_col is None else seed_col] = 1
    elif seed == "random":
        rng = np.random.default_rng(0)
        row = (rng.random(width) < 0.5).astype(np.uint8)
    else:
        raise ValueError(f"unknown seed mode {seed!r}")

    rows = np.empty((steps + 1, width), dtype=np.uint8)
    rows[0] = row
    for i in range(1, steps + 1):
        row = step(row, table)
        rows[i] = row
    return rows
