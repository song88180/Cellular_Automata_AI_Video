"""Conway's Game of Life (B3/S23) on a toroidal grid, plus a small pattern library.

Patterns use the LifeWiki plaintext convention: 'O' = alive, '.' = dead.
"""
from typing import Dict

import numpy as np

_NEIGHBOR_OFFSETS = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]


def neighbor_count(grid: np.ndarray) -> np.ndarray:
    g = grid.astype(np.uint8)
    return sum(np.roll(np.roll(g, dr, axis=0), dc, axis=1) for dr, dc in _NEIGHBOR_OFFSETS)


def step(grid: np.ndarray) -> np.ndarray:
    """One generation: born with exactly 3 neighbors, survive with 2 or 3."""
    n = neighbor_count(grid)
    return (n == 3) | (grid & (n == 2))


def parse(pattern: str) -> np.ndarray:
    lines = pattern.strip("\n").splitlines()
    width = max(len(line) for line in lines)
    arr = np.zeros((len(lines), width), dtype=bool)
    for r, line in enumerate(lines):
        for c, ch in enumerate(line):
            arr[r, c] = ch == "O"
    return arr


def place(grid: np.ndarray, pattern: np.ndarray, r: int, c: int) -> None:
    h, w = pattern.shape
    grid[r: r + h, c: c + w] |= pattern


PATTERNS: Dict[str, str] = {
    "block": """
OO
OO
""",
    "beehive": """
.OO.
O..O
.OO.
""",
    "blinker": """
OOO
""",
    "toad": """
.OOO
OOO.
""",
    "pulsar": """
..OOO...OOO..
.............
O....O.O....O
O....O.O....O
O....O.O....O
..OOO...OOO..
.............
..OOO...OOO..
O....O.O....O
O....O.O....O
O....O.O....O
.............
..OOO...OOO..
""",
    "loaf": """
.OO.
O..O
.O.O
..O.
""",
    "boat": """
OO.
O.O
.O.
""",
    "beacon": """
OO..
OO..
..OO
..OO
""",
    "pentadecathlon": """
..O....O..
OO.OOOO.OO
..O....O..
""",
    "lwss": """
.O..O
O....
O...O
OOOO.
""",
    "glider": """
.O.
..O
OOO
""",
    "gosper_glider_gun": """
........................O...........
......................O.O...........
............OO......OO............OO
...........O...O....OO............OO
OO........O.....O...OO..............
OO........O...O.OO....O.O...........
..........O.....O.......O...........
...........O...O....................
............OO......................
""",
}


def pattern(name: str) -> np.ndarray:
    return parse(PATTERNS[name])
