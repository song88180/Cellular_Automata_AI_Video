"""Generic turmite engine (2D Turing machine on a grid). Langton's ant is the
1-state / 2-color special case.

Semantics follow Greg Turk's reference implementation
(https://github.com/gturk1/turmites, turmites/rules.pde) exactly:
  heading: 0=up, 1=left, 2=down, 3=right
  turn:    F=+0, L=+1, B=+2, R=+3   (heading = (heading + turn) % 4)
  per step: read cell color -> look up (turn, new color, new state) for the
  current state -> turn -> write color -> change state -> move forward one cell.
Rule strings use Turk's syntax too (see `vant` and `turmite`).
"""
from dataclasses import dataclass
from typing import List

import numpy as np

TURNS = {"F": 0, "L": 1, "B": 2, "R": 3}
HALT = -1
_DR = (-1, 0, 1, 0)  # indexed by heading
_DC = (0, -1, 0, 1)


@dataclass
class TurmiteRule:
    turn: np.ndarray       # (n_states, n_colors) int, HALT = undefined
    new_color: np.ndarray  # (n_states, n_colors) int
    new_state: np.ndarray  # (n_states, n_colors) int

    @property
    def n_colors(self) -> int:
        return self.turn.shape[1]


def vant(rule: str, cycle_offset: int = 0) -> TurmiteRule:
    """Single-state 'vant' (Langton's ant generalization): the i-th letter is the
    turn on color i, and color i becomes i+1 (the last color becomes
    `cycle_offset`). E.g. vant("RL") is Langton's original ant."""
    n = len(rule)
    turn = np.array([[TURNS[ch.upper()] for ch in rule]])
    new_color = np.array([[i + 1 if i < n - 1 else cycle_offset for i in range(n)]])
    return TurmiteRule(turn=turn, new_color=new_color, new_state=np.zeros((1, n), dtype=int))


def turmite(states: List[str]) -> TurmiteRule:
    """Multi-state turmite in Turk's notation: one string per state (states are
    1-based), one space-separated word per current cell color a, b, c, ...;
    each word = turn letter + new color letter + new state number, e.g. "Lb1".
    "---" marks an undefined (halting) entry."""
    rows = [s.split() for s in states]
    n_states, n_colors = len(rows), max(len(r) for r in rows)
    turn = np.full((n_states, n_colors), HALT, dtype=int)
    new_color = np.zeros((n_states, n_colors), dtype=int)
    new_state = np.zeros((n_states, n_colors), dtype=int)
    for s, words in enumerate(rows):
        for c, w in enumerate(words):
            if w.startswith("-"):
                continue
            turn[s, c] = TURNS[w[0].upper()]
            new_color[s, c] = ord(w[1].lower()) - ord("a")
            new_state[s, c] = int(w[2:]) - 1
    return TurmiteRule(turn=turn, new_color=new_color, new_state=new_state)


class Ant:
    def __init__(self, r: int, c: int, heading: int = 0, state: int = 0):
        self.r, self.c, self.heading, self.state = r, c, heading, state
        self.halted = False


def run(grid: np.ndarray, ant: Ant, rule: TurmiteRule, steps: int) -> int:
    """Advance `ant` up to `steps` steps in place on a toroidal `grid` of color
    indices. Returns the number of steps actually taken (fewer if it halts)."""
    turn_t, color_t, state_t = rule.turn, rule.new_color, rule.new_state
    h, w = grid.shape
    r, c, hd, s = ant.r, ant.c, ant.heading, ant.state
    taken = 0
    for _ in range(steps):
        color = grid[r, c]
        t = turn_t[s, color]
        if t == HALT:
            ant.halted = True
            break
        hd = (hd + t) & 3
        grid[r, c] = color_t[s, color]
        s = state_t[s, color]
        r = (r + _DR[hd]) % h
        c = (c + _DC[hd]) % w
        taken += 1
    ant.r, ant.c, ant.heading, ant.state = r, c, hd, s
    return taken
