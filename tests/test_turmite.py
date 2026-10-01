import numpy as np

from sim.turmite import Ant, run, turmite, vant


def test_langton_first_steps_match_hand_trace():
    """Langton's ant, heading up on an all-white grid (heading 0=up, R=+3).
    Step 1: white -> turn right (face right), paint black, move right.
    Step 2: white -> turn right (face down), paint black, move down.
    Step 3: white -> turn right (face left), paint black, move left."""
    g = np.zeros((9, 9), dtype=int)
    ant = Ant(4, 4, heading=0)
    run(g, ant, vant("RL"), 3)
    assert (ant.r, ant.c, ant.heading) == (5, 4, 1)
    assert g[4, 4] == 1 and g[4, 5] == 1 and g[5, 5] == 1
    assert g.sum() == 3


def test_langton_builds_highway_period_104():
    """Well-known result: after ~10k steps Langton's ant enters a 104-step
    cycle that translates it 2 cells diagonally (the 'highway')."""
    g = np.zeros((300, 300), dtype=int)
    ant = Ant(150, 150, heading=0)
    run(g, ant, vant("RL"), 12000)
    r0, c0 = ant.r, ant.c
    run(g, ant, vant("RL"), 104)
    assert abs(ant.r - r0) == 2 and abs(ant.c - c0) == 2


def test_turmite_parser_and_halt():
    rule = turmite(["Lb1 Fa2", "Rb1 ---"])
    assert rule.turn.shape == (2, 2)
    g = np.zeros((5, 5), dtype=int)
    g[2, 2] = 1  # start on color b, state 1 -> Fa2
    ant = Ant(2, 2, heading=0)
    run(g, ant, rule, 1)
    assert ant.state == 1 and g[2, 2] == 0 and (ant.r, ant.c) == (1, 2)
    # now at state 2 on color a -> Rb1; then state 1 on color... just ensure halting works:
    g2 = np.ones((5, 5), dtype=int)
    ant2 = Ant(2, 2, heading=0, state=1)
    assert run(g2, ant2, rule, 10) == 0 and ant2.halted
