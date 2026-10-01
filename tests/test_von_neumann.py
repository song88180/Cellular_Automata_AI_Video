import numpy as np

from sim.von_neumann import C, S, T, U, construction_arm, is_T, step, t_dir, t_exc


def _run(g, n):
    for _ in range(n):
        g = step(g)
    return g


def test_wire_moves_signal_one_cell_per_step():
    g = np.zeros((1, 5), dtype=np.int16)
    g[0, :] = [T(0, 0, 1), T(0, 0), T(0, 0), T(0, 0), T(0, 0)]
    g = _run(g, 3)
    assert [t_exc(int(v)) for v in g[0]] == [0, 0, 0, 1, 0]


def test_construction_codes_follow_the_sensitized_tree():
    """Feed code bits into a blank cell and check every one of the 9 targets."""
    targets = {"10000": T(0, 0), "10001": T(0, 1), "1001": T(0, 2), "1010": T(0, 3),
               "1011": T(1, 0), "1100": T(1, 1), "1101": T(1, 2), "1110": T(1, 3), "1111": C(0, 0)}
    for code, want in targets.items():
        g = np.zeros((1, 2), dtype=np.int16)
        g[0, 0] = T(0, 0)            # source wire pointing east at the blank cell
        for bit in code:
            g[0, 0] = T(0, 0, int(bit))
            g = step(g)
        assert int(g[0, 1]) == want, code


def test_confluent_is_an_and_gate_with_two_step_delay():
    g = np.zeros((3, 3), dtype=np.int16)
    g[1, 0] = T(0, 0, 1)             # west input, excited
    g[0, 1] = T(0, 3, 0)             # north input, NOT excited
    g[1, 1] = C()
    g[1, 2] = T(0, 0)                # output wire
    g = _run(g, 3)
    assert t_exc(int(g[1, 2])) == 0  # AND fails
    g = np.zeros((3, 3), dtype=np.int16)
    g[1, 0] = T(0, 0, 1)
    g[0, 1] = T(0, 3, 1)
    g[1, 1] = C()
    g[1, 2] = T(0, 0)
    assert t_exc(int(_run(g, 2)[1, 2])) == 0 and t_exc(int(_run(g, 3)[1, 2])) == 1


def test_special_signal_destroys_ordinary_wire():
    g = np.zeros((1, 2), dtype=np.int16)
    g[0, 0] = T(1, 0, 1)
    g[0, 1] = T(0, 0)
    assert int(step(g)[0, 1]) == U


def test_construction_arm_builds_one_cell_every_period():
    g = construction_arm(6, 30)
    lengths = []
    for k in range(6):
        g = _run(g, 7)
        row = g[3]
        n = 0
        c = 4
        while c < 30 and is_T(int(row[c])) and t_dir(int(row[c])) == 0:
            n += 1
            c += 1
        lengths.append(n)
    assert lengths[-1] - lengths[0] == 5          # +1 wire per 7-step period
    assert all(b - a in (0, 1, 2) for a, b in zip(lengths, lengths[1:]))


def test_turning_arm_builds_a_staircase():
    from sim.von_neumann import arm_tip, construction_stair
    g = construction_stair(14, 30)
    for _ in range(9 * 12):
        g = step(g)
    (r, c), n = arm_tip(g, 1, 5)
    # ~12 cells built along a diagonal staircase: as many rows down as columns across
    assert n >= 12 and abs((r - 1) - (c - 5 - 2)) <= 2 and r >= 6
