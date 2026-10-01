import numpy as np

from sim.game_of_life import pattern, place, step


def _run(grid, n):
    for _ in range(n):
        grid = step(grid)
    return grid


def _empty(h=20, w=20):
    return np.zeros((h, w), dtype=bool)


def test_glider_translates_by_one_diagonal_every_4_generations():
    g = _empty()
    place(g, pattern("glider"), 5, 5)
    after = _run(g, 4)
    assert np.array_equal(after, np.roll(np.roll(g, 1, axis=0), 1, axis=1))


def test_blinker_has_period_2():
    g = _empty()
    place(g, pattern("blinker"), 10, 9)
    one = _run(g, 1)
    assert not np.array_equal(one, g)
    assert np.array_equal(_run(g, 2), g)


def test_pulsar_has_period_3():
    g = _empty(30, 30)
    place(g, pattern("pulsar"), 8, 8)
    assert not np.array_equal(_run(g, 1), g)
    assert not np.array_equal(_run(g, 2), g)
    assert np.array_equal(_run(g, 3), g)


def test_still_lifes_do_not_change():
    for name in ("block", "beehive"):
        g = _empty()
        place(g, pattern(name), 8, 8)
        assert np.array_equal(step(g), g), name


def test_gosper_gun_emits_gliders_and_keeps_firing():
    """Gun is period 30 and adds one 5-cell glider per period; on a big
    enough grid nothing wraps, so population grows by 5 every 30 gens."""
    g = np.zeros((120, 160), dtype=bool)
    place(g, pattern("gosper_glider_gun"), 10, 10)
    g = _run(g, 60)
    p60 = int(g.sum())
    g = _run(g, 30)
    p90 = int(g.sum())
    g = _run(g, 30)
    p120 = int(g.sum())
    assert p90 - p60 == 5
    assert p120 - p90 == 5


def test_new_still_lifes_and_oscillator_periods():
    for name in ("loaf", "boat"):
        g = _empty()
        place(g, pattern(name), 8, 8)
        assert np.array_equal(step(g), g), name
    for name, period, size in (("beacon", 2, 20), ("pentadecathlon", 15, 30)):
        g = _empty(size, size)
        place(g, pattern(name), size // 2 - 2, size // 2 - 5)
        for k in range(1, period):
            assert not np.array_equal(_run(g, k), g), (name, k)
        assert np.array_equal(_run(g, period), g), name


def test_lwss_moves_two_cells_every_four_generations():
    g = _empty(20, 30)
    place(g, pattern("lwss"), 8, 5)
    after = _run(g, 4)
    assert after.sum() == g.sum()
    assert np.array_equal(after, np.roll(g, -2, axis=1)) or np.array_equal(after, np.roll(g, 2, axis=1))


def test_counting_grid_demo_parts_never_interact():
    """Chapter 3's counting-grid demo: glider, blinker and T-tetromino must
    evolve exactly as if each were alone."""
    from chapters.chapter3_game_of_life import CG_PAD, _count_grid_start
    g = _count_grid_start()
    lab = np.zeros(g.shape, int)
    p = CG_PAD
    lab[p: p + 4, p: p + 4] = 1          # glider box
    lab[p: p + 3, p + 5: p + 10] = 2     # blinker box
    lab[p + 4: p + 8, p + 11: p + 16] = 3
    parts = [g & (lab == k) for k in (1, 2, 3)]
    assert sum(int(x.sum()) for x in parts) == int(g.sum())
    for _ in range(20):
        parts = [step(x) for x in parts]
        g = step(g)
        assert np.array_equal(np.logical_or.reduce(parts), g)
