import numpy as np

from sim.langton_loops import PERIOD, initial_loop, run


def _components(mask):
    """4-connected component sizes (tiny flood fill, no scipy needed)."""
    seen = np.zeros_like(mask, dtype=bool)
    sizes = []
    for r, c in zip(*np.nonzero(mask)):
        if seen[r, c]:
            continue
        stack, n = [(r, c)], 0
        seen[r, c] = True
        while stack:
            y, x = stack.pop()
            n += 1
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                yy, xx = y + dy, x + dx
                if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not seen[yy, xx]:
                    seen[yy, xx] = True
                    stack.append((yy, xx))
        sizes.append(n)
    return sorted(sizes)


def _grid(size=80):
    g = np.zeros((size, size), dtype=np.uint8)
    L = initial_loop()
    g[35:45, 30:45] = L
    return g


def test_initial_loop_is_langtons_86_cells():
    L = initial_loop()
    assert L.shape == (10, 15)
    assert int((L > 0).sum()) == 86
    assert set(np.unique(L)) <= set(range(8))


def test_one_period_gives_mother_and_daughter():
    g = run(_grid(), PERIOD)
    sizes = _components(g > 0)
    assert len(sizes) == 2
    assert all(80 <= s <= 95 for s in sizes)


def test_colony_keeps_growing():
    g = _grid(200)
    g1 = run(g, 400)
    g2 = run(g1, 400)
    n0, n1, n2 = (int((x > 0).sum()) for x in (g, g1, g2))
    assert n0 < n1 < n2
    assert len(_components(g2 > 0)) > len(_components(g1 > 0))
