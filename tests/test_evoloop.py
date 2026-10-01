import numpy as np

from sim import evoloop as ev


def test_tree_decoding_matches_the_explicit_table():
    """Two independent readings of Golly's Evoloop.rule must agree: the
    @TREE (neighbour order N, W, E, S, C) and the @TABLE (C, N, E, S, W with
    variables and rotate4 expanded; first match wins)."""
    lut = ev.rule_table()
    seen = set()
    for c, n, e, s, w, new in ev.table_transitions():
        k = ev._index(c, n, e, s, w)
        if k in seen:
            continue
        seen.add(k)
        assert lut[k] == new
    assert len(seen) > 50000


def test_ancestor_loop_reproduces():
    g = np.zeros((120, 120), dtype=np.uint8)
    g[52:69, 52:69] = ev.initial_loop()
    n0 = int((g > 0).sum())
    for _ in range(1000):
        g = ev.step(g)
    assert int((g > 0).sum()) > 3 * n0     # several loops by step 1000
