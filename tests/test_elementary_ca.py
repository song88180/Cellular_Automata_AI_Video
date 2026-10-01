from sim.elementary_ca import generate, rule_table, step
import numpy as np


def test_rule90_matches_binomial_parity_lucas_theorem():
    """Rule 90 from a single seed cell reproduces the Sierpinski triangle:
    cell (n, center+d) is active iff C(n, k) is odd where k=(n+d)/2, and by
    Lucas' theorem C(n, k) mod 2 == 1 iff (n & k) == k. This checks the
    implementation against pure number theory, independent of the step()
    logic itself."""
    width = 41
    steps = 16
    center = width // 2
    rows = generate(width, rule=90, steps=steps, seed="single")

    for n in range(steps + 1):
        for x in range(width):
            d = x - center
            if (d + n) % 2 != 0:
                expected = 0
            else:
                k = (d + n) // 2
                expected = 1 if (0 <= k <= n) and (n & k) == k else 0
            assert rows[n, x] == expected, f"n={n} x={x} d={d}"


def test_rule_table_bit_order():
    table = rule_table(90)  # 0b01011010
    assert list(table) == [0, 1, 0, 1, 1, 0, 1, 0]


def test_step_is_deterministic_and_wraps_around():
    row = np.array([1, 0, 0, 0, 1], dtype=np.uint8)
    table = rule_table(30)
    out1 = step(row, table)
    out2 = step(row, table)
    assert np.array_equal(out1, out2)
    assert out1.shape == row.shape
