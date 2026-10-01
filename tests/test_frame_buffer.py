import numpy as np

from engine.frame import Frame, FrameBuffer


def test_diff_detects_changes_when_frame_object_is_mutated_in_place():
    """Regression test: chapters may mutate the same Frame object across
    yields (e.g. incrementally revealing rows) for efficiency. FrameBuffer
    must snapshot arrays, not alias them, or a second diff() against the
    same (now-mutated) object would see no change at all."""
    buf = FrameBuffer(rows=3, cols=4)
    frame = Frame.blank(3, 4)

    frame.active[0, 0] = True
    frame.chars[0, 0] = "@"
    frame.colors[0, 0] = (255, 0, 0)
    changed1 = buf.diff(frame)
    assert (0, 0, "@", (255, 0, 0), None) in changed1

    # Mutate the SAME frame object in place (no new allocation).
    frame.active[1, 1] = True
    frame.chars[1, 1] = "#"
    frame.colors[1, 1] = (0, 255, 0)
    changed2 = buf.diff(frame)

    assert (1, 1, "#", (0, 255, 0), None) in changed2
    assert (0, 0, "@", (255, 0, 0), None) not in changed2  # unchanged cell shouldn't be re-emitted


def test_diff_first_call_returns_all_active_and_no_inactive():
    buf = FrameBuffer(rows=2, cols=2)
    frame = Frame.blank(2, 2)
    frame.active[0, 0] = True
    frame.chars[0, 0] = "x"
    frame.colors[0, 0] = (1, 2, 3)
    changed = buf.diff(frame)
    assert len(changed) == 4  # every cell reported on first frame
    assert (0, 0, "x", (1, 2, 3), None) in changed
    assert (0, 1, " ", None, None) in changed


def test_from_pixels_half_blocks():
    filled = np.array([[1, 1, 0, 0],
                       [1, 0, 1, 0]], dtype=bool)  # 2 pixel rows -> 1 char row
    rgb = np.zeros((2, 4, 3), dtype=np.uint8)
    rgb[0] = (10, 10, 10)
    rgb[1] = (20, 20, 20)
    f = Frame.from_pixels(filled, rgb, rows=1, cols=4)
    assert list(f.chars[0]) == ["▀", "▀", "▄", " "]
    assert list(f.bg_active[0]) == [True, False, False, False]
    assert tuple(f.colors[0, 0]) == (10, 10, 10) and tuple(f.bg[0, 0]) == (20, 20, 20)
    assert tuple(f.colors[0, 2]) == (20, 20, 20)  # bottom-only pixel drawn as fg of '▄'
