"""Engine: timeline packing/transitions/caching, rasteriser, soundtrack."""
import numpy as np

from engine.canvas import BG, Canvas
from engine.frame import WIDE_CONT, Frame, Scene
from engine.fx import blend_reveal, threshold_map, transition
from engine.timeline import TICK_FPS, Film, _pack, _unpack, build_film


def _frame(text="元胞 ab", fill=(200, 10, 10)):
    cv = Canvas(4, 16)
    cv.rect(0, 0, 3, 5, fill)
    f = cv.to_frame("中文", "english")
    f.put_text(3, 2, text, (1, 2, 3), bg=(9, 9, 9))
    return f


def test_put_text_marks_wide_continuation_cells():
    f = _frame()
    assert f.chars[3, 2] == "元" and f.chars[3, 3] == WIDE_CONT
    assert f.chars[3, 4] == "胞" and f.chars[3, 5] == WIDE_CONT
    assert f.chars[3, 7] == "a"


def test_pack_roundtrip_preserves_everything():
    f = _frame()
    g = _unpack(_pack(f), 4, 16)
    for name in ("chars", "colors", "active", "bg", "bg_active"):
        assert np.array_equal(getattr(f, name), getattr(g, name)), name


def test_build_film_tick_counts_transitions_and_dedup():
    def gen(n, fill):
        for _ in range(n):
            yield _frame(fill=fill)

    scenes = [
        Scene("a", lambda: gen(6, (255, 0, 0)), fps=12, transition="cut"),
        Scene("b", lambda: gen(12, (0, 255, 0)), fps=24, transition="dissolve", transition_ticks=6),
    ]
    film = build_film(scenes, 4, 16)
    assert film.n_ticks == 6 * 2 + 12
    assert [s.start for s in film.scenes] == [0, 12] and film.scenes[1].end == 24
    # identical frames of scene a are stored once; transition ticks are unique frames
    assert film.frame_index(0) == film.frame_index(11)
    assert len(set(film.tick_frame[12:18])) == 6
    assert film.frame(23).colors[0, 0].tolist() == [0, 255, 0]
    assert (12, "whoosh", 1.0) in film.sfx
    assert film.cues()[0][2:] == ("中文", "english")


def test_film_save_load_roundtrip(tmp_path):
    film = build_film([Scene("a", lambda: iter([_frame()] * 3), fps=24, transition="cut")], 4, 16)
    p = tmp_path / "f.npz"
    film.save(p)
    back = Film.load(p)
    assert back.n_ticks == 3 and back.subs == film.subs
    assert np.array_equal(back.frame(1).chars, film.frame(1).chars)


def test_reveal_endpoints():
    a, b = _frame(fill=(255, 0, 0)), _frame(fill=(0, 0, 255))
    tm = threshold_map("wipe", 4, 16)
    assert np.array_equal(blend_reveal(a, b, 0.0, tm).colors, a.colors)
    assert np.array_equal(transition("wipe", a, b, 1.0).colors, b.colors)


def test_raster_half_blocks_are_exact_squares():
    from engine.raster import CH, CW, raster_grid
    cv = Canvas(2, 2)
    cv.rect(0, 0, 1, 1, (255, 0, 0))   # top pixel of cell (0, 0)
    cv.rect(1, 1, 1, 1, (0, 255, 0))   # bottom pixel of cell (0, 1)
    img = raster_grid(cv.to_frame())
    assert img.shape == (2 * CH, 2 * CW, 3)
    assert (img[: CH // 2, :CW] == (255, 0, 0)).all()
    assert (img[CH // 2: CH, :CW] == BG).all()
    assert (img[CH // 2: CH, CW: 2 * CW] == (0, 255, 0)).all()


def test_soundtrack_is_finite_and_right_length():
    from engine import audio
    film = build_film([Scene("a", lambda: iter([_frame()] * 48), fps=24, transition="cut", music="epic")], 4, 16)
    film.sfx += [(5, "hit", 1.0), (10, "pop", 0.3), (20, "rise", 1.0)]
    x = audio.synthesize(film)
    assert x.shape == (int(film.duration * audio.SR), 2)
    assert np.isfinite(x).all() and np.abs(x).max() <= 1.0


def test_stitch_applies_declared_entry_transition():
    from engine.timeline import stitch

    def gen(fill):
        for _ in range(12):
            yield _frame(fill=fill)

    a = build_film([Scene("a", lambda: gen((255, 0, 0)), fps=24, transition="cut")], 4, 16)
    b = build_film([Scene("b", lambda: gen((0, 0, 255)), fps=24, transition="wipe", transition_ticks=6)], 4, 16)
    assert not b.scenes[0].applied
    s = stitch([a, b])
    assert s.n_ticks == 24 and s.scenes[1].start == 12 and s.scenes[1].applied
    assert (12, "whoosh", 1.0) in s.sfx
    mid = s.frame(14)
    assert not np.array_equal(mid.colors, b.frame(2).colors)       # blended
    assert np.array_equal(s.frame(20).colors, b.frame(8).colors)   # transition over
