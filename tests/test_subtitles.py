from engine.subtitles import build_srt, center_line, display_width, format_ts, parse_ts


def test_format_ts_carries_rounded_milliseconds():
    assert format_ts(22.9996) == "00:00:23,000"
    assert format_ts(59.9999) == "00:01:00,000"
    assert format_ts(3723.5) == "01:02:03,500"


def test_format_parse_roundtrip():
    for t in (0.0, 0.083, 12.345, 179.999):
        assert abs(parse_ts(format_ts(t)) - t) < 0.0011


def test_build_srt_keeps_every_cue():
    cues = [(i / 12, (i + 1) / 12, f"zh{i}", f"en{i}") for i in range(0, 300, 7)]
    text = build_srt(cues)
    assert text.count("-->") == len(cues)
    assert text.startswith("1\n00:00:00,000")


def test_cjk_is_double_width_and_centered():
    assert display_width("元胞ab") == 6
    line = center_line("元胞", 10)
    assert display_width(line) == 10 and line.strip() == "元胞"
