# tests/test_font_matcher.py
"""Render-and-compare matcher, verified on the Windows system font set
(15/15 across clean / JPEG40+blur / no-cv2 regimes, incl. Impact).
Heavy-OUTLINE captions remain known-fragile (research: <0.02 margins on
near-twins) - these tests pin the verified-good regimes only.

CRITICAL SYMMETRY RULE (panel-review finding): queries here are RAW
canvases from render_sample() - exactly what a wild video crop looks
like - so that query and candidates each pass through prep_crop exactly
once inside identify(). Feeding an already-ink-normalized image as the
query double-crops it and deterministically breaks heavy fonts (Impact
self-score collapses to 0.195). Do not "simplify" the tests to use the
candidate pipeline as the query."""
from pathlib import Path

import pytest
from PIL import Image

from magicat.modules.captions import font_matcher
from magicat.modules.captions.font_matcher import FontMatcher, render_sample

WIN_FONTS = Path("C:/Windows/Fonts")
FIVE = ["arial", "ariblk", "impact", "comic", "bahnschrift"]

pytestmark = pytest.mark.skipif(
    not all((WIN_FONTS / f"{k}.ttf").is_file() for k in FIVE),
    reason="system test fonts unavailable")


@pytest.fixture(scope="module")
def matcher() -> FontMatcher:
    m = FontMatcher.from_dirs([str(WIN_FONTS)])
    # narrow to the 5-font benchmark set for determinism
    m.fonts = {k: m.fonts[k] for k in FIVE}
    return m


def test_from_dirs_finds_fonts_and_env_extends(tmp_path, monkeypatch):
    monkeypatch.setenv("MAGICAT_FONT_DIRS", str(tmp_path))
    (tmp_path / "myfont.ttf").write_bytes(b"not a real font")
    m = FontMatcher.from_dirs([str(WIN_FONTS)])
    assert "arial" in m.fonts
    assert "myfont" in m.fonts            # env dir merged in


def test_clean_render_confusion_matrix(matcher):
    # every font's own (raw, wild-crop-like) render must win vs the others
    for key in FIVE:
        crop = render_sample(matcher.fonts[key], "HELLO WORLD")
        result = matcher.identify(crop, "HELLO WORLD")
        assert result.font_key == key, \
            f"{key} misidentified as {result.font_key}"
        assert result.score > 0.99        # self-match is ~1.0 by symmetry


def test_degraded_crop_still_wins(matcher, tmp_path):
    # JPEG q40 + slight blur (realistic video-crop degradation)
    from PIL import ImageFilter
    for key in FIVE:
        crop = render_sample(matcher.fonts[key], "HELLO WORLD")
        crop = crop.filter(ImageFilter.GaussianBlur(0.6))
        p = tmp_path / f"degraded_{key}.jpg"
        crop.convert("RGB").save(p, quality=40)
        with Image.open(p) as degraded:
            result = matcher.identify(degraded, "HELLO WORLD")
        assert result.font_key == key


def test_result_shape_and_confidence(matcher):
    crop = render_sample(matcher.fonts["arial"], "HELLO WORLD")
    r = matcher.identify(crop, "HELLO WORLD")
    assert r.ranked[0][0] == r.font_key
    assert 0.0 <= r.score <= 1.0
    assert r.margin >= 0.0
    assert isinstance(r.confident, bool)
    assert len(r.ranked) == len(FIVE)


def test_no_cv2_fallback_still_works_clean(matcher, monkeypatch):
    monkeypatch.setattr(font_matcher, "_HAVE_CV2", False)
    for key in FIVE:
        crop = render_sample(matcher.fonts[key], "HELLO WORLD")
        result = matcher.identify(crop, "HELLO WORLD")
        assert result.font_key == key


def test_no_fonts_raises():
    with pytest.raises(RuntimeError, match="no candidate fonts"):
        FontMatcher(fonts={}).identify(
            Image.new("L", (100, 40), 0), "X")
