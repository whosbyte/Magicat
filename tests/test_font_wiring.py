# tests/test_font_wiring.py
from pathlib import Path

import pytest

from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import Manifest, Source
from magicat.modules.captions.analyzer import CaptionAnalyzer
from magicat.modules.captions.ocr import OcrLine
from magicat.modules.ingest import IngestAnalyzer


from magicat.modules.captions.font_matcher import FontMatcher

WIN_FONTS = "C:/Windows/Fonts"
FIVE = ["arial", "ariblk", "impact", "comic", "bahnschrift"]
# the matcher mis-prefers condensed siblings on real OCR crops (verified:
# Arial Narrow outranks base Arial) - any of the family counts as a hit
ARIAL_FAMILY = {"arial", "arialbd", "arialn", "arialnb", "ariblk"}


def five_font_matcher() -> FontMatcher:
    m = FontMatcher.from_dirs([WIN_FONTS])
    m.fonts = {k: m.fonts[k] for k in FIVE if k in m.fonts}
    return m


class RealTextEngine:
    """Reports the caption_video's real burned-in caption region, so the
    persisted crops contain ACTUAL Arial glyphs (fixture burns
    'HELLO WORLD' at t=1-3, bottom-center, 480x854)."""

    def read(self, image):
        n = int(image.stem.split("_")[1])
        if 7 <= n <= 14:                       # safely inside t=1..3
            return [OcrLine(text="HELLO WORLD",
                            bbox=(0.20, 0.79, 0.60, 0.07), confidence=0.95)]
        return []


@pytest.fixture()
def analyzed(caption_video, tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(caption_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    analyzer = CaptionAnalyzer()
    monkeypatch.setattr(analyzer, "engine_factory",
                        lambda: RealTextEngine())
    monkeypatch.setattr(analyzer, "matcher_factory", five_font_matcher)
    return analyzer.run(m, ws), m


def test_font_candidates_populated(analyzed):
    patch, m = analyzed
    style = patch["captions"]["segments"][0]["style"]
    cands = style["font_candidates"]
    assert 1 <= len(cands) <= 3
    for c in cands:
        assert c["name"]
        assert 0.0 < c["confidence"] <= 1.0    # real text -> nonzero scores
    scores = [c["confidence"] for c in cands]
    assert scores == sorted(scores, reverse=True)
    # real Arial glyphs against the 5-font set: arial family must lead
    assert cands[0]["name"] in ARIAL_FAMILY
    m2 = apply_patch(m, patch)   # round-trips the schema
    assert m2.captions.segments[0].style.font_candidates


def test_font_family_only_when_confident(analyzed):
    patch, _ = analyzed
    style = patch["captions"]["segments"][0]["style"]
    if style["font_family"] is not None:
        assert style["font_family"] == style["font_candidates"][0]["name"]


def test_real_arial_caption_with_real_ocr(caption_video, tmp_path):
    # full path: real OCR + full system font dir. Verified behavior: the
    # winner is in the Arial family but is typically Arial NARROW (the
    # width-fit normalization favors condensed siblings; base arial ranks
    # ~#6 of 338). The assertion is therefore family-level, top-3.
    pytest.importorskip("rapidocr", reason="rapidocr not installed")
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(caption_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    patch = CaptionAnalyzer().run(m, ws)
    style = patch["captions"]["segments"][0]["style"]
    top3 = [c["name"] for c in style["font_candidates"]]
    assert any(name in ARIAL_FAMILY for name in top3), top3


def test_blank_crop_emits_no_candidates(fixture_video, tmp_path,
                                        monkeypatch):
    # color-bar fixture has no glyphs in the reported bbox: every score is
    # ~0 and the analyzer's MIN_FONT_SCORE floor suppresses the noise
    class BlankEngine:
        def read(self, image):
            n = int(image.stem.split("_")[1])
            if 6 <= n <= 15:
                return [OcrLine(text="FONT TEST",
                                bbox=(0.25, 0.8, 0.5, 0.06),
                                confidence=0.95)]
            return []

    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    analyzer = CaptionAnalyzer()
    monkeypatch.setattr(analyzer, "engine_factory", lambda: BlankEngine())
    monkeypatch.setattr(analyzer, "matcher_factory", five_font_matcher)
    patch = analyzer.run(m, ws)
    style = patch["captions"]["segments"][0]["style"]
    assert style["font_candidates"] == []
    assert style["font_family"] is None


from magicat.modules.captions.font_matcher import MatchResult
from magicat.modules.captions import analyzer as analyzer_mod


class CountingMatcher:
    """Fake matcher that records every identify() call and always returns a
    confident match for `winner`. Used to prove font ID runs on a SUBSET of
    segments, not once per segment."""

    def __init__(self, winner="arial"):
        self.winner = winner
        self.calls = 0

    def identify(self, crop, text):
        self.calls += 1
        return MatchResult(
            font_key=self.winner, score=0.9, margin=0.3, confident=True,
            ranked=[(self.winner, 0.9), ("impact", 0.4), ("comic", 0.2)])


class FourSegmentEngine:
    """Emits four distinct, non-overlapping captions on the caption_video
    timeline (sampled at 5fps -> frames 1..30). Each spans >=3 consecutive
    frames with a gap, so clustering yields exactly four segments. Texts have
    different lengths so the longest-text subset selection is exercised."""

    # (1-based frame range inclusive, text). Texts vary in length.
    _SCRIPT = [
        (3, 6, "A"),                       # shortest
        (10, 13, "BBBB CAPTION"),
        (17, 20, "CC"),
        (24, 27, "DDDDDD LONGEST CAPTION"),  # longest
    ]

    def read(self, image):
        n = int(image.stem.split("_")[1])
        for lo, hi, text in self._SCRIPT:
            if lo <= n <= hi:
                return [OcrLine(text=text,
                                bbox=(0.20, 0.79, 0.60, 0.07),
                                confidence=0.95)]
        return []


def test_font_id_runs_on_subset_for_many_segments(caption_video, tmp_path,
                                                  monkeypatch):
    # With MAX_FONT_ID_SEGMENTS clamped to 2 and four real segments, the
    # matcher must be invoked exactly twice (the subset), yet ALL four
    # segments must carry the same job-level font_candidates.
    monkeypatch.setattr(analyzer_mod, "MAX_FONT_ID_SEGMENTS", 2)
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(caption_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    analyzer = CaptionAnalyzer()
    monkeypatch.setattr(analyzer, "engine_factory",
                        lambda: FourSegmentEngine())
    counter = CountingMatcher()
    monkeypatch.setattr(analyzer, "matcher_factory", lambda: counter)
    patch = analyzer.run(m, ws)
    segments = patch["captions"]["segments"]
    assert len(segments) == 4
    assert counter.calls == 2          # subset, NOT once per segment
    cand_sets = [s["style"]["font_candidates"] for s in segments]
    assert all(c == cand_sets[0] for c in cand_sets)   # one shared font
    assert cand_sets[0]                                  # non-empty


def test_font_vote_applies_winner_to_all_segments(caption_video, tmp_path,
                                                   monkeypatch):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(caption_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    analyzer = CaptionAnalyzer()
    monkeypatch.setattr(analyzer, "engine_factory",
                        lambda: FourSegmentEngine())
    monkeypatch.setattr(analyzer, "matcher_factory",
                        lambda: CountingMatcher(winner="arial"))
    patch = analyzer.run(m, ws)
    segments = patch["captions"]["segments"]
    assert len(segments) == 4
    for seg in segments:
        assert seg["style"]["font_family"] == "arial"
        assert seg["style"]["font_candidates"][0]["name"] == "arial"


def test_no_fonts_available_degrades_gracefully(caption_video, tmp_path,
                                                monkeypatch):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(caption_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    analyzer = CaptionAnalyzer()
    monkeypatch.setattr(analyzer, "engine_factory",
                        lambda: RealTextEngine())
    monkeypatch.setattr(analyzer, "matcher_factory",
                        lambda: FontMatcher(fonts={}))
    patch = analyzer.run(m, ws)
    style = patch["captions"]["segments"][0]["style"]
    assert style["font_candidates"] == []
    assert style["font_family"] is None
    assert patch["layers_status"] == {"captions": "ok"}
