# tests/test_caption_analyzer.py
import pytest

from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import LayerState, Manifest, Source
from magicat.modules.captions.analyzer import CaptionAnalyzer
from magicat.modules.captions.ocr import OcrLine
from magicat.modules.ingest import IngestAnalyzer


class ScriptedEngine:
    """Returns captions on frames 6..15 (1-based ffmpeg numbering), which
    map to t=1.0..2.8 - like a real burn-in. Do NOT change the 6..15 range:
    frame_00006 -> enumerate index 5 -> t=1.0."""

    def read(self, image):
        n = int(image.stem.split("_")[1])      # frame_00001 -> 1
        if 6 <= n <= 15:                        # ffmpeg numbers from 1
            return [OcrLine(text="FAKE CAPTION",
                            bbox=(0.25, 0.8, 0.5, 0.06), confidence=0.95)]
        return []


def test_caption_analyzer_with_scripted_engine(fixture_video, tmp_path,
                                               monkeypatch):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    analyzer = CaptionAnalyzer()
    monkeypatch.setattr(analyzer, "engine_factory", lambda: ScriptedEngine())
    patch = analyzer.run(m, ws)
    segments = patch["captions"]["segments"]
    assert len(segments) == 1
    seg = segments[0]
    assert seg["text"] == "FAKE CAPTION"
    assert abs(seg["t_start"] - 1.0) < 0.01
    assert seg["style"]["fill"].startswith("#")
    assert seg["style"]["alignment"] == "center"   # bbox is centered
    # size = bbox height * frame height px (fixture is 320x640): 0.06*640
    assert 35 <= seg["style"]["size"] <= 42
    assert patch["layers_status"] == {"captions": "ok"}
    m2 = apply_patch(m, patch)                  # validates against schema
    assert m2.captions.segments[0].text == "FAKE CAPTION"


def test_caption_analyzer_end_to_end_real_ocr(caption_video, tmp_path):
    pytest.importorskip("rapidocr", reason="rapidocr not installed")
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(caption_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    patch = CaptionAnalyzer().run(m, ws)
    segments = patch["captions"]["segments"]
    assert len(segments) == 2
    first, second = segments
    assert "HELLO" in first["text"].upper()
    assert "SECOND" in second["text"].upper()
    assert abs(first["t_start"] - 1.0) <= 0.4
    assert abs(first["t_end"] - 3.0) <= 0.4
    assert abs(second["t_start"] - 3.5) <= 0.4
    assert first["bbox"][1] > 0.5               # bottom half of the frame


def test_pipeline_runs_captions_layer(fixture_video, tmp_path, monkeypatch):
    pytest.importorskip("rapidocr", reason="rapidocr not installed")
    from magicat.core.pipeline import run_job
    manifest = run_job(str(fixture_video), tmp_path / "job")
    # color-bar fixture has no captions: layer ok, zero segments
    assert manifest.layers_status["captions"] == LayerState.OK
    assert manifest.captions.segments == []
