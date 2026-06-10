# tests/test_caption_ocr.py
import numpy as np
import pytest

from magicat.modules.captions.ocr import OcrLine, RapidOcrEngine, quad_to_bbox


def test_quad_to_bbox_normalizes():
    quad = np.array([[100.0, 700.0], [380.0, 700.0],
                     [380.0, 760.0], [100.0, 760.0]])
    bbox = quad_to_bbox(quad, width=480, height=854)
    x, y, w, h = bbox
    assert abs(x - 100 / 480) < 1e-6
    assert abs(y - 700 / 854) < 1e-6
    assert abs(w - 280 / 480) < 1e-6
    assert abs(h - 60 / 854) < 1e-6


def test_ocr_line_model():
    line = OcrLine(text="HI", bbox=(0.1, 0.8, 0.5, 0.05), confidence=0.97)
    assert line.text == "HI"


def test_rapidocr_reads_caption_frame(caption_video, tmp_path):
    pytest.importorskip("rapidocr", reason="rapidocr not installed")
    from magicat.modules.captions.sampling import sample_frames
    samples = sample_frames(caption_video, tmp_path / "frames")
    # t=2.0 -> inside HELLO WORLD window
    frame = next(s for s in samples if abs(s.t - 2.0) < 1e-6)
    engine = RapidOcrEngine()
    lines = engine.read(frame.path)
    assert lines, "OCR found no text on a frame with a caption"
    joined = " ".join(l.text.upper() for l in lines)
    assert "HELLO" in joined
    assert all(0.0 <= v <= 1.0 for l in lines for v in l.bbox)
    assert lines[0].bbox[1] > 0.5  # caption sits in the lower half


def test_rapidocr_empty_frame_returns_no_lines(caption_video, tmp_path):
    pytest.importorskip("rapidocr", reason="rapidocr not installed")
    from magicat.modules.captions.sampling import sample_frames
    samples = sample_frames(caption_video, tmp_path / "frames")
    frame = next(s for s in samples if abs(s.t - 0.2) < 1e-6)  # before t=1
    assert RapidOcrEngine().read(frame.path) == []
