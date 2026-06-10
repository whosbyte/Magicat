# tests/test_caption_crops.py
from pathlib import Path

from PIL import Image

from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import Manifest, Source
from magicat.modules.captions.analyzer import CaptionAnalyzer
from magicat.modules.captions.ocr import OcrLine
from magicat.modules.ingest import IngestAnalyzer


class OneCaptionEngine:
    def read(self, image):
        n = int(image.stem.split("_")[1])
        if 6 <= n <= 15:
            return [OcrLine(text="CROPPED CAPTION",
                            bbox=(0.25, 0.8, 0.5, 0.06), confidence=0.95)]
        return []


def test_segment_crops_are_persisted(fixture_video, tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    analyzer = CaptionAnalyzer()
    monkeypatch.setattr(analyzer, "engine_factory",
                        lambda: OneCaptionEngine())
    patch = analyzer.run(m, ws)
    seg = patch["captions"]["segments"][0]
    crops = seg["crops"]
    assert 1 <= len(crops) <= 3
    for crop_path in crops:
        p = Path(crop_path)
        assert p.is_file()
        with Image.open(p) as img:
            w, h = img.size
            assert w > 0 and h > 0
            # crop covers the caption bbox plus margin: wider than tall
            assert w > h
    # round-trips the schema
    m2 = apply_patch(m, patch)
    assert m2.captions.segments[0].crops == crops
