# tests/test_render.py
from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import Manifest, Source
from magicat.modules.cuts_pyscenedetect import CutDetector
from magicat.modules.ingest import IngestAnalyzer
from magicat.modules.render_preview import PreviewRenderer
from tests.conftest import probe_duration


def test_preview_renders_full_timeline(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    m = apply_patch(m, CutDetector().run(m, ws))

    out = PreviewRenderer().export(m, ws)

    assert out.is_file()
    assert out.suffix == ".mp4"
    # all three shots present: duration matches the source
    assert abs(probe_duration(out) - (m.source.duration or 0)) < 0.3


def test_export_with_no_shots_raises(fixture_video, tmp_path):
    import pytest
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    with pytest.raises(ValueError, match="no shots"):
        PreviewRenderer().export(m, ws)
