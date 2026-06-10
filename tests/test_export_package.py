# tests/test_export_package.py
import zipfile

from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import Manifest, Source
from magicat.modules.cuts_pyscenedetect import CutDetector
from magicat.modules.export.package import PremiereResolvePackage
from magicat.modules.ingest import IngestAnalyzer
from magicat.modules.report import ReportExporter
from tests.conftest import run_ffmpeg


def prepared(fixture_video, tmp_path, with_music: bool):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    m = apply_patch(m, CutDetector().run(m, ws))
    m = apply_patch(m, {"captions": {"segments": [
        {"text": "HELLO", "t_start": 1.0, "t_end": 2.0}]}})
    if with_music:
        music = tmp_path / "music.mp3"
        run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=660:duration=10",
                    "-c:a", "libmp3lame", str(music)])
        audio = m.audio.model_dump(mode="json")
        audio["music"] = {
            "detected": True, "title": "T", "artist": "A",
            "timeline_offset": 1.0,
            "song_segment": {"start_in_song": 0.0, "duration": 5.0},
            "acquisition": {"status": "acquired", "file": str(music),
                            "links": {}},
        }
        m = apply_patch(m, {"audio": audio})
    ReportExporter().export(m, ws)   # zip includes the report
    return m, ws


def test_zip_contains_project_files(fixture_video, tmp_path):
    m, ws = prepared(fixture_video, tmp_path, with_music=True)
    out = PremiereResolvePackage().export(m, ws)
    assert out.name == "premiere_resolve.zip"
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "project.xml" in names
        assert "captions.srt" in names
        assert "report.html" in names
        assert "IMPORT_INSTRUCTIONS.txt" in names
        assert "media/source.mp4" in names
        assert "media/music.mp3" in names
        xml = zf.read("project.xml").decode("utf-8")
        # pathurls inside the zip are RELATIVE ("media/<name>") so the
        # project resolves against wherever the zip is extracted - an
        # absolute path into the (deleted) staging dir would dangle
        assert "<pathurl>media/source.mp4</pathurl>" in xml
        assert "exports/package" not in xml
        srt = zf.read("captions.srt").decode("utf-8")
        assert "HELLO" in srt


def test_zip_without_music_or_captions(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    m = apply_patch(m, CutDetector().run(m, ws))
    ReportExporter().export(m, ws)
    out = PremiereResolvePackage().export(m, ws)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "project.xml" in names
        assert "captions.srt" not in names      # nothing to caption
        assert "media/music.mp3" not in names


def test_pipeline_produces_zip(fixture_video, tmp_path):
    from magicat.core.pipeline import run_job
    manifest = run_job(str(fixture_video), tmp_path / "job")
    assert any(e.format == "premiere_resolve_zip"
               for e in manifest.exports)
    assert (tmp_path / "job" / "exports" / "premiere_resolve.zip").is_file()
    assert manifest.report["shots"]["count"] == 3
    assert manifest.report["layers"]["premiere_resolve_zip"] == "ok"
