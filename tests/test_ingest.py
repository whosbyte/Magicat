# tests/test_ingest.py
from pathlib import Path

from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest, Source
from magicat.modules.ingest import IngestAnalyzer, detect_platform, probe


def test_detect_platform():
    assert detect_platform("https://www.tiktok.com/@u/video/1") == "tiktok"
    assert detect_platform("https://www.instagram.com/reel/abc/") == "instagram"
    assert detect_platform("https://youtube.com/shorts/xyz") == "youtube"
    assert detect_platform("https://youtu.be/xyz") == "youtube"
    assert detect_platform("https://example.com/v.mp4") is None


def test_probe_reads_metadata(fixture_video):
    meta = probe(fixture_video)
    assert abs(meta["fps"] - 30.0) < 0.01
    assert meta["resolution"] == "320x640"
    assert abs(meta["duration"] - 6.0) < 0.2


def test_ingest_local_file(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    patch = IngestAnalyzer().run(m, ws)
    src = patch["source"]
    assert Path(src["file"]).name == "source.mp4"
    assert Path(src["file"]).is_file()
    assert src["resolution"] == "320x640"
    assert patch["layers_status"] == {"source": "ok"}


def test_ingest_url_uses_downloader(fixture_video, tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "job")
    url = "https://www.tiktok.com/@u/video/1"
    m = Manifest(job_id="j", source=Source(url=url))
    analyzer = IngestAnalyzer()
    monkeypatch.setattr(analyzer, "downloader", lambda u, dest: fixture_video)
    patch = analyzer.run(m, ws)
    assert patch["source"]["platform"] == "tiktok"
    assert Path(patch["source"]["file"]).is_file()


def test_probe_rejects_audio_only(tmp_path):
    import subprocess
    import pytest
    audio = tmp_path / "audio.m4a"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
         "-vn", str(audio)],
        check=True, capture_output=True)
    with pytest.raises(ValueError, match="no video stream"):
        probe(audio)
