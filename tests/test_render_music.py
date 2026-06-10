# tests/test_render_music.py
from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import Manifest, Source
from magicat.modules.cuts_pyscenedetect import CutDetector
from magicat.modules.ingest import IngestAnalyzer
from magicat.modules.render_preview import PreviewRenderer, build_filtergraph
from tests.conftest import probe_duration, run_ffmpeg


def test_filtergraph_no_music():
    fg = build_filtergraph([(0.0, 2.0), (2.0, 4.0)], with_music=False)
    assert "concat=n=2:v=1:a=1[vout][aout]" in fg
    assert "amix" not in fg


def test_filtergraph_with_music():
    fg = build_filtergraph([(0.0, 2.0)], with_music=True,
                           music_offset_s=1.5, music_volume=0.8)
    assert "adelay=1500:all=1" in fg
    assert "amix=inputs=2:duration=first:normalize=0" in fg
    assert "[vout][aconcat]" in fg


def music_fixture(tmp_path):
    p = tmp_path / "music.mp3"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=880:duration=30",
                "-c:a", "libmp3lame", str(p)])
    return p


def analyzed_manifest(fixture_video, ws):
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    m = apply_patch(m, CutDetector().run(m, ws))
    return m


def test_preview_with_music_keeps_video_duration(fixture_video, tmp_path):
    # 30s music on a 6s video: duration=first must clamp to 6s (the old
    # two-pass approach would have drifted/overrun)
    ws = Workspace(tmp_path / "job")
    m = analyzed_manifest(fixture_video, ws)
    music = music_fixture(tmp_path)
    audio = m.audio.model_dump(mode="json")
    audio["music"] = {
        "detected": True, "title": "T", "artist": "A",
        "timeline_offset": 1.0,
        "song_segment": {"start_in_song": 0.0, "duration": 5.0},
        "acquisition": {"status": "acquired", "file": str(music),
                        "links": {}},
    }
    m = apply_patch(m, {"audio": audio})
    out = PreviewRenderer().export(m, ws)
    assert abs(probe_duration(out) - (m.source.duration or 0)) < 0.3


def test_preview_without_music_unchanged(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    m = analyzed_manifest(fixture_video, ws)
    out = PreviewRenderer().export(m, ws)
    assert abs(probe_duration(out) - (m.source.duration or 0)) < 0.3
