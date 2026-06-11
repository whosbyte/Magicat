# tests/test_export_capcut.py
import json
import zipfile

import pytest

from magicat.core.interfaces import SkippedExport
from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import Manifest, Shot, Source
from magicat.modules.export.capcut import MICROS, CapCutExporter
from tests.conftest import run_ffmpeg


def capcut_manifest(fixture_video, tmp_path, music: bool = False):
    audio = {}
    if music:
        mp3 = tmp_path / "music.mp3"
        run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=660:duration=10",
                    "-c:a", "libmp3lame", str(mp3)])
        audio = {"music": {
            "detected": True, "title": "Song", "artist": "Artist",
            "timeline_offset": 2.0,
            "song_segment": {"start_in_song": 0.0, "duration": 4.0},
            "acquisition": {"status": "acquired", "file": str(mp3),
                            "links": {}},
        }}
    return Manifest(
        job_id="j",
        source=Source(file=str(fixture_video), fps=30.0,
                      resolution="320x640", duration=6.0),
        shots=[Shot(id="shot_000", start=0.0, end=2.0),
               Shot(id="shot_001", start=2.0, end=6.0)],
        captions={"segments": [{
            "text": "HELLO WORLD", "t_start": 1.0, "t_end": 3.0,
            "style": {"fill": "#FDFDFD", "alignment": "center"},
        }]},
        audio=audio,
    )


def load_draft(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        content = next(n for n in names if n.endswith("draft_content.json"))
        return json.loads(zf.read(content)), names


def test_exporter_disabled_by_flag_raises_skipped(fixture_video, tmp_path,
                                                  monkeypatch):
    monkeypatch.setenv("MAGICAT_CAPCUT_EXPORT", "0")
    ws = Workspace(tmp_path / "job")
    with pytest.raises(SkippedExport):
        CapCutExporter().export(capcut_manifest(fixture_video, tmp_path), ws)


def test_draft_zip_structure(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    out = CapCutExporter().export(capcut_manifest(fixture_video, tmp_path),
                                  ws)
    assert out.name == "capcut_draft.zip"
    draft, names = load_draft(out)
    assert any(n.endswith("draft_meta_info.json") for n in names)
    assert any(n == "CAPCUT_INSTRUCTIONS.txt" for n in names)
    # FORMAT SNAPSHOT (pycapcut 0.0.3 / CapCut int'l): pin the load-bearing
    # structure so a dependency bump that changes the format fails loudly
    track_types = [t["type"] for t in draft["tracks"]]
    assert "video" in track_types
    assert "text" in track_types
    video_track = next(t for t in draft["tracks"] if t["type"] == "video")
    segs = video_track["segments"]
    assert len(segs) == 2
    # microseconds + duration semantics (trange is start+DURATION)
    assert segs[0]["target_timerange"]["start"] == 0
    assert segs[0]["target_timerange"]["duration"] == 2 * MICROS
    assert segs[1]["target_timerange"]["start"] == 2 * MICROS
    assert segs[1]["target_timerange"]["duration"] == 4 * MICROS
    assert segs[0]["source_timerange"]["start"] == 0
    assert segs[1]["source_timerange"]["start"] == 2 * MICROS
    # one shared source material
    material_ids = {s["material_id"] for s in segs}
    assert len(material_ids) == 1
    assert len(draft["materials"]["videos"]) == 1


def test_draft_includes_music_at_offset(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    out = CapCutExporter().export(
        capcut_manifest(fixture_video, tmp_path, music=True), ws)
    draft, _ = load_draft(out)
    audio_track = next(t for t in draft["tracks"] if t["type"] == "audio")
    seg = audio_track["segments"][0]
    assert seg["target_timerange"]["start"] == 2 * MICROS
    assert seg["target_timerange"]["duration"] == 4 * MICROS


def test_draft_includes_caption_text(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    out = CapCutExporter().export(capcut_manifest(fixture_video, tmp_path),
                                  ws)
    draft, _ = load_draft(out)
    texts = json.dumps(draft["materials"].get("texts", []))
    assert "HELLO WORLD" in texts


def test_caption_style_mapped(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    out = CapCutExporter().export(capcut_manifest(fixture_video, tmp_path),
                                  ws)
    draft, _ = load_draft(out)
    text_material = draft["materials"]["texts"][0]
    content = json.dumps(text_material)
    # fill #FDFDFD -> ~0.9922 per channel; alignment center
    assert "0.992" in content


def test_no_shots_raises_value_error(tmp_path):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file="x.mp4"))
    with pytest.raises(ValueError, match="no shots"):
        CapCutExporter().export(m, ws)


def test_pipeline_capcut_layer(fixture_video, tmp_path):
    from magicat.core.pipeline import run_job
    manifest = run_job(str(fixture_video), tmp_path / "job")
    assert manifest.layers_status["capcut_zip"].value == "ok"
    assert (tmp_path / "job" / "exports" / "capcut_draft.zip").is_file()


def test_pipeline_capcut_flag_off_marks_skipped(fixture_video, tmp_path,
                                                monkeypatch):
    monkeypatch.setenv("MAGICAT_CAPCUT_EXPORT", "0")
    from magicat.core.pipeline import run_job
    manifest = run_job(str(fixture_video), tmp_path / "job")
    assert manifest.layers_status["capcut_zip"].value == "skipped"
