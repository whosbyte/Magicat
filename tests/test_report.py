# tests/test_report.py
from magicat.core.workspace import Workspace
from magicat.manifest.schema import LayerState, Manifest, Shot, Source
from magicat.modules.report import ReportExporter, build_report


def rich_manifest() -> Manifest:
    return Manifest(
        job_id="job12345678",
        source=Source(file="C:/m/source.mp4", url="https://t.example/v/1",
                      platform="tiktok", fps=30.0, resolution="480x854",
                      duration=6.0),
        shots=[Shot(id="shot_000", start=0.0, end=2.0,
                    keyframes=["C:/m/kf0.jpg"]),
               Shot(id="shot_001", start=2.0, end=6.0,
                    keyframes=["C:/m/kf1.jpg"])],
        audio={"music": {
            "detected": True, "title": "Around the World",
            "artist": "Daft Punk", "provider": "audd",
            "timeline_offset": 1.0,
            "song_segment": {"start_in_song": 30.0, "duration": 5.0},
            "acquisition": {"status": "acquired", "file": "C:/m/music.mp3",
                            "links": {"spotify": "https://sp/x",
                                      "soundcloud": "https://sc/y"}},
        }},
        captions={"segments": [{
            "text": "HELLO WORLD", "t_start": 1.0, "t_end": 3.0,
            "style": {"font_family": "arial",
                      "font_candidates": [
                          {"name": "arial", "confidence": 0.91},
                          {"name": "calibri", "confidence": 0.62}],
                      "fill": "#FDFDFD", "alignment": "center"},
        }]},
        layers_status={"source": LayerState.OK, "shots": LayerState.OK,
                       "music": LayerState.OK,
                       "captions": LayerState.OK},
    )


def test_build_report_dict():
    report = build_report(rich_manifest())
    assert report["job_id"] == "job12345678"
    assert report["source"]["platform"] == "tiktok"
    assert report["shots"]["count"] == 2
    music = report["music"]
    assert music["title"] == "Around the World"
    assert music["identified_by"] == "audd"
    assert music["links"]["spotify"] == "https://sp/x"
    assert music["used_segment"] == {"start_in_song": 30.0, "duration": 5.0}
    caps = report["captions"]
    assert caps["count"] == 1
    assert caps["fonts"] == ["arial"]
    assert caps["transcript"] == ["HELLO WORLD"]
    assert report["layers"]["music"] == "ok"


def test_build_report_no_music_no_captions():
    report = build_report(Manifest(job_id="j"))
    assert report["music"]["detected"] is False
    assert report["captions"]["count"] == 0
    assert report["captions"]["fonts"] == []


def test_html_report_renders(tmp_path):
    ws = Workspace(tmp_path / "job")
    out = ReportExporter().export(rich_manifest(), ws)
    assert out.name == "report.html"
    html = out.read_text(encoding="utf-8")
    assert "Around the World" in html
    assert "Daft Punk" in html
    assert "HELLO WORLD" in html
    assert "arial" in html
    assert "https://sp/x" in html
    assert "2 shots detected" in html


def test_html_escapes_user_text(tmp_path):
    m = Manifest(job_id="j", captions={"segments": [{
        "text": "<script>alert(1)</script>", "t_start": 0.0, "t_end": 1.0,
    }]})
    ws = Workspace(tmp_path / "job")
    html = ReportExporter().export(m, ws).read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
