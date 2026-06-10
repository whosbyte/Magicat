# tests/test_report_sources.py
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest, Shot, Source
from magicat.modules.report import ReportExporter, build_report


def manifest_with_sources() -> Manifest:
    return Manifest(
        job_id="j", source=Source(file="x.mp4", duration=6.0),
        shots=[Shot(id="shot_000", start=0.0, end=3.0)],
        source_matches=[{
            "shot_id": "shot_000",
            "candidates": [
                {"url": "https://www.tiktok.com/@u/video/9",
                 "title": "Original clip", "score": 0.9},
                {"url": "https://vimeo.com/123", "title": "Mirror",
                 "score": 0.5},
            ],
        }])


def test_build_report_includes_sources():
    report = build_report(manifest_with_sources())
    sources = report["sources"]
    assert sources["searched"] is True
    assert sources["shots"][0]["shot_id"] == "shot_000"
    assert sources["shots"][0]["candidates"][0]["url"] == \
        "https://www.tiktok.com/@u/video/9"


def test_build_report_sources_absent():
    report = build_report(Manifest(job_id="j"))
    assert report["sources"]["searched"] is False
    assert report["sources"]["shots"] == []


def test_html_report_renders_source_links(tmp_path):
    ws = Workspace(tmp_path / "job")
    html = ReportExporter().export(manifest_with_sources(), ws) \
        .read_text(encoding="utf-8")
    assert "Source footage" in html
    assert "https://www.tiktok.com/@u/video/9" in html
    assert "Original clip" in html


def test_html_report_no_sources_section_message(tmp_path):
    ws = Workspace(tmp_path / "job")
    html = ReportExporter().export(Manifest(job_id="j"), ws) \
        .read_text(encoding="utf-8")
    assert "No source search performed" in html
