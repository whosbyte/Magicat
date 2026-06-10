# tests/test_backlog_m3.py
"""M2-review backlog items: schema additions + behavior fixes (M3 Task 1-2)."""
from pathlib import Path

import pytest

from magicat.manifest.schema import (
    Acquisition,
    CaptionSegment,
    Manifest,
    Music,
)


def test_music_provider_field():
    m = Music(detected=True, title="T", artist="A", provider="audd")
    assert m.provider == "audd"
    assert Music().provider is None


def test_acquisition_skip_reason_field():
    a = Acquisition(status="skipped", skip_reason="policy:link_only")
    assert a.skip_reason == "policy:link_only"
    assert Acquisition().skip_reason is None


def test_caption_segment_crops_field():
    seg = CaptionSegment(text="X", t_start=0.0, t_end=1.0,
                         crops=["a.png", "b.png"])
    assert seg.crops == ["a.png", "b.png"]
    assert CaptionSegment(text="X", t_start=0.0, t_end=1.0).crops == []


def test_align_carries_provider():
    from magicat.modules.audio.identify import align
    from magicat.modules.audio.extract import AudioWindow
    from magicat.modules.audio.providers import SongMatch

    windows = [AudioWindow(t_start=0.0, path=Path("w.wav"))]
    matches = [SongMatch(title="T", artist="A", song_offset_s=10.0,
                         provider="acrcloud")]
    music = align(windows, matches, video_duration=20.0, window_s=12.0)
    assert music["provider"] == "acrcloud"


def test_acquire_records_skip_reason_link_only(tmp_path, monkeypatch):
    from magicat.core.workspace import Workspace
    from magicat.modules.audio.acquire import Candidate, MusicAcquisition
    from magicat.manifest.schema import Source

    ws = Workspace(tmp_path / "job")
    monkeypatch.setenv("MAGICAT_ACQUISITION_POLICY", "link_only")
    analyzer = MusicAcquisition()
    monkeypatch.setattr(analyzer, "prober", lambda q: Candidate(
        url="https://soundcloud.com/x/y", title="T", duration=100.0,
        license="all-rights-reserved", source="soundcloud"))
    m = Manifest(job_id="j", source=Source(file="x.mp4"),
                 audio={"music": {"detected": True, "title": "T",
                                  "artist": "A", "duration_s": 100.0,
                                  "song_segment": {"start_in_song": 0.0,
                                                   "duration": 10.0}}})
    patch = analyzer.run(m, ws)
    acq = patch["audio"]["music"]["acquisition"]
    assert acq["status"] == "skipped"
    assert acq["skip_reason"] == "policy:link_only"


def test_acquire_records_skip_reason_license_gate(tmp_path, monkeypatch):
    from magicat.core.workspace import Workspace
    from magicat.modules.audio.acquire import Candidate, MusicAcquisition
    from magicat.manifest.schema import Source

    ws = Workspace(tmp_path / "job")
    monkeypatch.setenv("MAGICAT_ACQUISITION_POLICY", "licensed_only")
    analyzer = MusicAcquisition()
    monkeypatch.setattr(analyzer, "prober", lambda q: Candidate(
        url="https://soundcloud.com/x/y", title="T", duration=100.0,
        license="all-rights-reserved", source="soundcloud"))
    m = Manifest(job_id="j", source=Source(file="x.mp4"),
                 audio={"music": {"detected": True, "title": "T",
                                  "artist": "A", "duration_s": 100.0,
                                  "song_segment": {"start_in_song": 0.0,
                                                   "duration": 10.0}}})
    patch = analyzer.run(m, ws)
    acq = patch["audio"]["music"]["acquisition"]
    assert acq["status"] == "skipped"
    assert acq["skip_reason"] == "license:all-rights-reserved"


def test_caption_t_end_clamped_to_duration(fixture_video, tmp_path,
                                           monkeypatch):
    from magicat.core.workspace import Workspace
    from magicat.manifest.patch import apply_patch
    from magicat.manifest.schema import Source
    from magicat.modules.captions.analyzer import CaptionAnalyzer
    from magicat.modules.captions.ocr import OcrLine
    from magicat.modules.ingest import IngestAnalyzer

    class TailEngine:
        """Caption runs to the very last sampled frame (frame 25..30 at
        5fps = t 4.8..5.8; raw cluster t_end = 5.8+0.2 = 6.0)."""

        def read(self, image):
            n = int(image.stem.split("_")[1])
            if n >= 25:
                return [OcrLine(text="TAIL CAPTION",
                                bbox=(0.25, 0.8, 0.5, 0.06),
                                confidence=0.95)]
            return []

    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    # force a duration the raw cluster t_end (6.0) definitely exceeds, so
    # this test FAILS until the clamp exists (no tautology on probe jitter)
    src = m.source.model_dump(mode="json")
    src["duration"] = 5.9
    m = apply_patch(m, {"source": src})
    analyzer = CaptionAnalyzer()
    monkeypatch.setattr(analyzer, "engine_factory", lambda: TailEngine())
    patch = analyzer.run(m, ws)
    seg = patch["captions"]["segments"][0]
    assert seg["t_end"] == 5.9            # clamped to source.duration


def test_run_ffprobe_helper(fixture_video):
    from magicat.core.ffmpeg import run_ffprobe
    data = run_ffprobe(fixture_video, "format=duration")
    assert abs(float(data["format"]["duration"]) - 6.0) < 0.2


def test_sanitize_query():
    from magicat.modules.audio.acquire import sanitize_query
    assert sanitize_query('Artist: "Title"') == "Artist Title"
    assert sanitize_query("A;B|C&D") == "A B C D"
    assert sanitize_query("  plain  text  ") == "plain text"
    assert sanitize_query("Don't Stop Believin'") == "Don't Stop Believin'"


def test_unknown_duration_rejects_absurdly_long_candidate():
    from magicat.modules.audio.acquire import validate_candidate
    from magicat.modules.audio.acquire import Candidate
    ten_hour_loop = Candidate(
        url="https://youtube.com/watch?v=1",
        title="Around the World (10 hour loop)", duration=36000.0,
        license=None, source="youtube")
    match_info = {"title": "Around the World", "artist": "Daft Punk",
                  "duration_s": None}   # provider gave no duration
    assert validate_candidate(ten_hour_loop, match_info) is False


def test_cli_default_workdir_is_per_job(fixture_video, tmp_path,
                                        monkeypatch):
    from typer.testing import CliRunner
    from magicat.cli import app
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    r1 = runner.invoke(app, ["run", str(fixture_video)])
    r2 = runner.invoke(app, ["run", str(fixture_video)])
    assert r1.exit_code == 0 and r2.exit_code == 0
    jobs = list((tmp_path / "jobs").iterdir())
    assert len(jobs) == 2          # one fresh directory per job, no reuse


def test_cli_job_dir_matches_manifest_job_id(fixture_video, tmp_path,
                                             monkeypatch):
    import json
    from typer.testing import CliRunner
    from magicat.cli import app
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    assert runner.invoke(app, ["run", str(fixture_video)]).exit_code == 0
    job_dir = next((tmp_path / "jobs").iterdir())
    manifest = json.loads((job_dir / "manifest.json").read_text("utf-8"))
    assert manifest["job_id"].startswith(job_dir.name)
