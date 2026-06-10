# tests/test_audio_analyzer.py
import pytest

from magicat import config
from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import LayerState, Manifest, Source
from magicat.modules.audio.analyzer import AudioAnalyzer
from magicat.modules.audio.providers import SongMatch
from magicat.modules.ingest import IngestAnalyzer


class OneSongProvider:
    name = "fake"

    def identify(self, clip):
        return SongMatch(title="Fixture Song", artist="Fixture Artist",
                         song_offset_s=30.0, provider="fake",
                         links={"song_link": "https://example.com/s"})


@pytest.fixture()
def ingested(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    return m, ws


def test_no_providers_configured_skips_layer(ingested, monkeypatch):
    m, ws = ingested
    analyzer = AudioAnalyzer()
    monkeypatch.setattr(analyzer, "provider_factory", lambda: [])
    patch = analyzer.run(m, ws)
    assert patch == {"layers_status": {"music": "skipped"}}


def test_detects_song_and_fills_music_section(ingested, monkeypatch):
    m, ws = ingested
    analyzer = AudioAnalyzer()
    monkeypatch.setattr(analyzer, "provider_factory",
                        lambda: [OneSongProvider()])
    patch = analyzer.run(m, ws)
    music = patch["audio"]["music"]
    assert music["detected"] is True
    assert music["title"] == "Fixture Song"
    assert music["timeline_offset"] == 0.0
    assert music["song_segment"]["start_in_song"] == 30.0
    assert music["acquisition"]["links"]["song_link"] == "https://example.com/s"
    assert patch["layers_status"] == {"music": "ok"}
    # validates against the schema
    m2 = apply_patch(m, patch)
    assert m2.audio.music.detected is True


class NoMatchProvider:
    name = "nomatch"

    def identify(self, clip):
        return None


def test_no_match_is_ok_with_detected_false(ingested, monkeypatch):
    m, ws = ingested
    analyzer = AudioAnalyzer()
    monkeypatch.setattr(analyzer, "provider_factory",
                        lambda: [NoMatchProvider()])
    patch = analyzer.run(m, ws)
    assert patch["audio"]["music"]["detected"] is False
    assert patch["layers_status"] == {"music": "ok"}


def test_fallback_to_second_provider(ingested, monkeypatch):
    # spec 6.3 step 2: AudD primary, ACRCloud fallback - when the primary
    # finds nothing across all windows, the next provider in the chain runs
    m, ws = ingested
    analyzer = AudioAnalyzer()
    monkeypatch.setattr(analyzer, "provider_factory",
                        lambda: [NoMatchProvider(), OneSongProvider()])
    patch = analyzer.run(m, ws)
    assert patch["audio"]["music"]["detected"] is True
    assert patch["audio"]["music"]["title"] == "Fixture Song"


def test_config_acquisition_policy(monkeypatch):
    assert config.acquisition_policy() == "always"
    monkeypatch.setenv("MAGICAT_ACQUISITION_POLICY", "link_only")
    assert config.acquisition_policy() == "link_only"
    monkeypatch.setenv("MAGICAT_ACQUISITION_POLICY", "bogus")
    with pytest.raises(ValueError):
        config.acquisition_policy()


def test_pipeline_includes_audio_analysis(fixture_video, tmp_path):
    # ambient provider env is cleared by the autouse fixture (Task 1)
    from magicat.core.pipeline import run_job
    manifest = run_job(str(fixture_video), tmp_path / "job")
    assert manifest.layers_status["music"] == LayerState.SKIPPED
