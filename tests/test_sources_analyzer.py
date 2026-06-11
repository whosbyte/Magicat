# tests/test_sources_analyzer.py
from pathlib import Path

import pytest

from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import LayerState, Manifest, Shot, Source
from magicat.modules.sources.analyzer import SourceMatchAnalyzer
from magicat.modules.sources.providers import ImageMatch, ProviderError


def manifest_with_shots(tmp_path) -> Manifest:
    keyframes = []
    for i in range(2):
        kf = tmp_path / f"kf{i}.jpg"
        kf.write_bytes(b"\xff\xd8fake")
        keyframes.append(str(kf))
    return Manifest(
        job_id="j", source=Source(file="x.mp4", duration=6.0),
        shots=[Shot(id="shot_000", start=0.0, end=3.0,
                    keyframes=[keyframes[0]]),
               Shot(id="shot_001", start=3.0, end=6.0,
                    keyframes=[keyframes[1]])])


class OneHitProvider:
    name = "fake"

    def __init__(self):
        self.calls = []

    def search(self, keyframe: Path):
        self.calls.append(str(keyframe))
        return [ImageMatch(source_url="https://www.tiktok.com/@u/video/9",
                           title="Original", score=0.9, provider="fake")]


def test_no_providers_skips_layer(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "job")
    analyzer = SourceMatchAnalyzer()
    monkeypatch.setattr(analyzer, "provider_factory", lambda: [])
    patch = analyzer.run(manifest_with_shots(tmp_path), ws)
    assert patch == {"layers_status": {"source_matches": "skipped"}}


def test_matches_mapped_per_shot(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "job")
    analyzer = SourceMatchAnalyzer()
    provider = OneHitProvider()
    monkeypatch.setattr(analyzer, "provider_factory", lambda: [provider])
    m = manifest_with_shots(tmp_path)
    patch = analyzer.run(m, ws)
    sm = patch["source_matches"]
    assert len(sm) == 2
    assert sm[0]["shot_id"] == "shot_000"
    cand = sm[0]["candidates"][0]
    assert cand["url"] == "https://www.tiktok.com/@u/video/9"
    assert cand["title"] == "Original"
    assert cand["score"] == 0.9
    assert patch["layers_status"] == {"source_matches": "ok"}
    # one search per shot (middle keyframe)
    assert len(provider.calls) == 2
    m2 = apply_patch(m, patch)          # validates against the schema
    assert m2.source_matches[0].candidates[0].url.endswith("/9")


def test_provider_errors_degrade_per_shot(tmp_path, monkeypatch):
    class FlakyProvider:
        name = "flaky"

        def __init__(self):
            self.n = 0

        def search(self, keyframe):
            self.n += 1
            if self.n == 1:
                raise ProviderError("quota")
            return [ImageMatch(source_url="https://a/b", score=0.5,
                               provider="flaky")]

    ws = Workspace(tmp_path / "job")
    analyzer = SourceMatchAnalyzer()
    monkeypatch.setattr(analyzer, "provider_factory",
                        lambda: [FlakyProvider()])
    patch = analyzer.run(manifest_with_shots(tmp_path), ws)
    sm = patch["source_matches"]
    assert sm[0]["candidates"] == []          # shot 0 errored -> empty
    assert len(sm[1]["candidates"]) == 1      # shot 1 fine
    assert patch["layers_status"] == {"source_matches": "ok"}


def test_all_provider_calls_dead_marks_layer_failed(tmp_path, monkeypatch):
    class DeadProvider:
        name = "dead"

        def search(self, keyframe):
            raise ProviderError("bad key")

    ws = Workspace(tmp_path / "job")
    analyzer = SourceMatchAnalyzer()
    monkeypatch.setattr(analyzer, "provider_factory",
                        lambda: [DeadProvider()])
    patch = analyzer.run(manifest_with_shots(tmp_path), ws)
    assert patch == {"layers_status": {"source_matches": "failed"}}


def test_multi_provider_results_merged_and_ranked(tmp_path, monkeypatch):
    class P1:
        name = "p1"

        def search(self, keyframe):
            return [ImageMatch(source_url="https://www.tiktok.com/@u/v/1",
                               score=0.9, provider="p1")]

    class P2:
        name = "p2"

        def search(self, keyframe):
            return [
                ImageMatch(source_url="https://tiktok.com/@u/v/dup",
                           score=0.8, provider="p2"),     # same domain: deduped
                ImageMatch(source_url="https://vimeo.com/123",
                           score=0.7, provider="p2"),
            ]

    ws = Workspace(tmp_path / "job")
    analyzer = SourceMatchAnalyzer()
    monkeypatch.setattr(analyzer, "provider_factory", lambda: [P1(), P2()])
    patch = analyzer.run(manifest_with_shots(tmp_path), ws)
    urls = [c["url"] for c in patch["source_matches"][0]["candidates"]]
    assert urls == ["https://www.tiktok.com/@u/v/1", "https://vimeo.com/123"]


def test_pipeline_skips_without_keys(fixture_video, tmp_path):
    from magicat.core.pipeline import run_job
    manifest = run_job(str(fixture_video), tmp_path / "job")
    assert manifest.layers_status["source_matches"] == LayerState.SKIPPED
