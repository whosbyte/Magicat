# tests/test_pipeline.py
import pytest

from magicat.core.pipeline import run_job
from magicat.manifest.schema import LayerState


def test_run_job_end_to_end(fixture_video, tmp_path):
    workdir = tmp_path / "job"
    manifest = run_job(str(fixture_video), workdir)

    assert manifest.layers_status["source"] == LayerState.OK
    assert manifest.layers_status["shots"] == LayerState.OK
    assert manifest.layers_status["preview_mp4"] == LayerState.OK
    assert len(manifest.shots) == 3
    assert any(e.format == "preview_mp4" for e in manifest.exports)
    assert (workdir / "manifest.json").is_file()
    assert (workdir / "exports" / "preview.mp4").is_file()


def test_analyzer_failure_degrades_gracefully(fixture_video, tmp_path,
                                              monkeypatch):
    from magicat.core import registry

    def boom(manifest, ws):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(registry.get_analyzer("cut_detection"), "run", boom)
    manifest = run_job(str(fixture_video), tmp_path / "job")
    assert manifest.layers_status["shots"] == LayerState.FAILED
    # exporter cannot render without shots and must say so now
    assert manifest.layers_status["preview_mp4"] == LayerState.FAILED
    assert (tmp_path / "job" / "manifest.json").is_file()


def test_ingest_failure_is_fatal(tmp_path):
    with pytest.raises(Exception):
        run_job(str(tmp_path / "does_not_exist.mp4"), tmp_path / "job")


def test_manifest_paths_are_absolute(fixture_video, tmp_path, monkeypatch):
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    manifest = run_job(str(fixture_video), Path("jobs") / "rel")
    assert Path(manifest.source.file).is_absolute()
    for shot in manifest.shots:
        for kf in shot.keyframes:
            assert Path(kf).is_absolute()
    for export in manifest.exports:
        assert Path(export.artifact).is_absolute()
