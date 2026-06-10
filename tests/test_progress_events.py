# tests/test_progress_events.py
from magicat.core.pipeline import run_job


def test_progress_callback_receives_stage_events(fixture_video, tmp_path):
    seen: list[tuple[str, str]] = []
    run_job(str(fixture_video), tmp_path / "job",
            on_progress=lambda stage, state: seen.append((stage, state)))

    assert seen[0] == ("ingest", "start")
    assert ("ingest", "ok") in seen
    assert ("cut_detection", "start") in seen
    assert ("cut_detection", "ok") in seen
    # music layer is skipped without provider keys (env isolated by fixture)
    assert ("audio_analysis", "skipped") in seen
    assert ("caption_analysis", "ok") in seen
    assert ("preview_mp4", "start") in seen
    assert ("preview_mp4", "ok") in seen
    assert ("premiere_resolve_zip", "ok") in seen
    assert seen[-1] == ("job", "done")
    # every start eventually resolves
    starts = {s for s, st in seen if st == "start"}
    resolved = {s for s, st in seen if st in ("ok", "failed", "skipped")}
    assert starts <= resolved | {"job"}


def test_progress_callback_reports_failure(fixture_video, tmp_path,
                                           monkeypatch):
    from magicat.core import registry

    def boom(manifest, ws):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(registry.get_analyzer("cut_detection"), "run", boom)
    seen: list[tuple[str, str]] = []
    run_job(str(fixture_video), tmp_path / "job",
            on_progress=lambda stage, state: seen.append((stage, state)))
    assert ("cut_detection", "failed") in seen
    assert seen[-1] == ("job", "done")


def test_progress_callback_optional(fixture_video, tmp_path):
    manifest = run_job(str(fixture_video), tmp_path / "job")
    assert manifest.layers_status["shots"].value == "ok"


def test_raising_progress_callback_does_not_fail_job(fixture_video,
                                                     tmp_path):
    def explosive(stage, state):
        raise RuntimeError("telemetry down")

    manifest = run_job(str(fixture_video), tmp_path / "job",
                       on_progress=explosive)
    assert manifest.layers_status["shots"].value == "ok"
