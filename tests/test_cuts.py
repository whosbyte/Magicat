# tests/test_cuts.py
from pathlib import Path

import pytest

from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest, Source
from magicat.modules.cuts_pyscenedetect import CutDetector
from magicat.modules.ingest import IngestAnalyzer
from magicat.manifest.patch import apply_patch


@pytest.fixture()
def ingested(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    return m, ws


def test_detects_three_shots_at_known_cuts(ingested):
    m, ws = ingested
    patch = CutDetector().run(m, ws)
    shots = patch["shots"]
    assert len(shots) == 3
    # ground truth: cuts at 2.0s and 4.0s (tolerance: spec says +/-2 frames @30fps)
    assert abs(shots[0]["end"] - 2.0) <= 0.1
    assert abs(shots[1]["start"] - 2.0) <= 0.1
    assert abs(shots[1]["end"] - 4.0) <= 0.1
    assert shots[0]["start"] == 0.0
    assert abs(shots[2]["end"] - 6.0) <= 0.2
    assert patch["layers_status"] == {"shots": "ok"}


def test_each_shot_has_keyframes(ingested):
    m, ws = ingested
    patch = CutDetector().run(m, ws)
    for shot in patch["shots"]:
        assert len(shot["keyframes"]) == 3
        for kf in shot["keyframes"]:
            p = Path(kf)
            assert p.is_file() and p.stat().st_size > 0


def test_uncut_video_yields_single_shot(fixture_video, tmp_path):
    # feed only the first shot to the detector: no cuts -> one full-length shot
    import subprocess
    single = tmp_path / "single.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(fixture_video), "-t", "2", "-c", "copy", str(single)],
        check=True, capture_output=True)
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(single)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    patch = CutDetector().run(m, ws)
    assert len(patch["shots"]) == 1
    assert patch["shots"][0]["start"] == 0.0
