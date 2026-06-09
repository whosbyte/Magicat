# tests/test_workspace.py
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest


def test_workspace_creates_layout(tmp_path):
    ws = Workspace(tmp_path / "job1")
    assert ws.media_dir.is_dir()
    assert ws.keyframes_dir.is_dir()
    assert ws.exports_dir.is_dir()


def test_manifest_persistence_round_trip(tmp_path):
    ws = Workspace(tmp_path / "job1")
    m = Manifest(job_id="job1")
    ws.save_manifest(m)
    assert ws.manifest_path.is_file()
    assert ws.load_manifest() == m
