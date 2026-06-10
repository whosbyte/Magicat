# tests/test_patch.py
import pytest
from pydantic import ValidationError

from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import LayerState, Manifest


def test_patch_replaces_section():
    m = Manifest(job_id="j")
    m2 = apply_patch(m, {"shots": [
        {"id": "shot_000", "start": 0.0, "end": 2.0},
    ]})
    assert len(m2.shots) == 1
    assert m2.shots[0].end == 2.0
    assert m.shots == []  # original untouched


def test_layers_status_merges_instead_of_replacing():
    m = Manifest(job_id="j", layers_status={"shots": LayerState.OK})
    m2 = apply_patch(m, {"layers_status": {"music": "failed"}})
    assert m2.layers_status == {
        "shots": LayerState.OK, "music": LayerState.FAILED,
    }


def test_invalid_patch_raises():
    m = Manifest(job_id="j")
    with pytest.raises(ValidationError):
        apply_patch(m, {"shots": [{"id": "x"}]})  # missing start/end


def test_unknown_section_raises():
    m = Manifest(job_id="j")
    with pytest.raises(ValidationError):
        apply_patch(m, {"bogus": 1})


def test_exports_append_instead_of_replacing():
    m = Manifest(job_id="j", exports=[
        {"format": "preview_mp4", "artifact": "a.mp4"}])
    m2 = apply_patch(m, {"exports": [
        {"format": "premiere_zip", "artifact": "b.zip"}]})
    assert [e.format for e in m2.exports] == ["preview_mp4", "premiere_zip"]
