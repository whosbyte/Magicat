# tests/test_schema.py
import json

import pytest
from pydantic import ValidationError

from magicat.manifest.schema import (
    MANIFEST_VERSION,
    LayerState,
    Manifest,
    Shot,
    Source,
)


def test_minimal_manifest_has_sane_defaults():
    m = Manifest(job_id="job1")
    assert m.manifest_version == MANIFEST_VERSION
    assert m.shots == []
    assert m.source_matches == []
    assert m.audio.music.detected is False
    assert m.captions.segments == []
    assert m.layers_status == {}
    assert m.exports == []


def test_manifest_json_round_trip():
    m = Manifest(
        job_id="job1",
        source=Source(file="C:/x/source.mp4", fps=30.0, duration=6.0,
                      resolution="320x640", platform="tiktok",
                      url="https://www.tiktok.com/@u/video/1"),
        shots=[Shot(id="shot_000", start=0.0, end=2.0, confidence=0.9)],
        layers_status={"shots": LayerState.OK},
    )
    data = json.loads(m.model_dump_json())
    m2 = Manifest.model_validate(data)
    assert m2 == m
    assert data["layers_status"]["shots"] == "ok"


def test_unknown_fields_rejected():
    with pytest.raises(ValidationError):
        Manifest(job_id="j", bogus_field=1)
