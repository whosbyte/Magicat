# tests/test_export_srt.py
from magicat.manifest.schema import Manifest
from magicat.modules.export.srt import srt_timestamp, to_srt


def test_srt_timestamp():
    assert srt_timestamp(0.0) == "00:00:00,000"
    assert srt_timestamp(1.5) == "00:00:01,500"
    assert srt_timestamp(3661.25) == "01:01:01,250"


def test_to_srt_two_segments():
    m = Manifest(job_id="j", captions={"segments": [
        {"text": "HELLO WORLD", "t_start": 1.0, "t_end": 3.0},
        {"text": "SECOND LINE", "t_start": 3.5, "t_end": 5.2},
    ]})
    srt = to_srt(m)
    blocks = srt.strip().split("\n\n")
    assert len(blocks) == 2
    assert blocks[0].splitlines() == [
        "1", "00:00:01,000 --> 00:00:03,000", "HELLO WORLD"]
    assert blocks[1].splitlines() == [
        "2", "00:00:03,500 --> 00:00:05,200", "SECOND LINE"]
    assert srt.endswith("\n")


def test_to_srt_empty():
    assert to_srt(Manifest(job_id="j")) == ""
