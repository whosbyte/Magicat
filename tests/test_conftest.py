# tests/test_conftest.py
from tests.conftest import probe_duration


def test_fixture_video_is_six_seconds(fixture_video):
    assert fixture_video.is_file()
    assert abs(probe_duration(fixture_video) - 6.0) < 0.2
