# tests/test_audio_providers.py
from pathlib import Path

import pytest

from magicat.modules.audio import providers
from magicat.modules.audio.providers import (
    AudDProvider,
    ProviderError,
    SongMatch,
    parse_timecode,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture()
def clip(tmp_path) -> Path:
    p = tmp_path / "clip.wav"
    p.write_bytes(b"RIFFfake")
    return p


def test_parse_timecode():
    assert parse_timecode("02:32") == 152.0
    assert parse_timecode("1:02:03") == 3723.0
    assert parse_timecode("00:00") == 0.0


def test_audd_success(monkeypatch, clip):
    payload = {"status": "success", "result": {
        "artist": "Imagine Dragons", "title": "Warriors",
        "album": "Warriors", "timecode": "02:32",
        "song_link": "https://lis.tn/Warriors",
        "spotify": {"external_urls": {
            "spotify": "https://open.spotify.com/track/abc"}, "id": "abc"},
    }}
    captured = {}

    def fake_post(url, data=None, files=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        return FakeResponse(payload)

    monkeypatch.setattr(providers.requests, "post", fake_post)
    match = AudDProvider(api_token="tok").identify(clip)
    assert captured["url"] == "https://api.audd.io/"
    assert captured["data"]["api_token"] == "tok"
    assert match.title == "Warriors"
    assert match.artist == "Imagine Dragons"
    assert match.song_offset_s == 152.0
    assert match.provider == "audd"
    assert match.links["song_link"] == "https://lis.tn/Warriors"
    assert match.links["spotify"] == "https://open.spotify.com/track/abc"
    assert match.provider_ids["spotify"] == "abc"


def test_audd_no_match_returns_none(monkeypatch, clip):
    monkeypatch.setattr(
        providers.requests, "post",
        lambda *a, **k: FakeResponse({"status": "success", "result": None}))
    assert AudDProvider(api_token="tok").identify(clip) is None


def test_audd_api_error_raises(monkeypatch, clip):
    payload = {"status": "error",
               "error": {"error_code": 901, "error_message": "limit reached"}}
    monkeypatch.setattr(providers.requests, "post",
                        lambda *a, **k: FakeResponse(payload))
    with pytest.raises(ProviderError, match="901"):
        AudDProvider(api_token="tok").identify(clip)


def test_song_match_defaults():
    m = SongMatch(title="T", artist="A", song_offset_s=1.0, provider="x")
    assert m.provider_ids == {}
    assert m.links == {}
    assert m.duration_s is None


def test_audd_missing_timecode_raises_provider_error(monkeypatch, clip):
    payload = {"status": "success", "result": {
        "artist": "A", "title": "T"}}
    monkeypatch.setattr(providers.requests, "post",
                        lambda *a, **k: FakeResponse(payload))
    with pytest.raises(ProviderError, match="timecode"):
        AudDProvider(api_token="tok").identify(clip)


def test_audd_malformed_timecode_raises_provider_error(monkeypatch, clip):
    payload = {"status": "success", "result": {
        "artist": "A", "title": "T", "timecode": "02:32.5"}}
    monkeypatch.setattr(providers.requests, "post",
                        lambda *a, **k: FakeResponse(payload))
    with pytest.raises(ProviderError, match="malformed"):
        AudDProvider(api_token="tok").identify(clip)


ACR_HIT = {
    "status": {"code": 0, "msg": "Success", "version": "1.0"},
    "metadata": {"music": [{
        "title": "Around the World",
        "artists": [{"name": "Daft Punk"}],
        "score": 95,
        "play_offset_ms": 45000,
        "duration_ms": 428000,
        "sample_begin_time_offset_ms": 1000,
        "db_begin_time_offset_ms": 34000,
        "external_ids": {"isrc": "GBDUW0600746"},
        "external_metadata": {"spotify": {"track": {"id": "spt1"}}},
    }]},
}


def test_acrcloud_signature_string_format():
    from magicat.modules.audio.providers import ACRCloudProvider
    p = ACRCloudProvider(host="h.example.com", access_key="key",
                         access_secret="secret")
    assert p._string_to_sign("12345") == (
        "POST\n/v1/identify\nkey\naudio\n1\n12345")


def test_acrcloud_success(monkeypatch, clip):
    from magicat.modules.audio.providers import ACRCloudProvider
    captured = {}

    def fake_post(url, files=None, data=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        return FakeResponse(ACR_HIT)

    monkeypatch.setattr(providers.requests, "post", fake_post)
    p = ACRCloudProvider(host="h.example.com", access_key="key",
                         access_secret="secret")
    match = p.identify(clip)
    assert captured["url"] == "https://h.example.com/v1/identify"
    assert captured["data"]["access_key"] == "key"
    assert captured["data"]["signature"]          # present and non-empty
    # song position at clip start = (db_begin - sample_begin) / 1000
    assert match.song_offset_s == 33.0
    assert match.title == "Around the World"
    assert match.artist == "Daft Punk"
    assert match.score == 95
    assert match.duration_s == 428.0
    assert match.provider_ids["isrc"] == "GBDUW0600746"
    assert match.provider_ids["spotify"] == "spt1"
    assert match.links["spotify"] == "https://open.spotify.com/track/spt1"


def test_acrcloud_no_match(monkeypatch, clip):
    from magicat.modules.audio.providers import ACRCloudProvider
    payload = {"status": {"code": 1001, "msg": "No result", "version": "1.0"}}
    monkeypatch.setattr(providers.requests, "post",
                        lambda *a, **k: FakeResponse(payload))
    p = ACRCloudProvider(host="h", access_key="k", access_secret="s")
    assert p.identify(clip) is None


def test_acrcloud_error_raises(monkeypatch, clip):
    from magicat.modules.audio.providers import ACRCloudProvider
    payload = {"status": {"code": 3001, "msg": "wrong access key",
                          "version": "1.0"}}
    monkeypatch.setattr(providers.requests, "post",
                        lambda *a, **k: FakeResponse(payload))
    p = ACRCloudProvider(host="h", access_key="k", access_secret="s")
    with pytest.raises(ProviderError, match="3001"):
        p.identify(clip)


def test_acrcloud_negative_offset_clamped(monkeypatch, clip):
    from magicat.modules.audio.providers import ACRCloudProvider
    hit = {"status": {"code": 0, "msg": "Success", "version": "1.0"},
           "metadata": {"music": [{
               "title": "T", "artists": [{"name": "A"}], "score": 80,
               "play_offset_ms": 100,
               "sample_begin_time_offset_ms": 5000,
               "db_begin_time_offset_ms": 2000,
           }]}}
    monkeypatch.setattr(providers.requests, "post",
                        lambda *a, **k: FakeResponse(hit))
    p = ACRCloudProvider(host="h", access_key="k", access_secret="s")
    assert p.identify(clip).song_offset_s == 0.0
