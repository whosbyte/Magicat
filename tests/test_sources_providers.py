# tests/test_sources_providers.py
from pathlib import Path

import pytest

from magicat.modules.sources import providers
from magicat.modules.sources.providers import (
    ImageMatch,
    ProviderError,
    SerpapiLensProvider,
    VisionWebProvider,
    providers_from_env,
)


class FakeResponse:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self._payload


@pytest.fixture()
def keyframe(tmp_path) -> Path:
    p = tmp_path / "kf.jpg"
    p.write_bytes(b"\xff\xd8fakejpg")
    return p


LENS_HIT = {"visual_matches": [
    {"position": 1, "title": "Original clip",
     "link": "https://www.tiktok.com/@creator/video/123",
     "source": "tiktok.com",
     "thumbnail": "https://t/1.jpg"},
    {"position": 2, "title": "Repost",
     "link": "https://youtube.com/watch?v=abc",
     "source": "youtube.com",
     "thumbnail": "https://t/2.jpg"},
]}


def test_lens_provider_parses_matches(monkeypatch, keyframe):
    captured = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return FakeResponse(LENS_HIT)

    monkeypatch.setattr(providers.requests, "get", fake_get)
    provider = SerpapiLensProvider(
        api_key="k", url_resolver=lambda p: f"https://pub.example/{p.name}")
    matches = provider.search(keyframe)
    assert captured["url"] == "https://serpapi.com/search"
    assert captured["params"]["engine"] == "google_lens"
    assert captured["params"]["url"] == "https://pub.example/kf.jpg"
    assert len(matches) == 2
    first = matches[0]
    assert first.source_url == "https://www.tiktok.com/@creator/video/123"
    assert first.title == "Original clip"
    assert first.provider == "serpapi_lens"
    assert matches[0].score > matches[1].score   # position 1 ranks higher


def test_lens_provider_error_body_raises(monkeypatch, keyframe):
    monkeypatch.setattr(
        providers.requests, "get",
        lambda *a, **k: FakeResponse({"error": "Invalid API key"}))
    provider = SerpapiLensProvider(api_key="k",
                                   url_resolver=lambda p: "https://x/y.jpg")
    with pytest.raises(ProviderError, match="Invalid API key"):
        provider.search(keyframe)


VISION_HIT = {"responses": [{"webDetection": {
    "pagesWithMatchingImages": [
        {"url": "https://blog.example/post", "pageTitle": "A post"},
        {"url": "https://www.tiktok.com/@creator/video/123",
         "pageTitle": "the original"},
    ],
    "webEntities": [{"score": 1.2, "description": "Dancing"}],
}}]}


def test_vision_provider_parses_pages(monkeypatch, keyframe):
    captured = {}

    def fake_post(url, params=None, json=None, timeout=None):
        captured["params"] = params
        captured["json"] = json
        return FakeResponse(VISION_HIT)

    monkeypatch.setattr(providers.requests, "post", fake_post)
    matches = VisionWebProvider(api_key="vk").search(keyframe)
    assert captured["params"] == {"key": "vk"}
    feature = captured["json"]["requests"][0]["features"][0]
    assert feature["type"] == "WEB_DETECTION"
    assert captured["json"]["requests"][0]["image"]["content"]  # base64 set
    assert len(matches) == 2
    assert matches[0].source_url == "https://blog.example/post"
    assert matches[0].provider == "gcv_web"


def test_vision_provider_response_error_raises(monkeypatch, keyframe):
    payload = {"responses": [{"error": {"code": 403,
                                        "message": "quota exceeded"}}]}
    monkeypatch.setattr(providers.requests, "post",
                        lambda *a, **k: FakeResponse(payload))
    with pytest.raises(ProviderError, match="quota exceeded"):
        VisionWebProvider(api_key="vk").search(keyframe)


def test_vision_provider_malformed_response_raises(monkeypatch, keyframe):
    monkeypatch.setattr(providers.requests, "post",
                        lambda *a, **k: FakeResponse({"unexpected": True}))
    with pytest.raises(ProviderError, match="malformed"):
        VisionWebProvider(api_key="vk").search(keyframe)


def test_vision_provider_empty_detection_returns_no_matches(monkeypatch,
                                                            keyframe):
    monkeypatch.setattr(
        providers.requests, "post",
        lambda *a, **k: FakeResponse({"responses": [{}]}))
    assert VisionWebProvider(api_key="vk").search(keyframe) == []


def test_providers_from_env(monkeypatch):
    assert providers_from_env() == []        # env cleared by autouse fixture

    # vision alone: works with local bytes, no hosting needed
    monkeypatch.setenv("GOOGLE_VISION_API_KEY", "vk")
    chain = providers_from_env()
    assert [p.name for p in chain] == ["gcv_web"]

    # lens requires BOTH the key and a public base url
    monkeypatch.setenv("SERPAPI_KEY", "sk")
    assert [p.name for p in providers_from_env()] == ["gcv_web"]
    monkeypatch.setenv("MAGICAT_PUBLIC_BASE_URL", "https://pub.example")
    assert [p.name for p in providers_from_env()] == ["serpapi_lens",
                                                      "gcv_web"]

    monkeypatch.setenv("MAGICAT_RIS_PROVIDER", "vision")
    assert [p.name for p in providers_from_env()] == ["gcv_web"]
    monkeypatch.setenv("MAGICAT_RIS_PROVIDER", "none")
    assert providers_from_env() == []
    monkeypatch.setenv("MAGICAT_RIS_PROVIDER", "bogus")
    with pytest.raises(ValueError):
        providers_from_env()


def test_image_match_defaults():
    m = ImageMatch(source_url="https://x/y")
    assert m.score == 0.0
    assert m.provider == ""
    assert m.title is None
