# Magicat M5 — CapCut Export & Reverse Video Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** The complete vision: every shot gets source-footage candidate links (reverse image search over its keyframes), and every job exports a CapCut-importable draft — alongside the existing Premiere/Resolve zip, preview, and report.

**Architecture:** Reverse search mirrors the proven music-ID pattern exactly: an `ImageMatch` model + `ReverseImageProvider` protocol + env-keyed implementations (SerpAPI Google Lens primary, Google Cloud Vision WEB_DETECTION secondary) + `providers_from_env()` chain + offline fakes in every test; results map into the manifest's `source_matches` slot (reserved since M1) and surface in the report/UI. The CapCut exporter wraps `pycapcut==0.0.3` (pinned exact — the format is reverse-engineered) behind the existing Exporter contract, with a `SkippedExport` mechanism so the spec's feature flag can disable it without failing the layer. v1 ranking is torch-free (provider order → score → domain dedupe); CLIP re-ranking is a recorded deferral.

**Tech Stack:** requests (both providers — raw HTTP, no SDKs), pycapcut==0.0.3 (CapCut International draft generation; time unit µs), stdlib zipfile/json. (pycapcut is a pure-python wheel; its pymediainfo dependency bundles MediaInfo.dll on Windows — non-Windows runners need the system libmediainfo package.)

**Spec:** `docs/superpowers/specs/2026-06-09-magicat-framework-design.md` §6.6 (reverse search), §7 CapCut row, §13 M5 row.

**Research basis (verified 2026-06-10/11, incl. temp-venv dissection of pycapcut output):**
- **pycapcut 0.0.3** targets CapCut INTERNATIONAL (pyJianYingDraft targets Chinese JianYing — wrong app). API: `DraftFolder(path).create_draft(name, w, h) -> ScriptFile`; `add_track(TrackType.video|audio|text)`; `add_segment(...)`; `save()`. Time in MICROSECONDS; `trange(start, duration)` — second arg is DURATION, not end. Reuse one `VideoMaterial` across segments to share material_id. Local mp3 emits material type `extract_music`. Draft folder = `draft_content.json` + `draft_meta_info.json`; media referenced by ABSOLUTE path; stamps app_version 6.7.0 / version 360000 / new_version 140.0.0 from a bundled template. Its uiautomation auto-export is Windows-only — DO NOT use; we only write the draft folder.
- **SerpAPI Google Lens**: `GET https://serpapi.com/search?engine=google_lens&type=visual_matches&url=<PUBLIC_IMAGE_URL>&api_key=...&hl=en&country=us`. `url` MUST be public — NO upload/base64 path exists (open roadmap issue) → the provider takes an injected `url_resolver(Path) -> str`. Response: `visual_matches[]` with `position`, `title`, `link` (the SOURCE PAGE — exactly what we want), `source`, `thumbnail`, `image`. Errors use real HTTP status codes (401/429/400...) — `raise_for_status()` works; also defensively check `data.get("error")`. Free 250/mo. Env `SERPAPI_KEY`.
- **Bing Visual Search is DEAD** (all Bing Search APIs retired 2025-08-11; Azure "Grounding with Bing" is LLM grounding, not a SERP) — secondary is **Google Cloud Vision WEB_DETECTION**: `POST https://vision.googleapis.com/v1/images:annotate?key=<KEY>` with `{"requests":[{"image":{"content":"<BASE64>"},"features":[{"type":"WEB_DETECTION","maxResults":20}]}]}` — accepts LOCAL bytes directly. Response `webDetection`: `pagesWithMatchingImages[] {url, pageTitle}`, `fullMatchingImages[]`, `partialMatchingImages[]`, `webEntities[] {score, description}`. Per-response `error` object possible → ProviderError. $3.50/1k, first 1k/mo free PER FEATURE. Env `GOOGLE_VISION_API_KEY`.
- **TinEye**: viable third (accepts uploads, backlink provenance) but paid-only — NOT implemented in v1 (recorded).
- **CLIP re-rank skipped** in v1 (torch ~1GB+; rank by provider order → score → registered-domain dedupe; optional-extra pattern later, like Demucs/TransNetV2).

**Environment variables (all optional — layers degrade per the house pattern):**

| Var | Meaning |
|---|---|
| `SERPAPI_KEY` | enables the Lens provider (ALSO requires `MAGICAT_PUBLIC_BASE_URL`) |
| `MAGICAT_PUBLIC_BASE_URL` | public base URL under which `jobs/public/` keyframe copies are reachable (e.g. a tunnel/S3 front) — Lens needs public image URLs |
| `GOOGLE_VISION_API_KEY` | enables the Vision provider (local bytes — works with zero hosting) |
| `MAGICAT_RIS_PROVIDER` | `auto` (default) \| `lens` \| `vision` \| `none` |
| `MAGICAT_CAPCUT_EXPORT` | `1` (default) \| `0` — the spec's feature flag for the reverse-engineered format |

No keys → `source_matches` layer = `skipped` (the spec's "flagged" gate falls out naturally).

---

## File Structure

```
pyproject.toml                      # MODIFY: + pycapcut==0.0.3
tests/conftest.py                   # MODIFY: env-isolation tuple += 5 new vars
magicat/
  core/interfaces.py                # MODIFY: + SkippedExport exception
  core/pipeline.py                  # MODIFY: ANALYZERS += reverse_search;
                                    #         EXPORTERS += capcut_zip; SkippedExport handling
  modules/
    report.py                       # MODIFY: sources section in dict + HTML
    sources/
      __init__.py                   # NEW (empty)
      providers.py                  # NEW: ImageMatch, protocol, Lens, Vision, providers_from_env
      ranking.py                    # NEW: merge + score-sort + domain dedupe (pure)
      analyzer.py                   # NEW: SourceMatchAnalyzer ("reverse_search")
    export/
      capcut.py                     # NEW: CapCutExporter ("capcut_zip", pycapcut-backed)
  server/
    app.py                          # MODIFY: ARTIFACTS += capcut_draft.zip
    static/app.js                   # MODIFY: sources in summary + capcut download link
tests/
  test_sources_providers.py         # NEW
  test_sources_ranking.py           # NEW
  test_sources_analyzer.py          # NEW
  test_export_capcut.py             # NEW
  test_report_sources.py            # NEW
  test_api.py                       # MODIFY: capcut artifact + UI link tests
```

---

### Task 1: Reverse-search providers (Lens + Vision) + ranking

**Files:**
- Modify: `tests/conftest.py`
- Create: `magicat/modules/sources/__init__.py` (empty), `magicat/modules/sources/providers.py`, `magicat/modules/sources/ranking.py`
- Test: `tests/test_sources_providers.py`, `tests/test_sources_ranking.py`

- [ ] **Step 1: Env isolation** — in `tests/conftest.py`, extend the autouse `_isolated_magicat_env` tuple with: `"SERPAPI_KEY", "MAGICAT_PUBLIC_BASE_URL", "GOOGLE_VISION_API_KEY", "MAGICAT_RIS_PROVIDER", "MAGICAT_CAPCUT_EXPORT"`.

- [ ] **Step 2: Write the failing tests** — create `tests/test_sources_providers.py`:

```python
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
```

And create `tests/test_sources_ranking.py`:

```python
# tests/test_sources_ranking.py
from magicat.modules.sources.providers import ImageMatch
from magicat.modules.sources.ranking import domain_of, rank_matches


def m(url, score=0.5, provider="p"):
    return ImageMatch(source_url=url, score=score, provider=provider)


def test_domain_of():
    assert domain_of("https://www.tiktok.com/@u/video/1") == "tiktok.com"
    assert domain_of("https://m.youtube.com/watch?v=1") == "youtube.com"
    assert domain_of("https://blog.example.co/post") == "blog.example.co"
    assert domain_of("not a url") == ""


def test_rank_orders_by_score_desc():
    ranked = rank_matches([m("https://a/1", 0.2), m("https://b/2", 0.9)])
    assert [r.source_url for r in ranked] == ["https://b/2", "https://a/1"]


def test_rank_dedupes_by_domain_keeping_best():
    ranked = rank_matches([
        m("https://www.tiktok.com/@u/video/1", 0.9),
        m("https://www.tiktok.com/@u/video/2", 0.7),   # same domain: dropped
        m("https://youtube.com/watch?v=1", 0.6),
    ])
    assert [r.source_url for r in ranked] == [
        "https://www.tiktok.com/@u/video/1",
        "https://youtube.com/watch?v=1",
    ]


def test_rank_caps_results():
    matches = [m(f"https://site{i}.example/x", 1.0 - i * 0.1)
               for i in range(8)]
    assert len(rank_matches(matches, limit=5)) == 5


def test_rank_empty():
    assert rank_matches([]) == []
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_sources_providers.py tests/test_sources_ranking.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 4: Implement** — create `magicat/modules/sources/providers.py`:

```python
# magicat/modules/sources/providers.py
"""Reverse image search providers behind one protocol (spec section 6.6).

Mirrors the music-ID provider pattern. PRIMARY: SerpAPI Google Lens - the
richest source-page coverage, but it ONLY accepts a PUBLIC image url (no
upload path exists), so the provider takes an injected url_resolver and is
only auto-enabled when MAGICAT_PUBLIC_BASE_URL is configured. SECONDARY:
Google Cloud Vision WEB_DETECTION - accepts local bytes directly (base64),
the practical zero-hosting local provider. Bing Visual Search retired
2025-08-11 (do not implement); TinEye is paid-only (recorded deferral).
"""
from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable

import requests
from pydantic import BaseModel


class ProviderError(RuntimeError):
    """A provider returned an API-level error (auth, quota, bad image...)."""


class ImageMatch(BaseModel):
    source_url: str                 # the SOURCE PAGE link
    title: str | None = None
    source: str | None = None       # site/domain label from the provider
    thumbnail: str | None = None
    score: float = 0.0
    provider: str = ""


@runtime_checkable
class ReverseImageProvider(Protocol):
    name: str

    def search(self, keyframe: Path) -> list[ImageMatch]: ...


UrlResolver = Callable[[Path], str]


class SerpapiLensProvider:
    name = "serpapi_lens"

    def __init__(self, api_key: str, url_resolver: UrlResolver) -> None:
        self.api_key = api_key
        self.resolve = url_resolver

    def search(self, keyframe: Path) -> list[ImageMatch]:
        public_url = self.resolve(keyframe)
        resp = requests.get(
            "https://serpapi.com/search",
            params={"engine": "google_lens", "type": "visual_matches",
                    "url": public_url, "api_key": self.api_key,
                    "hl": "en", "country": "us"},
            timeout=30)
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            raise ProviderError(f"SerpAPI: {data['error']}")
        matches = []
        for item in data.get("visual_matches", []):
            link = item.get("link")
            if not link:
                continue
            position = item.get("position", 100)
            matches.append(ImageMatch(
                source_url=link,
                title=item.get("title"),
                source=item.get("source"),
                thumbnail=item.get("thumbnail"),
                # position 1 -> 1.0, decaying; keeps cross-provider sort sane
                score=max(0.0, 1.0 - (position - 1) * 0.05),
                provider=self.name))
        return matches


class VisionWebProvider:
    name = "gcv_web"

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    def search(self, keyframe: Path) -> list[ImageMatch]:
        content = base64.b64encode(keyframe.read_bytes()).decode("ascii")
        resp = requests.post(
            "https://vision.googleapis.com/v1/images:annotate",
            params={"key": self.api_key},
            json={"requests": [{
                "image": {"content": content},
                "features": [{"type": "WEB_DETECTION", "maxResults": 20}],
            }]},
            timeout=30)
        resp.raise_for_status()
        body = resp.json()["responses"][0]
        if "error" in body:
            raise ProviderError(f"GCV: {body['error']['message']}")
        detection = body.get("webDetection", {})
        matches = []
        for i, page in enumerate(detection.get("pagesWithMatchingImages",
                                               [])):
            url = page.get("url")
            if not url:
                continue
            matches.append(ImageMatch(
                source_url=url,
                title=page.get("pageTitle"),
                score=max(0.0, 0.9 - i * 0.05),  # full-match pages rank high
                provider=self.name))
        return matches


def default_url_resolver(jobs_root: Path | None = None) -> UrlResolver:
    """Copy a keyframe under jobs/public/ and return its public URL.

    Requires MAGICAT_PUBLIC_BASE_URL to point at a host serving that
    directory (tunnel, bucket front, reverse proxy). Local-only setups
    use the Vision provider instead.
    """
    import shutil
    import uuid

    base = os.environ["MAGICAT_PUBLIC_BASE_URL"].rstrip("/")
    # resolve() honors the repo's absolute-path contract (manifests and
    # served dirs must not depend on the process CWD at read time); the
    # public dir is <cwd-at-start>/jobs/public, which MAGICAT_PUBLIC_BASE_URL
    # must front - recorded in Out of Scope
    root = (Path(jobs_root) if jobs_root else Path("jobs")).resolve()

    def resolve(keyframe: Path) -> str:
        public_dir = root / "public"
        public_dir.mkdir(parents=True, exist_ok=True)
        name = f"{uuid.uuid4().hex[:16]}{keyframe.suffix}"
        shutil.copy2(keyframe, public_dir / name)
        return f"{base}/public/{name}"

    return resolve


def providers_from_env() -> list[ReverseImageProvider]:
    """Ordered chain: Lens (needs key + public base url), then Vision."""
    selection = os.environ.get("MAGICAT_RIS_PROVIDER", "auto")
    if selection == "none":
        return []
    if selection not in ("auto", "lens", "vision"):
        raise ValueError(f"unknown MAGICAT_RIS_PROVIDER {selection!r}")

    chain: list[ReverseImageProvider] = []
    serp_key = os.environ.get("SERPAPI_KEY")
    public_base = os.environ.get("MAGICAT_PUBLIC_BASE_URL")
    if selection in ("auto", "lens") and serp_key and public_base:
        chain.append(SerpapiLensProvider(
            api_key=serp_key, url_resolver=default_url_resolver()))
    vision_key = os.environ.get("GOOGLE_VISION_API_KEY")
    if selection in ("auto", "vision") and vision_key:
        chain.append(VisionWebProvider(api_key=vision_key))
    return chain
```

And `magicat/modules/sources/ranking.py`:

```python
# magicat/modules/sources/ranking.py
"""Merge + rank reverse-search matches (v1: torch-free, no CLIP).

Rank by score descending, dedupe by registered domain (one hit per site -
a recap blog posting nine screenshots must not drown the original), cap
the list. CLIP re-ranking is a recorded deferral (optional-extra pattern).
"""
from __future__ import annotations

import urllib.parse

from magicat.modules.sources.providers import ImageMatch

DROP_PREFIXES = ("www.", "m.", "mobile.", "amp.")


def domain_of(url: str) -> str:
    try:
        netloc = urllib.parse.urlparse(url).netloc.lower()
    except ValueError:
        return ""
    if not netloc:
        return ""
    for prefix in DROP_PREFIXES:
        if netloc.startswith(prefix):
            netloc = netloc[len(prefix):]
            break
    return netloc


def rank_matches(matches: list[ImageMatch],
                 limit: int = 5) -> list[ImageMatch]:
    ranked = sorted(matches, key=lambda m: -m.score)
    seen: set[str] = set()
    out: list[ImageMatch] = []
    for match in ranked:
        domain = domain_of(match.source_url)
        if domain and domain in seen:
            continue
        seen.add(domain)
        out.append(match)
        if len(out) >= limit:
            break
    return out
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_sources_providers.py tests/test_sources_ranking.py -v`
Expected: 7 + 5 PASS. Full suite: 192 passed, 2 skipped.

- [ ] **Step 6: Commit**

```bash
git add magicat/modules/sources tests/test_sources_providers.py tests/test_sources_ranking.py tests/conftest.py
git commit -m "feat: reverse image search providers (Lens + Vision) with ranking"
```

---

### Task 2: SourceMatchAnalyzer + pipeline wiring

**Files:**
- Create: `magicat/modules/sources/analyzer.py`
- Modify: `magicat/core/pipeline.py`
- Test: `tests/test_sources_analyzer.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_sources_analyzer.py`:

```python
# tests/test_sources_analyzer.py
from pathlib import Path

import pytest

from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import LayerState, Manifest, Shot, Source
from magicat.modules.sources.analyzer import SourceMatchAnalyzer
from magicat.modules.sources.providers import ImageMatch, ProviderError


def manifest_with_shots(tmp_path) -> Manifest:
    keyframes = []
    for i in range(2):
        kf = tmp_path / f"kf{i}.jpg"
        kf.write_bytes(b"\xff\xd8fake")
        keyframes.append(str(kf))
    return Manifest(
        job_id="j", source=Source(file="x.mp4", duration=6.0),
        shots=[Shot(id="shot_000", start=0.0, end=3.0,
                    keyframes=[keyframes[0]]),
               Shot(id="shot_001", start=3.0, end=6.0,
                    keyframes=[keyframes[1]])])


class OneHitProvider:
    name = "fake"

    def __init__(self):
        self.calls = []

    def search(self, keyframe: Path):
        self.calls.append(str(keyframe))
        return [ImageMatch(source_url="https://www.tiktok.com/@u/video/9",
                           title="Original", score=0.9, provider="fake")]


def test_no_providers_skips_layer(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "job")
    analyzer = SourceMatchAnalyzer()
    monkeypatch.setattr(analyzer, "provider_factory", lambda: [])
    patch = analyzer.run(manifest_with_shots(tmp_path), ws)
    assert patch == {"layers_status": {"source_matches": "skipped"}}


def test_matches_mapped_per_shot(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "job")
    analyzer = SourceMatchAnalyzer()
    provider = OneHitProvider()
    monkeypatch.setattr(analyzer, "provider_factory", lambda: [provider])
    m = manifest_with_shots(tmp_path)
    patch = analyzer.run(m, ws)
    sm = patch["source_matches"]
    assert len(sm) == 2
    assert sm[0]["shot_id"] == "shot_000"
    cand = sm[0]["candidates"][0]
    assert cand["url"] == "https://www.tiktok.com/@u/video/9"
    assert cand["title"] == "Original"
    assert cand["score"] == 0.9
    assert patch["layers_status"] == {"source_matches": "ok"}
    # one search per shot (middle keyframe)
    assert len(provider.calls) == 2
    m2 = apply_patch(m, patch)          # validates against the schema
    assert m2.source_matches[0].candidates[0].url.endswith("/9")


def test_provider_errors_degrade_per_shot(tmp_path, monkeypatch):
    class FlakyProvider:
        name = "flaky"

        def __init__(self):
            self.n = 0

        def search(self, keyframe):
            self.n += 1
            if self.n == 1:
                raise ProviderError("quota")
            return [ImageMatch(source_url="https://a/b", score=0.5,
                               provider="flaky")]

    ws = Workspace(tmp_path / "job")
    analyzer = SourceMatchAnalyzer()
    monkeypatch.setattr(analyzer, "provider_factory",
                        lambda: [FlakyProvider()])
    patch = analyzer.run(manifest_with_shots(tmp_path), ws)
    sm = patch["source_matches"]
    assert sm[0]["candidates"] == []          # shot 0 errored -> empty
    assert len(sm[1]["candidates"]) == 1      # shot 1 fine
    assert patch["layers_status"] == {"source_matches": "ok"}


def test_multi_provider_results_merged_and_ranked(tmp_path, monkeypatch):
    class P1:
        name = "p1"

        def search(self, keyframe):
            return [ImageMatch(source_url="https://www.tiktok.com/@u/v/1",
                               score=0.9, provider="p1")]

    class P2:
        name = "p2"

        def search(self, keyframe):
            return [
                ImageMatch(source_url="https://tiktok.com/@u/v/dup",
                           score=0.8, provider="p2"),     # same domain: deduped
                ImageMatch(source_url="https://vimeo.com/123",
                           score=0.7, provider="p2"),
            ]

    ws = Workspace(tmp_path / "job")
    analyzer = SourceMatchAnalyzer()
    monkeypatch.setattr(analyzer, "provider_factory", lambda: [P1(), P2()])
    patch = analyzer.run(manifest_with_shots(tmp_path), ws)
    urls = [c["url"] for c in patch["source_matches"][0]["candidates"]]
    assert urls == ["https://www.tiktok.com/@u/v/1", "https://vimeo.com/123"]


def test_pipeline_skips_without_keys(fixture_video, tmp_path):
    from magicat.core.pipeline import run_job
    manifest = run_job(str(fixture_video), tmp_path / "job")
    assert manifest.layers_status["source_matches"] == LayerState.SKIPPED
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_sources_analyzer.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement** — create `magicat/modules/sources/analyzer.py`:

```python
# magicat/modules/sources/analyzer.py
"""Reverse search per shot: middle keyframe -> provider chain -> ranked
source candidates into manifest.source_matches (slot reserved since M1)."""
from __future__ import annotations

import logging
from pathlib import Path

from magicat.core.registry import register_analyzer
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest
from magicat.modules.sources.providers import (
    ProviderError,
    providers_from_env,
)
from magicat.modules.sources.ranking import rank_matches

log = logging.getLogger(__name__)

CANDIDATES_PER_SHOT = 5


@register_analyzer
class SourceMatchAnalyzer:
    name = "reverse_search"
    layer = "source_matches"
    needs_gpu = False
    provider_factory = staticmethod(providers_from_env)  # injectable

    def run(self, manifest: Manifest, ws: Workspace) -> dict:
        providers = self.provider_factory()
        if not providers:
            return {"layers_status": {"source_matches": "skipped"}}

        source_matches = []
        for shot in manifest.shots:
            matches = []
            if shot.keyframes:
                # the middle keyframe is the shot's most representative frame
                keyframe = Path(shot.keyframes[len(shot.keyframes) // 2])
                for provider in providers:
                    try:
                        matches.extend(provider.search(keyframe))
                    except (ProviderError, OSError) as exc:
                        log.warning("reverse search failed for %s via %s: %s",
                                    shot.id, provider.name, exc)
            ranked = rank_matches(matches, limit=CANDIDATES_PER_SHOT)
            source_matches.append({
                "shot_id": shot.id,
                "candidates": [{
                    "url": m.source_url,
                    "title": m.title,
                    "thumbnail": m.thumbnail,
                    "score": m.score,
                } for m in ranked],
            })
        return {"source_matches": source_matches,
                "layers_status": {"source_matches": "ok"}}
```

- [ ] **Step 4: Wire into the pipeline** — in `magicat/core/pipeline.py`:

```python
ANALYZERS = ["cut_detection", "audio_analysis", "caption_analysis",
             "reverse_search", "music_acquisition"]
```

and add `import magicat.modules.sources.analyzer  # noqa: F401` as the LAST entry of `load_builtin_modules` (it sorts after report).

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_sources_analyzer.py tests/test_pipeline.py tests/test_progress_events.py -v`
Expected: all PASS — NOTE: `test_progress_events.py::test_progress_callback_receives_stage_events` asserts a fixed event sequence; it does NOT assert reverse_search, and its "every start resolves" invariant accommodates the new analyzer automatically (start → skipped). Full suite: 197 passed, 2 skipped.

- [ ] **Step 6: Commit**

```bash
git add magicat/modules/sources/analyzer.py magicat/core/pipeline.py tests/test_sources_analyzer.py
git commit -m "feat: reverse-search analyzer wired into pipeline"
```

---

### Task 3: Sources in report + UI

**Files:**
- Modify: `magicat/modules/report.py`, `magicat/server/static/app.js`
- Test: `tests/test_report_sources.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_report_sources.py`:

```python
# tests/test_report_sources.py
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest, Shot, Source
from magicat.modules.report import ReportExporter, build_report


def manifest_with_sources() -> Manifest:
    return Manifest(
        job_id="j", source=Source(file="x.mp4", duration=6.0),
        shots=[Shot(id="shot_000", start=0.0, end=3.0)],
        source_matches=[{
            "shot_id": "shot_000",
            "candidates": [
                {"url": "https://www.tiktok.com/@u/video/9",
                 "title": "Original clip", "score": 0.9},
                {"url": "https://vimeo.com/123", "title": "Mirror",
                 "score": 0.5},
            ],
        }])


def test_build_report_includes_sources():
    report = build_report(manifest_with_sources())
    sources = report["sources"]
    assert sources["searched"] is True
    assert sources["shots"][0]["shot_id"] == "shot_000"
    assert sources["shots"][0]["candidates"][0]["url"] == \
        "https://www.tiktok.com/@u/video/9"


def test_build_report_sources_absent():
    report = build_report(Manifest(job_id="j"))
    assert report["sources"]["searched"] is False
    assert report["sources"]["shots"] == []


def test_html_report_renders_source_links(tmp_path):
    ws = Workspace(tmp_path / "job")
    html = ReportExporter().export(manifest_with_sources(), ws) \
        .read_text(encoding="utf-8")
    assert "Source footage" in html
    assert "https://www.tiktok.com/@u/video/9" in html
    assert "Original clip" in html


def test_html_report_no_sources_section_message(tmp_path):
    ws = Workspace(tmp_path / "job")
    html = ReportExporter().export(Manifest(job_id="j"), ws) \
        .read_text(encoding="utf-8")
    assert "No source search performed" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_report_sources.py -v`
Expected: FAIL — KeyError 'sources'.

- [ ] **Step 3: Implement** — in `magicat/modules/report.py`:

(a) In `build_report`, add a `"sources"` key to the returned dict (after `"captions"`):

```python
        "sources": {
            "searched": bool(manifest.source_matches),
            "shots": [{
                "shot_id": sm.shot_id,
                "candidates": [{
                    "url": c.url, "title": c.title, "score": c.score,
                } for c in sm.candidates],
            } for sm in manifest.source_matches],
        },
```

(b) In `_PAGE`, add a section between Music and Captions:

```html
<h2>Source footage</h2>$sources_html
```

(c) In `_render_html`, build `sources_html` and pass it to `substitute`:

```python
    sources = report["sources"]
    if sources["searched"]:
        rows = []
        for shot in sources["shots"]:
            links = " ".join(
                f'<a class="tag" href="{_esc(c["url"])}">'
                f'{_esc(c["title"] or c["url"])}</a>'
                for c in shot["candidates"]
                if str(c["url"]).lower().startswith(("http://", "https://")))
            rows.append(f"<tr><td>{_esc(shot['shot_id'])}</td>"
                        f"<td>{links or '-'}</td></tr>")
        sources_html = f"<table>{''.join(rows)}</table>"
    else:
        sources_html = ("<p>No source search performed (configure a "
                        "reverse-search provider key).</p>")
```

⚠️ CRITICAL WIRING (a missing kwarg here KeyErrors EVERY report render, breaking the whole suite): the existing `_PAGE.substitute(...)` call at the end of `_render_html` must ALSO gain the new kwarg — add this line alongside the existing `music_html=music_html,` / `captions_html=captions_html,` arguments:

```python
        sources_html=sources_html,
```

(d) In `magicat/server/static/app.js`, in `finish()`, after the captions line append:

```javascript
  const sources = report.sources || {};
  if (sources.searched && (sources.shots || []).length) {
    const links = sources.shots.flatMap((s) => s.candidates || [])
      .slice(0, 5)
      .map((c) => `<a href="${esc(c.url)}" target="_blank">${esc(c.title || c.url)}</a>`)
      .join(" · ");
    html += `<p>Source candidates: ${links}</p>`;
  }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_report_sources.py tests/test_report.py tests/test_api.py -v`
Expected: all PASS (existing report/API tests unaffected — the new dict key is additive). Full suite: 201 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat/modules/report.py magicat/server/static/app.js tests/test_report_sources.py
git commit -m "feat: source-footage links in report and UI"
```

---

### Task 4: SkippedExport mechanism + CapCut draft exporter

**Files:**
- Modify: `pyproject.toml`, `magicat/core/interfaces.py`, `magicat/core/pipeline.py`
- Create: `magicat/modules/export/capcut.py`
- Test: `tests/test_export_capcut.py`

- [ ] **Step 1: Add the dependency** — in `pyproject.toml` `dependencies`, add `"pycapcut==0.0.3",` (PINNED EXACT: the draft format is reverse-engineered; any bump must re-run the format snapshot tests). Run `.venv/Scripts/python -m pip install -e .[dev]`.

Then VERIFY the native MediaInfo layer is functional (pycapcut's Video/AudioMaterial probe media via pymediainfo, a ctypes wrapper — the Windows wheel bundles MediaInfo.dll, so this passes here; on Linux/macOS the system libmediainfo package would be required):

```
.venv/Scripts/python -c "import pymediainfo; print(pymediainfo.MediaInfo.can_parse())"
```

Expected: `True`. If False, STOP and report BLOCKED.

- [ ] **Step 2: Write the failing tests** — create `tests/test_export_capcut.py`:

```python
# tests/test_export_capcut.py
import json
import zipfile

import pytest

from magicat.core.interfaces import SkippedExport
from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import Manifest, Shot, Source
from magicat.modules.export.capcut import MICROS, CapCutExporter
from tests.conftest import run_ffmpeg


def capcut_manifest(fixture_video, tmp_path, music: bool = False):
    audio = {}
    if music:
        mp3 = tmp_path / "music.mp3"
        run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=660:duration=10",
                    "-c:a", "libmp3lame", str(mp3)])
        audio = {"music": {
            "detected": True, "title": "Song", "artist": "Artist",
            "timeline_offset": 2.0,
            "song_segment": {"start_in_song": 0.0, "duration": 4.0},
            "acquisition": {"status": "acquired", "file": str(mp3),
                            "links": {}},
        }}
    return Manifest(
        job_id="j",
        source=Source(file=str(fixture_video), fps=30.0,
                      resolution="320x640", duration=6.0),
        shots=[Shot(id="shot_000", start=0.0, end=2.0),
               Shot(id="shot_001", start=2.0, end=6.0)],
        captions={"segments": [{
            "text": "HELLO WORLD", "t_start": 1.0, "t_end": 3.0,
            "style": {"fill": "#FDFDFD", "alignment": "center"},
        }]},
        audio=audio,
    )


def load_draft(zip_path):
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        content = next(n for n in names if n.endswith("draft_content.json"))
        return json.loads(zf.read(content)), names


def test_exporter_disabled_by_flag_raises_skipped(fixture_video, tmp_path,
                                                  monkeypatch):
    monkeypatch.setenv("MAGICAT_CAPCUT_EXPORT", "0")
    ws = Workspace(tmp_path / "job")
    with pytest.raises(SkippedExport):
        CapCutExporter().export(capcut_manifest(fixture_video, tmp_path), ws)


def test_draft_zip_structure(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    out = CapCutExporter().export(capcut_manifest(fixture_video, tmp_path),
                                  ws)
    assert out.name == "capcut_draft.zip"
    draft, names = load_draft(out)
    assert any(n.endswith("draft_meta_info.json") for n in names)
    assert any(n == "CAPCUT_INSTRUCTIONS.txt" for n in names)
    # FORMAT SNAPSHOT (pycapcut 0.0.3 / CapCut int'l): pin the load-bearing
    # structure so a dependency bump that changes the format fails loudly
    track_types = [t["type"] for t in draft["tracks"]]
    assert "video" in track_types
    assert "text" in track_types
    video_track = next(t for t in draft["tracks"] if t["type"] == "video")
    segs = video_track["segments"]
    assert len(segs) == 2
    # microseconds + duration semantics (trange is start+DURATION)
    assert segs[0]["target_timerange"]["start"] == 0
    assert segs[0]["target_timerange"]["duration"] == 2 * MICROS
    assert segs[1]["target_timerange"]["start"] == 2 * MICROS
    assert segs[1]["target_timerange"]["duration"] == 4 * MICROS
    assert segs[0]["source_timerange"]["start"] == 0
    assert segs[1]["source_timerange"]["start"] == 2 * MICROS
    # one shared source material
    material_ids = {s["material_id"] for s in segs}
    assert len(material_ids) == 1
    assert len(draft["materials"]["videos"]) == 1


def test_draft_includes_music_at_offset(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    out = CapCutExporter().export(
        capcut_manifest(fixture_video, tmp_path, music=True), ws)
    draft, _ = load_draft(out)
    audio_track = next(t for t in draft["tracks"] if t["type"] == "audio")
    seg = audio_track["segments"][0]
    assert seg["target_timerange"]["start"] == 2 * MICROS
    assert seg["target_timerange"]["duration"] == 4 * MICROS


def test_draft_includes_caption_text(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    out = CapCutExporter().export(capcut_manifest(fixture_video, tmp_path),
                                  ws)
    draft, _ = load_draft(out)
    texts = json.dumps(draft["materials"].get("texts", []))
    assert "HELLO WORLD" in texts


def test_no_shots_raises_value_error(tmp_path):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file="x.mp4"))
    with pytest.raises(ValueError, match="no shots"):
        CapCutExporter().export(m, ws)


def test_pipeline_capcut_layer(fixture_video, tmp_path):
    from magicat.core.pipeline import run_job
    manifest = run_job(str(fixture_video), tmp_path / "job")
    assert manifest.layers_status["capcut_zip"].value == "ok"
    assert (tmp_path / "job" / "exports" / "capcut_draft.zip").is_file()


def test_pipeline_capcut_flag_off_marks_skipped(fixture_video, tmp_path,
                                                monkeypatch):
    monkeypatch.setenv("MAGICAT_CAPCUT_EXPORT", "0")
    from magicat.core.pipeline import run_job
    manifest = run_job(str(fixture_video), tmp_path / "job")
    assert manifest.layers_status["capcut_zip"].value == "skipped"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_export_capcut.py -v`
Expected: FAIL — ImportError (SkippedExport / capcut module).

- [ ] **Step 4: Implement.**

(a) `magicat/core/interfaces.py` — add near the top (after ManifestPatch import):

```python
class SkippedExport(Exception):
    """An exporter declining to run (feature flag off) - NOT a failure."""
```

(b) `magicat/core/pipeline.py` — import it (`from magicat.core.interfaces import SkippedExport`) and in the exporter loop add a branch BEFORE the generic except:

```python
        except SkippedExport:
            log.info("exporter %s skipped (disabled)", fmt)
            manifest = apply_patch(
                manifest, {"layers_status": {fmt: "skipped"}})
            progress(fmt, "skipped")
        except Exception:
            ...existing failed handling unchanged...
```

Also: `EXPORTERS = ["preview_mp4", "report_html", "premiere_resolve_zip", "capcut_zip"]` and add `import magicat.modules.export.capcut  # noqa: F401` to `load_builtin_modules` (alphabetical, before export.package).

(c) Create `magicat/modules/export/capcut.py`:

```python
# magicat/modules/export/capcut.py
"""CapCut draft exporter (spec section 7 CapCut row - feature-flagged).

Wraps pycapcut (PINNED ==0.0.3): generates a CapCut-International draft
folder (draft_content.json + draft_meta_info.json, microsecond times,
absolute media paths) and zips it. The format is reverse-engineered -
the test suite pins the load-bearing structure so a dependency bump that
silently changes it fails in CI, not in a user's CapCut.

MAGICAT_CAPCUT_EXPORT=0 disables via SkippedExport (layer 'skipped').
Media is referenced by absolute path into the JOB's media dir (CapCut
drafts always use absolute paths); the zip carries instructions.
"""
from __future__ import annotations

import os
import shutil
import zipfile
from pathlib import Path

from magicat.core.interfaces import SkippedExport
from magicat.core.registry import register_exporter
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest

MICROS = 1_000_000

INSTRUCTIONS = """Magicat CapCut draft
=====================

1. Extract this zip. Inside is one folder: the draft.
2. Move that folder into CapCut's local drafts directory, e.g.
   C:\\Users\\<you>\\AppData\\Local\\CapCut\\User Data\\Projects\\com.lveditor.draft\\
3. Restart CapCut - the project appears in your local drafts.
4. Media is referenced by ABSOLUTE path from the Magicat job folder -
   keep that folder, or relink the clips inside CapCut if you move it.

The draft format is reverse-engineered (pycapcut, CapCut International);
if a CapCut update rejects it, re-export with a newer Magicat.
"""


@register_exporter
class CapCutExporter:
    format = "capcut_zip"

    def export(self, manifest: Manifest, ws: Workspace) -> Path:
        if os.environ.get("MAGICAT_CAPCUT_EXPORT", "1") == "0":
            raise SkippedExport("MAGICAT_CAPCUT_EXPORT=0")
        if not manifest.shots:
            raise ValueError("no shots in manifest - nothing to export")

        import pycapcut as cc

        staging = ws.exports_dir / "capcut_staging"
        if staging.is_dir():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)

        width, height = 1080, 1920
        if manifest.source.resolution and "x" in manifest.source.resolution:
            width, height = (int(v)
                             for v in manifest.source.resolution.split("x"))

        draft_name = f"magicat_{manifest.job_id[:8]}"
        folder = cc.DraftFolder(str(staging))
        script = folder.create_draft(draft_name, width, height)

        script.add_track(cc.TrackType.video)
        source_material = cc.VideoMaterial(str(manifest.source.file))
        cursor = 0
        for shot in manifest.shots:
            start_us = round(shot.start * MICROS)
            duration_us = round((shot.end - shot.start) * MICROS)
            # pycapcut raises ValueError when source end exceeds the
            # MediaInfo-reported duration (which can run ~10ms short of
            # ffprobe's container duration) - clamp the tail shot
            duration_us = min(duration_us,
                              max(0, source_material.duration - start_us))
            if duration_us <= 0:
                continue   # shot starts past the probed media end: skip
            segment = cc.VideoSegment(
                source_material,
                target_timerange=cc.trange(cursor, duration_us),
                source_timerange=cc.trange(start_us, duration_us))
            script.add_segment(segment)
            cursor += duration_us

        music = manifest.audio.music
        if music.detected and music.acquisition.file \
                and Path(music.acquisition.file).is_file():
            script.add_track(cc.TrackType.audio)
            offset_us = round(music.timeline_offset * MICROS)
            duration_us = round(music.song_segment.duration * MICROS)
            audio_segment = cc.AudioSegment(
                cc.AudioMaterial(str(music.acquisition.file)),
                target_timerange=cc.trange(offset_us, duration_us),
                source_timerange=cc.trange(0, duration_us))
            script.add_segment(audio_segment)

        if manifest.captions.segments:
            script.add_track(cc.TrackType.text)
            for seg in manifest.captions.segments:
                start_us = round(seg.t_start * MICROS)
                duration_us = round((seg.t_end - seg.t_start) * MICROS)
                text_segment = cc.TextSegment(
                    seg.text, cc.trange(start_us, duration_us))
                script.add_segment(text_segment)

        script.save()

        out = ws.exports_dir / "capcut_draft.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("CAPCUT_INSTRUCTIONS.txt", INSTRUCTIONS)
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(staging).as_posix())
        shutil.rmtree(staging)
        return out
```

API NOTE for the implementer: the pycapcut surface above (`DraftFolder.create_draft`, `TrackType`, `VideoMaterial`, `VideoSegment(material, target_timerange=, source_timerange=)`, `AudioMaterial`/`AudioSegment`, `TextSegment(text, trange)`, `script.add_segment`, `script.save`) follows the research dissection of pycapcut 0.0.3 — if a call signature differs in the installed package (positional vs keyword, segment-add per track), check `.venv/Lib/site-packages/pycapcut/` sources and adapt MINIMALLY, reporting every adaptation. The TESTS pin the OUTPUT format, which is what matters.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_export_capcut.py tests/test_pipeline.py tests/test_progress_events.py -v`
Expected: all PASS (the progress-events invariant absorbs the new exporter automatically). Full suite: 208 passed, 2 skipped.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml magicat/core/interfaces.py magicat/core/pipeline.py magicat/modules/export/capcut.py tests/test_export_capcut.py
git commit -m "feat: flag-gated CapCut draft exporter with format snapshot tests"
```

---

### Task 5: CapCut artifact in the server + UI

**Files:**
- Modify: `magicat/server/app.py`, `magicat/server/static/app.js`
- Test: `tests/test_api.py` (append 2 tests)

- [ ] **Step 1: Write the failing tests.** First restructure the existing `client` fixture in `tests/test_api.py` so tests can reach the store (REPLACE the current `client` fixture with this pair — every existing test keeps working since `client` still yields the TestClient):

```python
@pytest.fixture()
def client_and_store(tmp_path, monkeypatch):
    monkeypatch.delenv("MAGICAT_API_KEY", raising=False)
    store = JobStore(tmp_path / "jobs.db")
    runner = InlineRunner(store, fake_pipeline)
    app = create_app(store=store, runner=runner,
                     jobs_root=tmp_path / "jobs")
    return TestClient(app), store


@pytest.fixture()
def client(client_and_store):
    return client_and_store[0]
```

Then append the two new tests:

```python
def test_capcut_artifact_allowlisted_but_absent_404(client):
    job_id = client.post("/api/jobs",
                         json={"url": "https://x/v/1"}).json()["job_id"]
    # fake pipeline doesn't create it: allowlisted name, absent file -> 404
    r = client.get(f"/api/jobs/{job_id}/artifacts/capcut_draft.zip")
    assert r.status_code == 404


def test_capcut_artifact_served_when_present(client_and_store):
    client, store = client_and_store
    job_id = client.post("/api/jobs",
                         json={"url": "https://x/v/1"}).json()["job_id"]
    job = store.get_job(job_id)
    from pathlib import Path
    exports = Path(job.workdir) / "exports"
    exports.mkdir(parents=True, exist_ok=True)
    (exports / "capcut_draft.zip").write_bytes(b"PK\x05\x06" + b"\x00" * 18)
    r = client.get(f"/api/jobs/{job_id}/artifacts/capcut_draft.zip")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/zip"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_api.py -v`
Expected: both new tests FAIL (capcut_draft.zip not in the ARTIFACTS allowlist → 404 in both).

- [ ] **Step 3: Implement.**

(a) `magicat/server/app.py` — add to the ARTIFACTS dict:

```python
    "capcut_draft.zip": ("exports/capcut_draft.zip", "application/zip"),
```

(b) `magicat/server/static/app.js` — in `finish()`, add a CapCut link to the downloads innerHTML (after the Premiere/Resolve link):

```javascript
     <a href="/api/jobs/${jobId}/artifacts/capcut_draft.zip" download>CapCut draft</a>
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_api.py -v`
Expected: 15 PASS. Full suite: 210 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat/server/app.py magicat/server/static/app.js tests/test_api.py
git commit -m "feat: CapCut draft artifact in API allowlist and UI"
```

---

### Task 6: README + e2e + ship

**Files:**
- Modify: `README.md`, `tests/test_api_e2e.py` (one assertion)
- Test: full suite + smoke

- [ ] **Step 1: Extend the e2e** — in `tests/test_api_e2e.py::test_full_pipeline_through_api`, add `"capcut_draft.zip"` to the artifact-download loop tuple (now 5 artifacts).

- [ ] **Step 2: Run** `.venv/Scripts/python -m pytest tests/test_api_e2e.py -v` — expect 2 PASS (the real pipeline now also produces the CapCut zip).

- [ ] **Step 3: README.** Replace the Status line with:

```markdown
**Status:** M5 — the complete vision: cuts, music ID + acquisition,
captions + fonts, source-footage search, Premiere/Resolve AND CapCut
export, web service with live progress.
```

Append:

```markdown
## Source search & CapCut (M5)

Reverse video search (per-shot source-footage links) activates with a
provider key: `$env:GOOGLE_VISION_API_KEY` (works locally - sends image
bytes) or `$env:SERPAPI_KEY` + `$env:MAGICAT_PUBLIC_BASE_URL` (Google
Lens - needs publicly reachable keyframes). Without keys the layer skips.

Every job also exports `capcut_draft.zip` - a CapCut-International draft
folder (extract into CapCut's drafts directory; see the bundled
instructions). The format is reverse-engineered and pinned to
pycapcut 0.0.3 with snapshot tests; disable with
`$env:MAGICAT_CAPCUT_EXPORT = "0"` if a CapCut update rejects drafts.
```

- [ ] **Step 4: Full suite + smoke.** Run `.venv/Scripts/python -m pytest` — expect 210 passed, 2 skipped. Smoke: generate a 4 s clip (the M4 plan's ffmpeg command), `.venv/Scripts/magicat run smoke.mp4`, verify the summary shows `source_matches: skipped` (no keys), `capcut_zip: ok`, and the workdir's exports contain capcut_draft.zip whose draft_content.json parses. Delete smoke artifacts.

- [ ] **Step 5: Commit**

```bash
git add README.md tests/test_api_e2e.py
git commit -m "feat: M5 README and e2e capcut artifact coverage"
```

---

## Out of Scope (M5) — deliberate deferrals

- CLIP re-ranking of source candidates (torch ~1GB; v1 ranks by provider order → score → domain dedupe; add later via the optional-extra pattern like Demucs/TransNetV2).
- TinEye third provider (paid-only, no free tier; exact-dupe/backlink provenance — add if a customer pays for it).
- A built-in public keyframe host for the Lens provider (the `url_resolver` seam + `MAGICAT_PUBLIC_BASE_URL` covers tunnels/buckets; a first-party uploader is cloud-deployment work).
- Frame-level reverse VIDEO search (per-shot keyframes are the spec's design; full-video fingerprint search is a different product).
- SSRF egress controls for provider/yt-dlp fetches (recorded cloud-deployment blocker — carried from M4).
- Real-API integration tests for Lens/Vision (manual, key-gated; same policy as AudD/ACRCloud).

## Self-Review Notes (already applied)

- **Spec coverage:** §6.6 → Tasks 1/2/3 (providers per the research-corrected landscape: Lens + Vision, Bing dead, CLIP deferred — all recorded); §7 CapCut row → Task 4 (feature-flagged via MAGICAT_CAPCUT_EXPORT + SkippedExport, version-pinned ==0.0.3, format snapshot tests — exactly the spec's risk treatment); §13 M5 complete-vision row → Tasks 5/6 surface both in the product.
- **Module contract:** sources/ imports manifest+core+own package; export/capcut.py imports core+manifest+pycapcut; report/app/static changes are additive.
- **Type consistency:** ImageMatch (T1) → analyzer mapping to SourceCandidate fields url/title/thumbnail/score (T2) matches the M1 schema exactly; SkippedExport defined in interfaces (T4) caught by pipeline (T4); MICROS exported for tests.
- **Counts traced:** 180→192 (T1, +12) →197 (T2, +5) →201 (T3, +4) →208 (T4, +7) →210 (T5, +2) →210 (T6, +0 — extends an existing e2e assertion); skips stay 2.


