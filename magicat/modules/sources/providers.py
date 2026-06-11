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
        try:
            body = resp.json()["responses"][0]
        except (KeyError, IndexError, ValueError) as exc:
            raise ProviderError(f"GCV malformed response: {exc!r}") from exc
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
