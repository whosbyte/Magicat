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
