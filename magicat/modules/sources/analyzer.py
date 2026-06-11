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
        any_success = False
        any_attempt = False
        for shot in manifest.shots:
            matches = []
            if shot.keyframes:
                # the middle keyframe is the shot's most representative frame
                keyframe = Path(shot.keyframes[len(shot.keyframes) // 2])
                for provider in providers:
                    any_attempt = True
                    try:
                        matches.extend(provider.search(keyframe))
                        # success = the CALL returned (even an empty list);
                        # only ERRORS count as a dead provider
                        any_success = True
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

        if any_attempt and not any_success:
            # every call errored (bad key, quota): report a failure, not a
            # confident "no sources found" - mirrors the music layer
            return {"layers_status": {"source_matches": "failed"}}
        return {"source_matches": source_matches,
                "layers_status": {"source_matches": "ok"}}
