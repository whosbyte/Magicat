# magicat/modules/audio/providers.py
"""Music identification providers behind one protocol.

AudD quirks (docs.audd.io): errors arrive as HTTP 200 with status=="error";
no-match is status=="success" with result null; `timecode` ("MM:SS") is the
position in the recognized song where the submitted clip plays.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import requests
from pydantic import BaseModel, Field


class ProviderError(RuntimeError):
    """A provider returned an API-level error (quota, auth, bad audio...)."""


class SongMatch(BaseModel):
    title: str
    artist: str
    song_offset_s: float            # position in the SONG at the clip's start
    provider: str
    score: float = 100.0
    duration_s: float | None = None
    provider_ids: dict[str, str] = Field(default_factory=dict)
    links: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class MusicIdProvider(Protocol):
    name: str

    def identify(self, clip: Path) -> SongMatch | None: ...


def parse_timecode(timecode: str) -> float:
    """AudD timecode 'MM:SS' (or 'HH:MM:SS') -> seconds."""
    parts = [int(p) for p in timecode.split(":")]
    seconds = 0.0
    for part in parts:
        seconds = seconds * 60 + part
    return seconds


class AudDProvider:
    name = "audd"

    def __init__(self, api_token: str) -> None:
        self.api_token = api_token

    def identify(self, clip: Path) -> SongMatch | None:
        with open(clip, "rb") as f:
            resp = requests.post(
                "https://api.audd.io/",
                data={"api_token": self.api_token, "return": "spotify"},
                files={"file": f},
                timeout=30,
            )
        resp.raise_for_status()
        data = resp.json()
        if data["status"] == "error":
            err = data["error"]
            raise ProviderError(
                f"AudD error {err['error_code']}: {err['error_message']}")
        result = data.get("result")
        if not result:
            return None

        links: dict[str, str] = {}
        provider_ids: dict[str, str] = {}
        if result.get("song_link"):
            links["song_link"] = result["song_link"]
        spotify = result.get("spotify") or {}
        if spotify.get("external_urls", {}).get("spotify"):
            links["spotify"] = spotify["external_urls"]["spotify"]
        if spotify.get("id"):
            provider_ids["spotify"] = spotify["id"]

        return SongMatch(
            title=result["title"],
            artist=result["artist"],
            song_offset_s=parse_timecode(result["timecode"]),
            provider=self.name,
            provider_ids=provider_ids,
            links=links,
        )
