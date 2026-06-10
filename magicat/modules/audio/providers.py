# magicat/modules/audio/providers.py
"""Music identification providers behind one protocol.

AudD quirks (docs.audd.io): errors arrive as HTTP 200 with status=="error";
no-match is status=="success" with result null; `timecode` ("MM:SS") is the
position in the recognized song where the submitted clip plays.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import time
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

        timecode = result.get("timecode")
        if not timecode:
            raise ProviderError("AudD result missing timecode")
        try:
            song_offset_s = parse_timecode(timecode)
        except ValueError as exc:
            raise ProviderError(
                f"AudD malformed timecode {timecode!r}") from exc

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
            song_offset_s=song_offset_s,
            provider=self.name,
            provider_ids=provider_ids,
            links=links,
        )


class ACRCloudProvider:
    """Raw HTTP + HMAC-SHA1 signing (docs.acrcloud.com identification-api).

    The official pyacrcloud wheel does not exist for Windows, and the signing
    protocol is ~15 lines of stdlib - so no SDK.
    """

    name = "acrcloud"

    def __init__(self, host: str, access_key: str, access_secret: str) -> None:
        self.host = host
        self.access_key = access_key
        self.access_secret = access_secret

    def _string_to_sign(self, timestamp: str) -> str:
        return "\n".join(["POST", "/v1/identify", self.access_key,
                          "audio", "1", timestamp])

    def _signature(self, timestamp: str) -> str:
        digest = hmac.new(
            self.access_secret.encode("ascii"),
            self._string_to_sign(timestamp).encode("ascii"),
            digestmod=hashlib.sha1,
        ).digest()
        return base64.b64encode(digest).decode("ascii")

    def identify(self, clip: Path) -> SongMatch | None:
        timestamp = str(int(time.time()))
        sample = clip.read_bytes()
        resp = requests.post(
            f"https://{self.host}/v1/identify",
            files={"sample": (clip.name, sample, "audio/wav")},
            data={
                "access_key": self.access_key,
                "sample_bytes": str(len(sample)),
                "timestamp": timestamp,
                "signature": self._signature(timestamp),
                "data_type": "audio",
                "signature_version": "1",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        code = data["status"]["code"]
        if code == 1001:
            return None
        if code != 0:
            raise ProviderError(
                f"ACRCloud error {code}: {data['status']['msg']}")

        try:
            best = data["metadata"]["music"][0]   # best match first
            # song position at clip start (docs-sanctioned formula; the
            # play_offset_ms field reflects the matched region's position)
            offset_s = max(0.0, (best["db_begin_time_offset_ms"]
                                 - best["sample_begin_time_offset_ms"])
                           / 1000.0)
            title = best["title"]
        except (KeyError, IndexError) as exc:
            raise ProviderError(
                f"ACRCloud malformed hit response: {exc!r}") from exc

        provider_ids: dict[str, str] = {}
        links: dict[str, str] = {}
        isrc = best.get("external_ids", {}).get("isrc")
        if isrc:
            provider_ids["isrc"] = isrc
        spotify_id = (best.get("external_metadata", {})
                      .get("spotify", {}).get("track", {}).get("id"))
        if spotify_id:
            provider_ids["spotify"] = spotify_id
            links["spotify"] = f"https://open.spotify.com/track/{spotify_id}"
        if best.get("acrid"):
            provider_ids["acrcloud"] = best["acrid"]

        duration_ms = best.get("duration_ms")
        return SongMatch(
            title=title,
            artist=", ".join(a["name"] for a in best.get("artists", [])),
            song_offset_s=offset_s,
            provider=self.name,
            score=float(best.get("score", 0)),
            duration_s=duration_ms / 1000.0 if duration_ms else None,
            provider_ids=provider_ids,
            links=links,
        )
