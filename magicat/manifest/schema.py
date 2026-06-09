# magicat/manifest/schema.py
"""Reconstruction Manifest: the single source of truth for a job (spec section 4).

Every module reads the manifest and returns a patch to its own section.
Modules never call each other. Sections not used until later milestones
(audio, captions, source_matches) exist from day one so they drop in
without schema changes.
"""
from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

MANIFEST_VERSION = "1.0"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LayerState(str, Enum):
    PENDING = "pending"
    OK = "ok"
    FAILED = "failed"
    SKIPPED = "skipped"


class Source(StrictModel):
    url: str | None = None
    platform: str | None = None
    file: str | None = None          # normalized source.mp4 in the workspace
    fps: float | None = None
    resolution: str | None = None    # "WxH"
    duration: float | None = None    # seconds


class Shot(StrictModel):
    id: str
    start: float                     # seconds
    end: float
    keyframes: list[str] = Field(default_factory=list)  # file paths / URIs
    confidence: float = 1.0


class SourceCandidate(StrictModel):
    url: str
    title: str | None = None
    thumbnail: str | None = None
    score: float = 0.0


class SourceMatch(StrictModel):
    shot_id: str
    candidates: list[SourceCandidate] = Field(default_factory=list)


class SongSegment(StrictModel):
    start_in_song: float = 0.0
    duration: float = 0.0


class Acquisition(StrictModel):
    status: str = "skipped"          # acquired | failed | skipped
    file: str | None = None
    license: str | None = None
    links: dict[str, str] = Field(default_factory=dict)


class Music(StrictModel):
    detected: bool = False
    title: str | None = None
    artist: str | None = None
    provider_ids: dict[str, str] = Field(default_factory=dict)
    song_segment: SongSegment = Field(default_factory=SongSegment)
    timeline_offset: float = 0.0
    acquisition: Acquisition = Field(default_factory=Acquisition)


class Audio(StrictModel):
    speech_stem: str | None = None
    music: Music = Field(default_factory=Music)


class FontCandidate(StrictModel):
    name: str
    confidence: float = 0.0


class CaptionStyle(StrictModel):
    font_family: str | None = None
    font_candidates: list[FontCandidate] = Field(default_factory=list)
    size: float | None = None
    fill: str | None = None
    stroke: str | None = None
    shadow: str | None = None
    alignment: str | None = None


class CaptionSegment(StrictModel):
    text: str
    t_start: float
    t_end: float
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    style: CaptionStyle = Field(default_factory=CaptionStyle)


class Captions(StrictModel):
    segments: list[CaptionSegment] = Field(default_factory=list)


class Export(StrictModel):
    format: str
    artifact: str


class Manifest(StrictModel):
    manifest_version: str = MANIFEST_VERSION
    job_id: str
    source: Source = Field(default_factory=Source)
    shots: list[Shot] = Field(default_factory=list)
    source_matches: list[SourceMatch] = Field(default_factory=list)
    audio: Audio = Field(default_factory=Audio)
    captions: Captions = Field(default_factory=Captions)
    layers_status: dict[str, LayerState] = Field(default_factory=dict)
    exports: list[Export] = Field(default_factory=list)
    report: dict = Field(default_factory=dict)
