# tests/test_export_fcp7.py
import xml.etree.ElementTree as ET

import pytest

from magicat.manifest.schema import Manifest, Shot, Source
from magicat.modules.export.fcp7 import (
    pathurl,
    rate_for_fps,
    seconds_to_frames,
    to_xmeml,
)


def manifest_30fps(music: bool = False) -> Manifest:
    audio = {}
    if music:
        audio = {"music": {
            "detected": True, "title": "Song", "artist": "Artist",
            "timeline_offset": 2.0,
            "song_segment": {"start_in_song": 30.0, "duration": 10.0},
            "acquisition": {"status": "acquired",
                            "file": "C:/media/music.mp3", "links": {}},
        }}
    return Manifest(
        job_id="j",
        source=Source(file="C:/media/source.mp4", fps=30.0,
                      resolution="1080x1920", duration=30.0),
        shots=[Shot(id="shot_000", start=0.0, end=3.0),
               Shot(id="shot_001", start=3.0, end=7.0)],
        audio=audio,
    )


def test_rate_for_fps():
    assert rate_for_fps(30.0) == (30, False)
    assert rate_for_fps(29.97) == (30, True)
    assert rate_for_fps(23.976) == (24, True)
    assert rate_for_fps(25.0) == (25, False)
    assert rate_for_fps(60.0) == (60, False)


def test_seconds_to_frames():
    assert seconds_to_frames(2.0, 30.0) == 60
    assert seconds_to_frames(1.0, 23.976) == 24


def test_pathurl_windows():
    assert pathurl("C:/media/source.mp4") == \
        "file://localhost/C:/media/source.mp4"
    assert pathurl("C:\\media\\my clip.mp4") == \
        "file://localhost/C:/media/my%20clip.mp4"


def test_xmeml_structure_video_track():
    root = ET.fromstring(to_xmeml(manifest_30fps()))
    assert root.tag == "xmeml" and root.get("version") == "5"
    seq = root.find("sequence")
    fmt = seq.find("media/video/format/samplecharacteristics")
    assert fmt.findtext("width") == "1080"
    assert fmt.findtext("height") == "1920"
    clips = seq.findall("media/video/track/clipitem")
    assert len(clips) == 2
    c0, c1 = clips
    assert (c0.findtext("start"), c0.findtext("end"),
            c0.findtext("in"), c0.findtext("out")) == ("0", "90", "0", "90")
    assert (c1.findtext("start"), c1.findtext("end"),
            c1.findtext("in"), c1.findtext("out")) == ("90", "210", "90",
                                                       "210")
    # file defined fully once, then referenced empty by id
    file0 = c0.find("file")
    assert file0.get("id") == "file-1"
    assert file0.find("pathurl") is not None
    file1 = c1.find("file")
    assert file1.get("id") == "file-1"
    assert len(list(file1)) == 0


def test_xmeml_audio_tracks_no_music():
    root = ET.fromstring(to_xmeml(manifest_30fps(music=False)))
    tracks = root.findall("sequence/media/audio/track")
    assert len(tracks) == 1                      # source audio only
    clips = tracks[0].findall("clipitem")
    assert len(clips) == 2                       # mirrors the shots


def test_xmeml_music_track_at_offset():
    root = ET.fromstring(to_xmeml(manifest_30fps(music=True)))
    tracks = root.findall("sequence/media/audio/track")
    assert len(tracks) == 2
    music_clip = tracks[1].find("clipitem")
    assert music_clip.findtext("start") == "60"      # 2.0s * 30fps
    assert music_clip.findtext("end") == "360"       # +10s segment
    assert music_clip.findtext("in") == "0"          # trimmed file: from 0
    file2 = music_clip.find("file")
    assert file2.get("id") == "file-2"
    assert file2.findtext("pathurl") == \
        "file://localhost/C:/media/music.mp3"


def test_xmeml_round_trips_through_otio():
    otio = pytest.importorskip(
        "opentimelineio", reason="dev validator not installed")
    import opentimelineio.adapters as adapters
    if "fcp_xml" not in adapters.available_adapter_names():
        pytest.skip("fcp_xml adapter not installed")
    timeline = adapters.read_from_string(
        to_xmeml(manifest_30fps(music=True)), "fcp_xml")
    assert len(timeline.tracks) == 3             # 1 video + 2 audio
