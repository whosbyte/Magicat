# magicat/modules/export/fcp7.py
"""Hand-rolled FCP7 XML (xmeml v5) generator - imports into both Premiere
and DaVinci Resolve (research-verified structure; OTIO's fcp_xml adapter is
frozen abandonware, so no runtime OTIO dependency - it validates this
output in tests only).

xmeml semantics: start/end = timeline frames, in/out = source frames,
end-start == out-in. The source <file> is defined fully on its first
clipitem, then referenced by an EMPTY <file id="..."/> on later ones.
"""
from __future__ import annotations

import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import PureWindowsPath

from magicat.manifest.schema import Manifest

RATE_MAP = [
    (23.976, (24, True)), (24.0, (24, False)), (25.0, (25, False)),
    (29.97, (30, True)), (30.0, (30, False)), (50.0, (50, False)),
    (59.94, (60, True)), (60.0, (60, False)),
]


def rate_for_fps(fps: float) -> tuple[int, bool]:
    for known, mapping in RATE_MAP:
        if abs(fps - known) < 0.01:
            return mapping
    return (int(round(fps)), False)


def seconds_to_frames(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


def pathurl(path: str) -> str:
    p = PureWindowsPath(path)
    posix = p.as_posix()
    quoted = urllib.parse.quote(posix, safe="/:")
    if not p.drive:
        # relative path (zip-internal "media/x.mp4"): emit it bare so the
        # importer resolves it against the project file's own location
        return quoted
    # literal drive colon (what Premiere/Resolve/OTIO use); spaces -> %20
    return "file://localhost/" + quoted


def _rate(parent: ET.Element, timebase: int, ntsc: bool) -> None:
    rate = ET.SubElement(parent, "rate")
    ET.SubElement(rate, "timebase").text = str(timebase)
    ET.SubElement(rate, "ntsc").text = "TRUE" if ntsc else "FALSE"


def _full_file(parent: ET.Element, file_id: str, name: str, path: str,
               timebase: int, ntsc: bool, duration: int,
               width: int | None = None, height: int | None = None) -> None:
    f = ET.SubElement(parent, "file", id=file_id)
    ET.SubElement(f, "name").text = name
    ET.SubElement(f, "pathurl").text = pathurl(path)
    _rate(f, timebase, ntsc)
    ET.SubElement(f, "duration").text = str(duration)
    media = ET.SubElement(f, "media")
    if width and height:
        video = ET.SubElement(media, "video")
        sc = ET.SubElement(video, "samplecharacteristics")
        ET.SubElement(sc, "width").text = str(width)
        ET.SubElement(sc, "height").text = str(height)
        _rate(sc, timebase, ntsc)
    audio = ET.SubElement(media, "audio")
    ET.SubElement(audio, "channelcount").text = "2"


def _clipitem(track: ET.Element, item_id: str, name: str, duration: int,
              timebase: int, ntsc: bool, start: int, end: int,
              in_f: int, out_f: int) -> ET.Element:
    clip = ET.SubElement(track, "clipitem", id=item_id)
    ET.SubElement(clip, "name").text = name
    ET.SubElement(clip, "duration").text = str(duration)
    _rate(clip, timebase, ntsc)
    ET.SubElement(clip, "start").text = str(start)
    ET.SubElement(clip, "end").text = str(end)
    ET.SubElement(clip, "in").text = str(in_f)
    ET.SubElement(clip, "out").text = str(out_f)
    return clip


def to_xmeml(manifest: Manifest) -> str:
    src = manifest.source
    fps = src.fps or 30.0
    timebase, ntsc = rate_for_fps(fps)
    width, height = 1080, 1920
    if src.resolution and "x" in src.resolution:
        width, height = (int(v) for v in src.resolution.split("x"))
    src_frames = seconds_to_frames(src.duration or 0.0, fps)

    # timeline: shots laid end to end in order
    spans = []          # (timeline_start, timeline_end, in, out) frames
    cursor = 0
    for shot in manifest.shots:
        in_f = seconds_to_frames(shot.start, fps)
        out_f = seconds_to_frames(shot.end, fps)
        spans.append((cursor, cursor + (out_f - in_f), in_f, out_f))
        cursor += out_f - in_f
    seq_duration = cursor

    root = ET.Element("xmeml", version="5")
    seq = ET.SubElement(root, "sequence", id="magicat-seq-1")
    ET.SubElement(seq, "name").text = f"Magicat {manifest.job_id[:8]}"
    ET.SubElement(seq, "duration").text = str(seq_duration)
    _rate(seq, timebase, ntsc)
    tc = ET.SubElement(seq, "timecode")
    _rate(tc, timebase, ntsc)
    ET.SubElement(tc, "string").text = "00:00:00:00"
    ET.SubElement(tc, "frame").text = "0"
    ET.SubElement(tc, "displayformat").text = "NDF"

    media = ET.SubElement(seq, "media")
    video = ET.SubElement(media, "video")
    vformat = ET.SubElement(video, "format")
    sc = ET.SubElement(vformat, "samplecharacteristics")
    ET.SubElement(sc, "width").text = str(width)
    ET.SubElement(sc, "height").text = str(height)
    ET.SubElement(sc, "pixelaspectratio").text = "square"
    ET.SubElement(sc, "anamorphic").text = "FALSE"
    ET.SubElement(sc, "fielddominance").text = "none"
    _rate(sc, timebase, ntsc)

    src_name = PureWindowsPath(src.file or "source.mp4").name
    vtrack = ET.SubElement(video, "track")
    item = 0
    for i, (t_start, t_end, in_f, out_f) in enumerate(spans):
        item += 1
        clip = _clipitem(vtrack, f"clipitem-{item}",
                         manifest.shots[i].id, src_frames,
                         timebase, ntsc, t_start, t_end, in_f, out_f)
        if i == 0:
            _full_file(clip, "file-1", src_name, src.file or "",
                       timebase, ntsc, src_frames, width, height)
        else:
            ET.SubElement(clip, "file", id="file-1")

    audio = ET.SubElement(media, "audio")
    atrack = ET.SubElement(audio, "track")
    for i, (t_start, t_end, in_f, out_f) in enumerate(spans):
        item += 1
        clip = _clipitem(atrack, f"clipitem-{item}",
                         f"{manifest.shots[i].id}-audio", src_frames,
                         timebase, ntsc, t_start, t_end, in_f, out_f)
        ET.SubElement(clip, "file", id="file-1")
        st = ET.SubElement(clip, "sourcetrack")
        ET.SubElement(st, "mediatype").text = "audio"
        ET.SubElement(st, "trackindex").text = "1"

    music = manifest.audio.music
    if music.detected and music.acquisition.file:
        mtrack = ET.SubElement(audio, "track")
        m_start = seconds_to_frames(music.timeline_offset, fps)
        m_frames = seconds_to_frames(music.song_segment.duration, fps)
        item += 1
        clip = _clipitem(mtrack, f"clipitem-{item}", "music.mp3",
                         m_frames, timebase, ntsc,
                         m_start, m_start + m_frames, 0, m_frames)
        _full_file(clip, "file-2",
                   PureWindowsPath(music.acquisition.file).name,
                   music.acquisition.file, timebase, ntsc, m_frames)
        st = ET.SubElement(clip, "sourcetrack")
        ET.SubElement(st, "mediatype").text = "audio"
        ET.SubElement(st, "trackindex").text = "1"

    ET.indent(root)
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            "<!DOCTYPE xmeml>\n"
            + ET.tostring(root, encoding="unicode"))
