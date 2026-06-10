# magicat/modules/export/srt.py
"""Caption sidecar: SRT is the cross-NLE caption path (xmeml titles are
not portable - Premiere generators import as slugs elsewhere)."""
from __future__ import annotations

from magicat.manifest.schema import Manifest


def srt_timestamp(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def to_srt(manifest: Manifest) -> str:
    blocks = []
    for i, seg in enumerate(manifest.captions.segments, start=1):
        blocks.append(f"{i}\n{srt_timestamp(seg.t_start)} --> "
                      f"{srt_timestamp(seg.t_end)}\n{seg.text}\n")
    return "\n".join(blocks)
