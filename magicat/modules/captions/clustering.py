# magicat/modules/captions/clustering.py
"""Group per-frame OCR detections into caption segments (spec 6.5 step 2).

A detection joins an active cluster when its text is similar AND its box
overlaps (captions are positionally stable). Clusters survive one missed
frame (OCR flicker) and need >= MIN_FRAMES sightings (drops in-scene text
glimpses and one-frame noise - the M2 stand-in for spec step 3's burned-in
vs in-scene discrimination; full heuristics arrive with M3 font work).
"""
from __future__ import annotations

from collections import Counter
from pathlib import Path

import difflib

from magicat.modules.captions.ocr import OcrLine

TEXT_SIMILARITY_MIN = 0.8
IOU_MIN = 0.5
MIN_FRAMES = 3
MAX_MISSED_FRAMES = 1
MIN_CONFIDENCE = 0.6

Bbox = tuple[float, float, float, float]


def bbox_iou(a: Bbox, b: Bbox) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax0 + aw, bx0 + bw), min(ay0 + ah, by0 + bh)
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def text_similar(a: str, b: str) -> bool:
    ratio = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return ratio >= TEXT_SIMILARITY_MIN


class _Cluster:
    def __init__(self, t: float, ocr_line: OcrLine) -> None:
        self.t_first = t
        self.t_last = t
        self.texts = Counter({ocr_line.text: 1})
        self.bboxes = [ocr_line.bbox]
        self.missed = 0

    def matches(self, ocr_line: OcrLine) -> bool:
        return (text_similar(self.top_text(), ocr_line.text)
                and bbox_iou(self.bboxes[-1], ocr_line.bbox) >= IOU_MIN)

    def add(self, t: float, ocr_line: OcrLine) -> None:
        self.t_last = t
        self.texts[ocr_line.text] += 1
        self.bboxes.append(ocr_line.bbox)
        self.missed = 0

    def top_text(self) -> str:
        return self.texts.most_common(1)[0][0]

    def frame_count(self) -> int:
        return sum(self.texts.values())

    def mean_bbox(self) -> Bbox:
        n = len(self.bboxes)
        return tuple(sum(b[i] for b in self.bboxes) / n  # type: ignore
                     for i in range(4))


def cluster_detections(
        detections: list[tuple[float, list[OcrLine]]],
        frame_interval: float) -> list[dict]:
    """detections: [(t, ocr_lines)] in time order -> caption segment dicts."""
    active: list[_Cluster] = []
    finished: list[_Cluster] = []

    for t, ocr_lines in detections:
        usable = [l for l in ocr_lines if l.confidence >= MIN_CONFIDENCE]
        matched: set[int] = set()
        for ocr_line in usable:
            for idx, cluster in enumerate(active):
                if idx in matched:
                    continue
                if cluster.matches(ocr_line):
                    cluster.add(t, ocr_line)
                    matched.add(idx)
                    break
            else:
                active.append(_Cluster(t, ocr_line))
                matched.add(len(active) - 1)
        still_active = []
        for idx, cluster in enumerate(active):
            if idx in matched:
                still_active.append(cluster)
            else:
                cluster.missed += 1
                if cluster.missed > MAX_MISSED_FRAMES:
                    finished.append(cluster)
                else:
                    still_active.append(cluster)
        active = still_active
    finished.extend(active)

    segments = []
    for cluster in finished:
        if cluster.frame_count() < MIN_FRAMES:
            continue
        segments.append({
            "text": cluster.top_text(),
            "t_start": round(cluster.t_first, 3),
            "t_end": round(cluster.t_last + frame_interval, 3),
            "bbox": cluster.mean_bbox(),
            "style": {},
        })
    segments.sort(key=lambda s: s["t_start"])
    return segments


def estimate_fill(image: Path, bbox: Bbox) -> str:
    """Median color of the brightest quartile inside the caption box -
    a fair proxy for fill color on dark-video captions."""
    import numpy as np
    from PIL import Image

    img = Image.open(image).convert("RGB")
    width, height = img.size
    x, y, w, h = bbox
    crop = img.crop((int(x * width), int(y * height),
                     int((x + w) * width), int((y + h) * height)))
    pixels = np.asarray(crop).reshape(-1, 3).astype(float)
    if pixels.shape[0] == 0:
        return "#000000"   # degenerate box - no pixels to sample
    luminance = pixels @ [0.299, 0.587, 0.114]
    bright = pixels[luminance >= np.percentile(luminance, 75)]
    r, g, b = (int(c) for c in np.median(bright, axis=0))
    return f"#{r:02X}{g:02X}{b:02X}"
