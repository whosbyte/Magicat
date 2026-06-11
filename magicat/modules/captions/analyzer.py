# magicat/modules/captions/analyzer.py
"""Caption analysis: sample frames -> OCR -> cluster -> style (spec 6.5)."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path

from PIL import Image
from PIL import Image as PILImage

from magicat.core.registry import register_analyzer
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest
from magicat.modules.captions.clustering import cluster_detections, estimate_fill
from magicat.modules.captions.font_dirs import default_font_dirs
from magicat.modules.captions.font_matcher import FontMatcher
from magicat.modules.captions.ocr import RapidOcrEngine
from magicat.modules.captions.sampling import sample_frames

log = logging.getLogger(__name__)

SAMPLE_FPS = 5.0

CROP_MARGIN = 0.02   # normalized margin around the caption bbox

MIN_FONT_SCORE = 0.05   # below this, the "match" is noise (blank crops)

# Short-form videos virtually always use ONE caption font, so we identify it
# ONCE per job from a small representative subset (the longest-text segments,
# which carry the most glyphs and so match most reliably) rather than running
# the 345-font render-and-compare matcher per segment. The winning font is
# then applied to every segment. See spec 6.5 step 4.
MAX_FONT_ID_SEGMENTS = 5


def save_crop(frame_path: Path, bbox, dest: Path) -> str:
    """Cut the caption region (plus margin) out of a frame; returns path."""
    with Image.open(frame_path) as img:
        width, height = img.size
        x, y, w, h = bbox
        box = (max(0, int((x - CROP_MARGIN) * width)),
               max(0, int((y - CROP_MARGIN) * height)),
               min(width, int((x + w + CROP_MARGIN) * width)),
               min(height, int((y + h + CROP_MARGIN) * height)))
        if box[2] <= box[0] or box[3] <= box[1]:
            raise ValueError(f"degenerate caption bbox {bbox}")
        img.crop(box).save(dest)
    return str(dest)


@register_analyzer
class CaptionAnalyzer:
    name = "caption_analysis"
    layer = "captions"
    needs_gpu = False
    engine_factory = staticmethod(RapidOcrEngine)   # injectable for tests
    matcher_factory = staticmethod(
        lambda: FontMatcher.from_dirs(default_font_dirs()))  # injectable

    def run(self, manifest: Manifest, ws: Workspace) -> dict:
        engine = self.engine_factory()
        samples = sample_frames(Path(manifest.source.file),
                                ws.media_dir / "ocr_frames", fps=SAMPLE_FPS)
        detections = [(s.t, engine.read(s.path)) for s in samples]
        segments = cluster_detections(detections,
                                      frame_interval=1.0 / SAMPLE_FPS)

        # clustering extends t_end by one frame interval; never overshoot
        # the actual video duration
        if manifest.source.duration:
            for seg in segments:
                seg["t_end"] = min(seg["t_end"], manifest.source.duration)

        # style (spec 6.5 step 5, the cheaply-derivable parts): fill color
        # from the segment's middle frame, size from bbox height in pixels,
        # alignment from the bbox center. Stroke/shadow are M3.
        # NOTE: size is the glyph-bbox INK height in px (~80% of the
        # authoring font size), not the font's em size.
        frame_height = None
        if manifest.source.resolution:
            frame_height = int(manifest.source.resolution.split("x")[1])
        crops_dir = ws.media_dir / "caption_crops"
        if crops_dir.is_dir():
            shutil.rmtree(crops_dir)   # re-runs must not leave stale crops
        crops_dir.mkdir(parents=True)
        for i, seg in enumerate(segments):
            mid_t = (seg["t_start"] + seg["t_end"]) / 2
            frame = min(samples, key=lambda s: abs(s.t - mid_t))
            crop_times = {seg["t_start"], mid_t,
                          max(seg["t_start"], seg["t_end"] - 1.0 / SAMPLE_FPS)}
            seg["crops"] = []
            try:
                for j, ct in enumerate(sorted(crop_times)):
                    src_frame = min(samples, key=lambda s: abs(s.t - ct))
                    seg["crops"].append(save_crop(
                        src_frame.path, seg["bbox"],
                        crops_dir / f"seg_{i:03d}_{j}.png"))
            except Exception as exc:
                log.warning("caption crop failed: %s", exc)
                seg["crops"] = []
            x, _, w, h = seg["bbox"]
            center = x + w / 2
            if abs(center - 0.5) < 0.05:
                alignment = "center"
            else:
                alignment = "left" if center < 0.5 else "right"
            seg["style"] = {
                "fill": estimate_fill(frame.path, seg["bbox"]),
                "size": round(h * frame_height, 1) if frame_height else None,
                "alignment": alignment,
            }

        # job-level font identification: captions in a short-form video
        # virtually always share ONE font, so we render-and-compare only a
        # small representative subset (the longest-text segments that have
        # crops) and apply the winner to every segment. Font failures must
        # NEVER fail the captions layer, so anything that throws degrades to
        # no candidates.
        for seg in segments:
            seg["style"]["font_family"] = None
            seg["style"]["font_candidates"] = []
        matcher = self.matcher_factory()
        font_family, font_candidates = self._identify_job_font(
            segments, matcher)
        for seg in segments:
            seg["style"]["font_family"] = font_family
            seg["style"]["font_candidates"] = list(font_candidates)

        return {
            "captions": {"segments": segments},
            "layers_status": {"captions": "ok"},
        }

    @staticmethod
    def _identify_job_font(segments: list[dict], matcher):
        """Identify the single caption font shared across the job.

        Runs the matcher only on up to MAX_FONT_ID_SEGMENTS representative
        segments (those with the longest text among segments that have crops),
        majority-votes the winner, and pools per-font scores into a job-level
        top-3 candidate list. Returns (font_family, font_candidates) where
        font_family is non-None only when the winning sample(s) were
        confident. Degrades to (None, []) when there are no crops, the matcher
        errors, or every sample is below the noise floor.
        """
        with_crops = [s for s in segments if s["crops"]]
        if not with_crops:
            return None, []
        # longest text first -> most glyphs -> most reliable match. Tie-break
        # on t_start keeps selection deterministic.
        subset = sorted(with_crops,
                        key=lambda s: (-len(s["text"]), s["t_start"])
                        )[:MAX_FONT_ID_SEGMENTS]

        results = []   # (font_key, confident, ranked) for samples above floor
        for seg in subset:
            try:
                with PILImage.open(seg["crops"][0]) as crop:
                    result = matcher.identify(crop, seg["text"])
            except Exception as exc:
                log.warning("font identification failed: %s", exc)
                continue
            if result.score < MIN_FONT_SCORE:
                continue   # blank/garbage crop: no candidates beat noise
            results.append(result)

        if not results:
            return None, []

        # majority vote on the winning font_key; ties broken by highest mean
        # winning score across the samples that picked that key.
        votes: dict[str, list[float]] = {}
        for r in results:
            votes.setdefault(r.font_key, []).append(r.score)
        winner = max(votes,
                     key=lambda k: (len(votes[k]),
                                    sum(votes[k]) / len(votes[k])))

        # pool candidates: average each font's score across all samples that
        # ranked it, then take the global top-3.
        pooled: dict[str, list[float]] = {}
        for r in results:
            for name, score in r.ranked:
                pooled.setdefault(name, []).append(score)
        ranked = sorted(pooled.items(),
                        key=lambda kv: -sum(kv[1]) / len(kv[1]))
        font_candidates = [
            {"name": name, "confidence": round(sum(s) / len(s), 4)}
            for name, s in ranked[:3]]

        # font_family is set only when the winning sample(s) were confident.
        winner_confident = any(r.confident for r in results
                               if r.font_key == winner)
        font_family = winner if winner_confident else None
        return font_family, font_candidates
