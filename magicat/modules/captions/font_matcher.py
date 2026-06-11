# magicat/modules/captions/font_matcher.py
"""Identify a caption's font by render-and-compare (spec 6.5 step 4).

Approach (empirically verified on the Windows font set): render the OCR'd
text in every candidate font, normalize query and candidates to a 64px ink
band, stroke-normalize both masks (cancels outline/weight/JPEG fattening),
score by IoU. 5/5 on clean + JPEG40-degraded crops; heavy outlines and
near-twin fonts (verdana/tahoma) collapse to tiny margins, which is why
results are ALWAYS a ranked top-K with a `confident` flag (margin >= 0.06),
never a single hard answer.

cv2 is optional (transitive via scenedetect[opencv]); without it
stroke-normalization is a no-op and clean cases still match.
Never map bbox pixel height to ImageFont size - render at 4x and
normalize by ink height instead.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2
    _HAVE_CV2 = True
except Exception:                       # pragma: no cover
    _HAVE_CV2 = False

NORM_H = 64
PAD = 8
TARGET_STROKE = 3
MARGIN_CONFIDENT = 0.06


def _crop_to_ink(img_l: Image.Image, target_h: int, pad: int = PAD
                 ) -> Image.Image:
    arr = np.asarray(img_l)
    ys, xs = np.where(arr > 32)
    if len(xs) == 0:
        return Image.new("L", (target_h, target_h), 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    crop = img_l.crop((int(x0), int(y0), int(x1) + 1, int(y1) + 1))
    w, h = crop.size
    new_w = max(1, int(round(w * target_h / h)))
    crop = crop.resize((new_w, target_h), Image.LANCZOS)
    out = Image.new("L", (new_w + 2 * pad, target_h + 2 * pad), 0)
    out.paste(crop, (pad, pad))
    return out


def render_sample(font_path: str, text: str,
                  target_h: int = NORM_H) -> Image.Image:
    """Raw, UNCROPPED canvas render - shaped like a wild video crop.

    SYMMETRY RULE: identify() runs prep_crop on the query AND on each
    candidate's render_sample exactly once each. Pre-normalizing either
    side (an extra _crop_to_ink + LANCZOS pass) shrinks heavy-glyph ink
    masks and deterministically misranks fonts like Impact - verified
    failure mode, do not refactor this away.
    """
    font = ImageFont.truetype(font_path, target_h * 4)
    lines = text.count("\n") + 1
    canvas = Image.new("L", (target_h * 60, target_h * 12 * lines), 0)
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text((20, 20), text, fill=255, font=font,
                        spacing=int(target_h * 0.4))
    return canvas


def prep_crop(crop: Image.Image, target_h: int = NORM_H) -> Image.Image:
    gray = crop.convert("L")
    arr = np.asarray(gray)
    if arr.mean() > 127:                # dark-on-light: invert to bright ink
        gray = Image.fromarray(255 - arr)
    return _crop_to_ink(gray, target_h)


def to_mask(img: Image.Image, thresh: int = 128) -> np.ndarray:
    return (np.asarray(img) > thresh).astype(np.uint8)


def _fit_width(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    h, w = ref.shape
    img = Image.fromarray((other * 255).astype(np.uint8)).resize(
        (w, h), Image.LANCZOS)
    return (np.asarray(img) > 128).astype(np.uint8)


def stroke_normalize(mask: np.ndarray,
                     target: int = TARGET_STROKE) -> np.ndarray:
    if not _HAVE_CV2 or not mask.any():
        return mask
    dt = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    est = 2.0 * float(np.median(dt[mask.astype(bool)]))
    iters = int(round((est - target) / 2.0))
    if iters <= 0:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.erode(mask.astype(np.uint8), kernel, iterations=iters)


def score_pair(q_mask: np.ndarray, c_mask: np.ndarray) -> float:
    c_fit = _fit_width(q_mask, c_mask)
    qn, cn = stroke_normalize(q_mask), stroke_normalize(c_fit)
    inter = np.logical_and(qn, cn).sum()
    union = np.logical_or(qn, cn).sum()
    return float(inter / union) if union else 0.0


@dataclass
class MatchResult:
    font_key: str
    score: float
    margin: float
    confident: bool
    ranked: list[tuple[str, float]]


@dataclass
class FontMatcher:
    fonts: dict[str, str] = field(default_factory=dict)
    # memoize the prepped candidate mask per (font_path, text): rendering and
    # normalizing a candidate is the dominant cost, and a job that samples
    # several segments sharing the same caption text pays it only once.
    _cache: dict = field(default_factory=dict, repr=False)

    @classmethod
    def from_dirs(cls, dirs: list[str],
                  extra_env: str = "MAGICAT_FONT_DIRS") -> "FontMatcher":
        paths = list(dirs)
        paths += [p for p in os.environ.get(extra_env, "").split(os.pathsep)
                  if p]
        fonts: dict[str, str] = {}
        for d in paths:                  # later dirs win on key collision
            dp = Path(d)
            if not dp.is_dir():
                continue
            for f in sorted(dp.iterdir()):
                if f.suffix.lower() in (".ttf", ".otf", ".ttc"):
                    fonts[f.stem.lower()] = str(f)
        return cls(fonts=fonts)

    def identify(self, crop: Image.Image, text: str) -> MatchResult:
        q_mask = to_mask(prep_crop(crop, NORM_H))
        scores: dict[str, float] = {}
        for key, path in self.fonts.items():
            cache_key = (path, text)
            c_mask = self._cache.get(cache_key)
            if c_mask is None:
                try:
                    # SYMMETRY: candidate passes through prep_crop exactly
                    # like the query (raw render -> single ink-crop) - see
                    # render_sample docstring for the verified failure mode
                    candidate = prep_crop(render_sample(path, text, NORM_H),
                                          NORM_H)
                except Exception:        # unrenderable font file: skip it
                    continue
                c_mask = to_mask(candidate)
                self._cache[cache_key] = c_mask
            scores[key] = score_pair(q_mask, c_mask)
        if not scores:
            raise RuntimeError("no candidate fonts could be rendered")
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_key, best_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second
        return MatchResult(best_key, best_score, margin,
                           margin >= MARGIN_CONFIDENT, ranked)
