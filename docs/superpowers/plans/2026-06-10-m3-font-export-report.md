# Magicat M3 — Font ID, NLE Export & Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Identify caption fonts, export an editable Premiere/Resolve project (xmeml v5 + SRT + media + report, zipped), render the preview with the acquired music mixed at its timeline offset, and generate the user-facing report — completing the v1 deliverable.

**Architecture:** Three new pieces behind the existing plugin contracts: a render-and-compare font matcher inside the captions package (stroke-normalized IoU — no NN training), an `export` package with a hand-rolled xmeml v5 generator + SRT writer + zip packager (no OpenTimelineIO runtime dependency; its FCP7 adapter is frozen abandonware — OTIO is used only as an optional test validator), and a report builder whose dict lands in `manifest.report` and whose HTML lands in every export. The preview renderer is rebuilt as a single-pass `filter_complex` (kills the M1 AAC-drift debt and mixes music at `timeline_offset`). Two backlog tasks first pay down M2 review debt the exporters need.

**Tech Stack:** stdlib `xml.etree.ElementTree` (xmeml), PIL + numpy (+ cv2 transitively via scenedetect — matcher degrades gracefully without it), ffmpeg `filter_complex` (verified command shapes), `string.Template` (report HTML), `zipfile` (package). Dev-only: `opentimelineio>=0.18` + `otio-fcp-adapter` for an xmeml round-trip validation test (skips if absent).

**Spec:** `docs/superpowers/specs/2026-06-09-magicat-framework-design.md` §6.5 step 4 (font), §7 (assembly/exporters — xmeml stands in for the OTIO path with recorded rationale), §9 (report). Backlog: `memory/magicat-m2-backlog.md` items folded into Tasks 1–3.

**Research basis (verified empirically on this machine, 2026-06-10):**
- **xmeml:** Premiere + Resolve both import xmeml v5. `start`/`end` = timeline frames; `in`/`out` = source frames; `end-start == out-in`. File defined fully ONCE (first clipitem), then referenced by empty `<file id="..."/>`. Windows pathurl: `file://localhost/C:/path/file.mp4` (forward slashes, literal drive colon — what the OTIO writer emits and importers accept; spaces as `%20`). Rate map: 30→(30,FALSE), 29.97→(30,TRUE), 23.976→(24,TRUE), 25→(25,FALSE), 60→(60,FALSE). Set sequence `<format>` samplecharacteristics explicitly (1080x1920) or importers guess landscape. Captions do NOT travel portably in xmeml (Premiere generators come in as slugs) → SRT sidecar; Resolve imports SRT via File→Import→Subtitle as a SEPARATE step (document it).
- **Font matching:** stroke-normalized IoU over ink-normalized renders wins: 5/5 clean and JPEG40+blur on the 5-font Windows set, 10/10 clean on 10 fonts. Heavy outlines (stroke≥4) degrade and near-twin fonts (verdana/tahoma) collapse to <0.02 margins — hence ALWAYS top-K + `confident` flag (margin ≥ 0.06), never a single hard answer. Render at 4× then normalize by ink height (never map bbox px → ImageFont size). cv2 powers stroke-normalization; without it, plain IoU still wins clean cases.
- **ffmpeg single-pass:** `trim/atrim+setpts/asetpts` per segment → `concat=n=N:v=1:a=1` → music branch `volume=V,adelay=MS:all=1` → `amix=inputs=2:duration=first:normalize=0`. `duration=first` is load-bearing (default `longest` overruns: 6.5s audio on 6.0s video). `normalize=0` is load-bearing (default halves source dialog). Filtergraph passed as ONE list element, `shell=False` — no escaping needed.

---

## File Structure

```
pyproject.toml                      # MODIFY: dev extra += opentimelineio, otio-fcp-adapter
assets/fonts/<family>/              # NEW: OFL fonts + their OFL.txt (Task 5)
magicat/
  manifest/schema.py                # MODIFY: Music.provider, Acquisition.skip_reason,
                                    #         CaptionSegment.crops
  core/ffmpeg.py                    # MODIFY: + run_ffprobe(path, entries) helper
  core/pipeline.py                  # MODIFY: EXPORTERS list, report patch step
  cli.py                            # MODIFY: per-job workdir default, absolute manifest path
  modules/
    ingest.py                       # MODIFY: probe() uses run_ffprobe
    audio/extract.py                # MODIFY: wav_duration() uses run_ffprobe
    audio/identify.py               # MODIFY: provider provenance, tie-break comment
    audio/acquire.py                # MODIFY: skip_reason, query sanitize, duration cap
    captions/analyzer.py            # MODIFY: crop persistence, t_end clamp, font ID wiring
    captions/font_matcher.py        # NEW: render-and-compare matcher (verified design)
    export/
      __init__.py                   # NEW (empty)
      srt.py                        # NEW: manifest captions -> SRT text
      fcp7.py                       # NEW: manifest -> xmeml v5 ElementTree
      package.py                    # NEW: premiere_resolve_zip exporter
    render_preview.py               # MODIFY: single-pass filter_complex + music
    report.py                       # NEW: build_report(manifest) dict + report_html exporter
tests/
  test_backlog_m3.py                # NEW (Tasks 1-2 schema/behavior fixes)
  test_caption_crops.py             # NEW
  test_font_matcher.py              # NEW
  test_font_wiring.py               # NEW
  test_render_music.py              # NEW
  test_export_srt.py                # NEW
  test_export_fcp7.py               # NEW
  test_export_package.py            # NEW
  test_report.py                    # NEW
```

---

### Task 1: Backlog batch A — manifest fields + provenance + skip_reason + t_end clamp

**Files:**
- Modify: `magicat/manifest/schema.py`, `magicat/modules/audio/identify.py`, `magicat/modules/audio/acquire.py`, `magicat/modules/captions/analyzer.py`
- Test: `tests/test_backlog_m3.py` (new), `tests/test_audio_identify.py` (one assertion)

- [ ] **Step 1: Write the failing tests** — create `tests/test_backlog_m3.py`:

```python
# tests/test_backlog_m3.py
"""M2-review backlog items: schema additions + behavior fixes (M3 Task 1-2)."""
from pathlib import Path

import pytest

from magicat.manifest.schema import (
    Acquisition,
    CaptionSegment,
    Manifest,
    Music,
)


def test_music_provider_field():
    m = Music(detected=True, title="T", artist="A", provider="audd")
    assert m.provider == "audd"
    assert Music().provider is None


def test_acquisition_skip_reason_field():
    a = Acquisition(status="skipped", skip_reason="policy:link_only")
    assert a.skip_reason == "policy:link_only"
    assert Acquisition().skip_reason is None


def test_caption_segment_crops_field():
    seg = CaptionSegment(text="X", t_start=0.0, t_end=1.0,
                         crops=["a.png", "b.png"])
    assert seg.crops == ["a.png", "b.png"]
    assert CaptionSegment(text="X", t_start=0.0, t_end=1.0).crops == []


def test_align_carries_provider():
    from magicat.modules.audio.identify import align
    from magicat.modules.audio.extract import AudioWindow
    from magicat.modules.audio.providers import SongMatch

    windows = [AudioWindow(t_start=0.0, path=Path("w.wav"))]
    matches = [SongMatch(title="T", artist="A", song_offset_s=10.0,
                         provider="acrcloud")]
    music = align(windows, matches, video_duration=20.0, window_s=12.0)
    assert music["provider"] == "acrcloud"


def test_acquire_records_skip_reason_link_only(tmp_path, monkeypatch):
    from magicat.core.workspace import Workspace
    from magicat.modules.audio.acquire import Candidate, MusicAcquisition
    from magicat.manifest.schema import Source

    ws = Workspace(tmp_path / "job")
    monkeypatch.setenv("MAGICAT_ACQUISITION_POLICY", "link_only")
    analyzer = MusicAcquisition()
    monkeypatch.setattr(analyzer, "prober", lambda q: Candidate(
        url="https://soundcloud.com/x/y", title="T", duration=100.0,
        license="all-rights-reserved", source="soundcloud"))
    m = Manifest(job_id="j", source=Source(file="x.mp4"),
                 audio={"music": {"detected": True, "title": "T",
                                  "artist": "A", "duration_s": 100.0,
                                  "song_segment": {"start_in_song": 0.0,
                                                   "duration": 10.0}}})
    patch = analyzer.run(m, ws)
    acq = patch["audio"]["music"]["acquisition"]
    assert acq["status"] == "skipped"
    assert acq["skip_reason"] == "policy:link_only"


def test_acquire_records_skip_reason_license_gate(tmp_path, monkeypatch):
    from magicat.core.workspace import Workspace
    from magicat.modules.audio.acquire import Candidate, MusicAcquisition
    from magicat.manifest.schema import Source

    ws = Workspace(tmp_path / "job")
    monkeypatch.setenv("MAGICAT_ACQUISITION_POLICY", "licensed_only")
    analyzer = MusicAcquisition()
    monkeypatch.setattr(analyzer, "prober", lambda q: Candidate(
        url="https://soundcloud.com/x/y", title="T", duration=100.0,
        license="all-rights-reserved", source="soundcloud"))
    m = Manifest(job_id="j", source=Source(file="x.mp4"),
                 audio={"music": {"detected": True, "title": "T",
                                  "artist": "A", "duration_s": 100.0,
                                  "song_segment": {"start_in_song": 0.0,
                                                   "duration": 10.0}}})
    patch = analyzer.run(m, ws)
    acq = patch["audio"]["music"]["acquisition"]
    assert acq["status"] == "skipped"
    assert acq["skip_reason"] == "license:all-rights-reserved"


def test_caption_t_end_clamped_to_duration(fixture_video, tmp_path,
                                           monkeypatch):
    from magicat.core.workspace import Workspace
    from magicat.manifest.patch import apply_patch
    from magicat.manifest.schema import Source
    from magicat.modules.captions.analyzer import CaptionAnalyzer
    from magicat.modules.captions.ocr import OcrLine
    from magicat.modules.ingest import IngestAnalyzer

    class TailEngine:
        """Caption runs to the very last sampled frame (frame 25..30 at
        5fps = t 4.8..5.8; raw cluster t_end = 5.8+0.2 = 6.0)."""

        def read(self, image):
            n = int(image.stem.split("_")[1])
            if n >= 25:
                return [OcrLine(text="TAIL CAPTION",
                                bbox=(0.25, 0.8, 0.5, 0.06),
                                confidence=0.95)]
            return []

    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    # force a duration the raw cluster t_end (6.0) definitely exceeds, so
    # this test FAILS until the clamp exists (no tautology on probe jitter)
    src = m.source.model_dump(mode="json")
    src["duration"] = 5.9
    m = apply_patch(m, {"source": src})
    analyzer = CaptionAnalyzer()
    monkeypatch.setattr(analyzer, "engine_factory", lambda: TailEngine())
    patch = analyzer.run(m, ws)
    seg = patch["captions"]["segments"][0]
    assert seg["t_end"] == 5.9            # clamped to source.duration
```

And in `tests/test_audio_identify.py`, in `test_align_carries_metadata_from_best_match`, add one assertion after the duration_s line:

```python
    assert music["provider"] == "fake"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_backlog_m3.py tests/test_audio_identify.py -v`
Expected: schema tests FAIL with ValidationError (unknown fields `provider`/`skip_reason`/`crops`); `test_align_carries_provider` and the updated metadata test FAIL with KeyError `provider`; the two skip_reason tests FAIL with KeyError `skip_reason`; the clamp test FAILS — raw cluster t_end is 6.0 while the test pins source.duration to 5.9, so the `== 5.9` assertion fails until the clamp lands.

- [ ] **Step 3: Implement**

(a) `magicat/manifest/schema.py`:
- `Music` gains `provider: str | None = None` (after `artist`, before `duration_s`) — which service identified the song (report provenance).
- `Acquisition` gains `skip_reason: str | None = None` (after `status`) — why a skipped acquisition was skipped (`policy:link_only` | `license:<license>`).
- `CaptionSegment` gains `crops: list[str] = Field(default_factory=list)` (after `bbox`) — representative crop image paths (font-matcher input).

(b) `magicat/modules/audio/identify.py`, in `align()`'s return dict, add after `"artist": best.artist,`:

```python
        "provider": best.provider,
```

Also add the tie-break comment above the `winner = max(...)` line:

```python
    # ties resolve deterministically by dict insertion order (= earliest
    # window's identity wins) - acceptable at window-level precision
```

(c) `magicat/modules/audio/acquire.py`, in `MusicAcquisition.run`, the not-allowed branch currently sets `acq["status"] = "skipped"`. Replace that branch with:

```python
        if not download_allowed:
            acq["status"] = "skipped"
            acq["skip_reason"] = (
                "policy:link_only" if policy == "link_only"
                else f"license:{chosen.license}")
            return {"audio": audio,
                    "layers_status": {"music_acquisition": "ok"}}
```

(d) `magicat/modules/captions/analyzer.py`, in `run()` right after `segments = cluster_detections(...)`, add:

```python
        # clustering extends t_end by one frame interval; never overshoot
        # the actual video duration
        if manifest.source.duration:
            for seg in segments:
                seg["t_end"] = min(seg["t_end"], manifest.source.duration)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_backlog_m3.py tests/test_audio_identify.py -v`
Expected: 7 + 8 PASS. Full suite: 110 passed, 2 skipped (103 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add magicat tests
git commit -m "feat: provider provenance, skip_reason, caption crops field, t_end clamp"
```

---

### Task 2: Backlog batch B — ffprobe consolidation, CLI workdir, acquisition guards

**Files:**
- Modify: `magicat/core/ffmpeg.py`, `magicat/modules/ingest.py`, `magicat/modules/audio/extract.py`, `magicat/cli.py`, `magicat/modules/audio/acquire.py`, `README.md`, `magicat/modules/captions/analyzer.py`
- Test: `tests/test_backlog_m3.py` (append), `tests/test_cli.py` (one test update)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_backlog_m3.py`:

```python
def test_run_ffprobe_helper(fixture_video):
    from magicat.core.ffmpeg import run_ffprobe
    data = run_ffprobe(fixture_video, "format=duration")
    assert abs(float(data["format"]["duration"]) - 6.0) < 0.2


def test_sanitize_query():
    from magicat.modules.audio.acquire import sanitize_query
    assert sanitize_query('Artist: "Title"') == "Artist Title"
    assert sanitize_query("A;B|C&D") == "A B C D"
    assert sanitize_query("  plain  text  ") == "plain text"


def test_unknown_duration_rejects_absurdly_long_candidate():
    from magicat.modules.audio.acquire import validate_candidate
    from magicat.modules.audio.acquire import Candidate
    ten_hour_loop = Candidate(
        url="https://youtube.com/watch?v=1",
        title="Around the World (10 hour loop)", duration=36000.0,
        license=None, source="youtube")
    match_info = {"title": "Around the World", "artist": "Daft Punk",
                  "duration_s": None}   # provider gave no duration
    assert validate_candidate(ten_hour_loop, match_info) is False


def test_cli_default_workdir_is_per_job(fixture_video, tmp_path,
                                        monkeypatch):
    from typer.testing import CliRunner
    from magicat.cli import app
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    r1 = runner.invoke(app, ["run", str(fixture_video)])
    r2 = runner.invoke(app, ["run", str(fixture_video)])
    assert r1.exit_code == 0 and r2.exit_code == 0
    jobs = list((tmp_path / "jobs").iterdir())
    assert len(jobs) == 2          # one fresh directory per job, no reuse
```

And in `tests/test_cli.py`, REPLACE `test_run_command_prints_summary` with (the manifest path is now printed absolute):

```python
def test_run_command_prints_summary(fixture_video, tmp_path):
    workdir = tmp_path / "job"
    result = runner.invoke(
        app, ["run", str(fixture_video), "--workdir", str(workdir)])
    assert result.exit_code == 0
    assert "shots: 3" in result.output
    assert "preview_mp4" in result.output
    assert str(workdir.resolve()) in result.output   # absolute manifest path
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_backlog_m3.py tests/test_cli.py -v`
Expected: run_ffprobe ImportError; sanitize_query ImportError; ten-hour test FAILS (currently passes validation — no cap when expected duration unknown); per-job workdir FAILS (both runs reuse `jobs/latest`, 1 dir); CLI summary test FAILS (relative path printed).

- [ ] **Step 3: Implement**

(a) `magicat/core/ffmpeg.py` — append:

```python
import json


def run_ffprobe(path, entries: str) -> dict:
    """ffprobe -show_entries wrapper returning parsed JSON.

    entries: ffprobe -show_entries value, e.g. "format=duration" or
    "stream=r_frame_rate,width,height" (multiple groups joined with ':').
    """
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", entries,
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(out)
```

(b) `magicat/modules/audio/extract.py` — replace `wav_duration` body to use the helper (and drop the now-unused `json`/`subprocess` imports if nothing else uses them):

```python
from magicat.core.ffmpeg import run_ffmpeg, run_ffprobe


def wav_duration(path: Path) -> float:
    return float(run_ffprobe(path, "format=duration")["format"]["duration"])
```

(c) `magicat/modules/ingest.py` — rewrite `probe()` to use the helper (keep the video-stream guard and fps math identical):

```python
from magicat.core.ffmpeg import run_ffprobe


def probe(path: Path) -> dict:
    data = run_ffprobe(
        path, "stream=r_frame_rate,width,height:format=duration")
    if not data["streams"]:
        raise ValueError(f"no video stream in {path}")
    stream = data["streams"][0]
    num, den = (float(x) for x in stream["r_frame_rate"].split("/"))
    return {
        "fps": num / den if den else 0.0,
        "resolution": f"{stream['width']}x{stream['height']}",
        "duration": float(data["format"]["duration"]),
    }
```

NOTE: ffprobe `-select_streams` is dropped here; without it, audio streams also appear in `data["streams"]` (they have no width/height). Filter instead:

```python
    streams = [s for s in data.get("streams", []) if "width" in s]
    if not streams:
        raise ValueError(f"no video stream in {path}")
    stream = streams[0]
```

Use this filtered version (audio-only inputs must still raise — `tests/test_ingest.py::test_probe_rejects_audio_only` pins it).

(d) `magicat/cli.py` — per-job default workdir + absolute manifest path. Change the `run` signature default to `workdir: Path | None = typer.Option(None, "--workdir")` and resolve it inside:

```python
@app.command()
def run(
    input_arg: str = typer.Argument(..., metavar="URL_OR_FILE"),
    workdir: Path | None = typer.Option(None, "--workdir"),
) -> None:
    """Deconstruct a short-form video into a layered project."""
    logging.basicConfig(level=logging.INFO)
    if workdir is None:
        workdir = Path("jobs") / uuid.uuid4().hex[:12]   # fresh dir per job
    try:
        manifest = run_job(input_arg, workdir)
    except Exception as exc:  # ingest failure is fatal (spec section 5)
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=1)
```

Add `import uuid` to the imports. Change the final echo to the absolute path:

```python
    typer.echo(f"manifest: {Path(workdir).resolve() / 'manifest.json'}")
```

(e) `magicat/modules/audio/acquire.py`:

```python
import re

MAX_CANDIDATE_DURATION_S = 1800.0   # reject obvious loops/compilations


def sanitize_query(text: str) -> str:
    """Strip characters that confuse yt-dlp search-prefix parsing."""
    return re.sub(r"\s+", " ", re.sub(r'[:;|&"\']', " ", text)).strip()
```

In `validate_candidate`, after the existing duration-tolerance check, add the absolute cap (fires even when the expected duration is unknown):

```python
    if candidate.duration and candidate.duration > MAX_CANDIDATE_DURATION_S:
        return False
```

In `MusicAcquisition.run`, change `query_text = f"{music.artist} {music.title}"` to `query_text = sanitize_query(f"{music.artist} {music.title}")`.

(f) `magicat/modules/captions/analyzer.py` — the existing style-block comment reads:

```python
        # style (spec 6.5 step 5, the cheaply-derivable parts): fill color
        # from the segment's middle frame, size from bbox height in pixels,
        # alignment from the bbox center. Stroke/shadow are M3.
```

APPEND one line to that exact comment block (do not move it — Task 3 later restructures the loop BODY below this comment but never the comment itself, so ordering is safe):

```python
        # NOTE: size is the glyph-bbox INK height in px (~80% of the
        # authoring font size), not the font's em size.
```

(g) `README.md` — in the "Music & captions (M2)" section, append one line: `Note: link_only still performs network probes (to collect links); it only skips the download itself.`

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_backlog_m3.py tests/test_cli.py tests/test_ingest.py tests/test_audio_extract.py -v`
Expected: all PASS (the ingest/extract suites prove the ffprobe refactor is behavior-identical). Full suite: 114 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat tests README.md
git commit -m "refactor: ffprobe helper, per-job workdirs, acquisition guards"
```

---

### Task 3: Caption crop persistence

**Files:**
- Modify: `magicat/modules/captions/analyzer.py`
- Test: `tests/test_caption_crops.py` (new)

- [ ] **Step 1: Write the failing tests** — create `tests/test_caption_crops.py`:

```python
# tests/test_caption_crops.py
from pathlib import Path

from PIL import Image

from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import Manifest, Source
from magicat.modules.captions.analyzer import CaptionAnalyzer
from magicat.modules.captions.ocr import OcrLine
from magicat.modules.ingest import IngestAnalyzer


class OneCaptionEngine:
    def read(self, image):
        n = int(image.stem.split("_")[1])
        if 6 <= n <= 15:
            return [OcrLine(text="CROPPED CAPTION",
                            bbox=(0.25, 0.8, 0.5, 0.06), confidence=0.95)]
        return []


def test_segment_crops_are_persisted(fixture_video, tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    analyzer = CaptionAnalyzer()
    monkeypatch.setattr(analyzer, "engine_factory",
                        lambda: OneCaptionEngine())
    patch = analyzer.run(m, ws)
    seg = patch["captions"]["segments"][0]
    crops = seg["crops"]
    assert 1 <= len(crops) <= 3
    for crop_path in crops:
        p = Path(crop_path)
        assert p.is_file()
        with Image.open(p) as img:
            w, h = img.size
            assert w > 0 and h > 0
            # crop covers the caption bbox plus margin: wider than tall
            assert w > h
    # round-trips the schema
    m2 = apply_patch(m, patch)
    assert m2.captions.segments[0].crops == crops
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/Scripts/python -m pytest tests/test_caption_crops.py -v`
Expected: FAIL — KeyError `crops` (analyzer doesn't emit the field yet).

- [ ] **Step 3: Implement** — in `magicat/modules/captions/analyzer.py`:

Add imports: `from PIL import Image`.

Add a module-level helper:

```python
CROP_MARGIN = 0.02   # normalized margin around the caption bbox


def save_crop(frame_path: Path, bbox, dest: Path) -> str:
    """Cut the caption region (plus margin) out of a frame; returns path."""
    with Image.open(frame_path) as img:
        width, height = img.size
        x, y, w, h = bbox
        box = (max(0, int((x - CROP_MARGIN) * width)),
               max(0, int((y - CROP_MARGIN) * height)),
               min(width, int((x + w + CROP_MARGIN) * width)),
               min(height, int((y + h + CROP_MARGIN) * height)))
        img.crop(box).save(dest)
    return str(dest)
```

In `run()`, inside the per-segment style loop (which already computes `mid_t` and `frame`), persist up to three crops — first / middle / last frame of the segment's span:

```python
        crops_dir = ws.media_dir / "caption_crops"
        crops_dir.mkdir(parents=True, exist_ok=True)
        for i, seg in enumerate(segments):
            mid_t = (seg["t_start"] + seg["t_end"]) / 2
            frame = min(samples, key=lambda s: abs(s.t - mid_t))
            crop_times = {seg["t_start"], mid_t,
                          max(seg["t_start"], seg["t_end"] - 1.0 / SAMPLE_FPS)}
            seg["crops"] = []
            for j, ct in enumerate(sorted(crop_times)):
                src_frame = min(samples, key=lambda s: abs(s.t - ct))
                seg["crops"].append(save_crop(
                    src_frame.path, seg["bbox"],
                    crops_dir / f"seg_{i:03d}_{j}.png"))
            ...existing style block (fill/size/alignment) unchanged...
```

(Restructure the existing `for seg in segments:` loop into `for i, seg in enumerate(segments):` and keep the existing style assignments after the crop code. `crop_times` is a set, so a very short segment yields 1-2 unique crops — matching the test's `1 <= len <= 3`.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_caption_crops.py tests/test_caption_analyzer.py -v`
Expected: all PASS (existing caption-analyzer tests confirm no regression; their segments now also carry crops — the schema default keeps old assertions valid). Full suite: 115 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat/modules/captions/analyzer.py tests/test_caption_crops.py
git commit -m "feat: persist caption crops for font identification"
```

---

### Task 4: Font matcher core (stroke-normalized IoU — verified design)

**Files:**
- Create: `magicat/modules/captions/font_matcher.py`
- Test: `tests/test_font_matcher.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_font_matcher.py`:

```python
# tests/test_font_matcher.py
"""Render-and-compare matcher, verified on the Windows system font set
(15/15 across clean / JPEG40+blur / no-cv2 regimes, incl. Impact).
Heavy-OUTLINE captions remain known-fragile (research: <0.02 margins on
near-twins) - these tests pin the verified-good regimes only.

CRITICAL SYMMETRY RULE (panel-review finding): queries here are RAW
canvases from render_sample() - exactly what a wild video crop looks
like - so that query and candidates each pass through prep_crop exactly
once inside identify(). Feeding an already-ink-normalized image as the
query double-crops it and deterministically breaks heavy fonts (Impact
self-score collapses to 0.195). Do not "simplify" the tests to use the
candidate pipeline as the query."""
from pathlib import Path

import pytest
from PIL import Image

from magicat.modules.captions import font_matcher
from magicat.modules.captions.font_matcher import FontMatcher, render_sample

WIN_FONTS = Path("C:/Windows/Fonts")
FIVE = ["arial", "ariblk", "impact", "comic", "bahnschrift"]

pytestmark = pytest.mark.skipif(
    not all((WIN_FONTS / f"{k}.ttf").is_file() for k in FIVE),
    reason="system test fonts unavailable")


@pytest.fixture(scope="module")
def matcher() -> FontMatcher:
    m = FontMatcher.from_dirs([str(WIN_FONTS)])
    # narrow to the 5-font benchmark set for determinism
    m.fonts = {k: m.fonts[k] for k in FIVE}
    return m


def test_from_dirs_finds_fonts_and_env_extends(tmp_path, monkeypatch):
    monkeypatch.setenv("MAGICAT_FONT_DIRS", str(tmp_path))
    (tmp_path / "myfont.ttf").write_bytes(b"not a real font")
    m = FontMatcher.from_dirs([str(WIN_FONTS)])
    assert "arial" in m.fonts
    assert "myfont" in m.fonts            # env dir merged in


def test_clean_render_confusion_matrix(matcher):
    # every font's own (raw, wild-crop-like) render must win vs the others
    for key in FIVE:
        crop = render_sample(matcher.fonts[key], "HELLO WORLD")
        result = matcher.identify(crop, "HELLO WORLD")
        assert result.font_key == key, \
            f"{key} misidentified as {result.font_key}"
        assert result.score > 0.99        # self-match is ~1.0 by symmetry


def test_degraded_crop_still_wins(matcher, tmp_path):
    # JPEG q40 + slight blur (realistic video-crop degradation)
    from PIL import ImageFilter
    for key in FIVE:
        crop = render_sample(matcher.fonts[key], "HELLO WORLD")
        crop = crop.filter(ImageFilter.GaussianBlur(0.6))
        p = tmp_path / f"degraded_{key}.jpg"
        crop.convert("RGB").save(p, quality=40)
        with Image.open(p) as degraded:
            result = matcher.identify(degraded, "HELLO WORLD")
        assert result.font_key == key


def test_result_shape_and_confidence(matcher):
    crop = render_sample(matcher.fonts["arial"], "HELLO WORLD")
    r = matcher.identify(crop, "HELLO WORLD")
    assert r.ranked[0][0] == r.font_key
    assert 0.0 <= r.score <= 1.0
    assert r.margin >= 0.0
    assert isinstance(r.confident, bool)
    assert len(r.ranked) == len(FIVE)


def test_no_cv2_fallback_still_works_clean(matcher, monkeypatch):
    monkeypatch.setattr(font_matcher, "_HAVE_CV2", False)
    for key in FIVE:
        crop = render_sample(matcher.fonts[key], "HELLO WORLD")
        result = matcher.identify(crop, "HELLO WORLD")
        assert result.font_key == key


def test_no_fonts_raises():
    with pytest.raises(RuntimeError, match="no candidate fonts"):
        FontMatcher(fonts={}).identify(
            Image.new("L", (100, 40), 0), "X")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_font_matcher.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement** — create `magicat/modules/captions/font_matcher.py` with the research-verified design:

```python
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
            try:
                # SYMMETRY: candidate passes through prep_crop exactly like
                # the query (raw render -> single ink-crop) - see
                # render_sample docstring for the verified failure mode
                candidate = prep_crop(render_sample(path, text, NORM_H),
                                      NORM_H)
            except Exception:            # unrenderable font file: skip it
                continue
            scores[key] = score_pair(q_mask, to_mask(candidate))
        if not scores:
            raise RuntimeError("no candidate fonts could be rendered")
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        best_key, best_score = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_score - second
        return MatchResult(best_key, best_score, margin,
                           margin >= MARGIN_CONFIDENT, ranked)
```

NOTE for the implementer: `test_no_cv2_fallback_still_works_clean` monkeypatches the module-level `_HAVE_CV2` flag — `stroke_normalize` must read the flag at CALL time (`if not _HAVE_CV2`), exactly as written above; do not capture it in a default argument.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_font_matcher.py -v`
Expected: 6 PASS (each in well under 10 s; identify() over 5 fonts is ~5 renders + 5 IoUs). Full suite: 121 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat/modules/captions/font_matcher.py tests/test_font_matcher.py
git commit -m "feat: render-and-compare font matcher with stroke-normalized IoU"
```

---

### Task 5: OFL font bundle + default font dirs

**Files:**
- Create: `assets/fonts/<family>/<font>.ttf` + `assets/fonts/<family>/OFL.txt` for: Montserrat, BebasNeue, Anton, Oswald, Roboto, TikTokSans
- Create: `magicat/modules/captions/font_dirs.py`
- Test: `tests/test_font_matcher.py` (append 2 tests)

- [ ] **Step 1: Download the OFL fonts** (network step — these are all SIL Open Font License 1.1; the OFL.txt MUST travel with each font):

For each of these, download the Regular (and Bold where listed) TTF plus the license into `assets/fonts/<family>/`:

| Family | Source (github raw, `main` branch) | Files |
|---|---|---|
| Montserrat | `https://github.com/JulietaUla/Montserrat/raw/master/fonts/ttf/` | `Montserrat-Regular.ttf`, `Montserrat-Bold.ttf`; license `https://github.com/JulietaUla/Montserrat/raw/master/OFL.txt` |
| BebasNeue | `https://github.com/google/fonts/raw/main/ofl/bebasneue/` | `BebasNeue-Regular.ttf`; license `OFL.txt` from the same folder |
| Anton | `https://github.com/google/fonts/raw/main/ofl/anton/` | `Anton-Regular.ttf`; `OFL.txt` |
| Oswald | `https://github.com/google/fonts/raw/main/ofl/oswald/` | `Oswald[wght].ttf` (variable) — if PIL fails to load it in Step 3's sanity check, fall back to a static Regular from `https://github.com/vernnobile/OswaldFont/raw/master/3.0/Roman/400/Oswald-Regular.ttf` |
| Roboto | `https://github.com/google/fonts/raw/main/ofl/roboto/` | `Roboto[wdth,wght].ttf` — same variable-font caveat; static fallback `https://github.com/googlefonts/roboto-2/raw/main/src/hinted/Roboto-Regular.ttf` |
| TikTokSans | `https://github.com/google/fonts/raw/main/ofl/tiktoksans/` | `TikTokSans[opsz,slnt,wdth,wght].ttf`; `OFL.txt` |

Download with PowerShell `Invoke-WebRequest -Uri <url> -OutFile <path>` (or curl.exe). The variable-font filenames contain literal brackets — PERCENT-ENCODE them in the URL (`Oswald%5Bwght%5D.ttf`, `Roboto%5Bwdth,wght%5D.ttf`, `TikTokSans%5Bopsz,slnt,wdth,wght%5D.ttf`); curl rejects literal `[`. If a URL 404s, check the folder listing via the GitHub web UI / API for the actual filename and adapt; if a family cannot be fetched at all, SKIP it, note it in your report, and continue — the matcher must work with whatever subset exists.

REGARDLESS of download outcomes, create `assets/fonts/README.md` (committed, so the bundle root ALWAYS exists even if every download failed):

```markdown
# Bundled caption fonts

OFL-licensed faces extending the font matcher beyond the OS set (each
family ships with its OFL.txt - the license must travel with the font).
TikTok "Classic" maps to Montserrat as the closest FREE proxy for the
commercial Proxima Nova. Add private fonts via $env:MAGICAT_FONT_DIRS.
```

Sanity-check every downloaded file loads: `.venv/Scripts/python -c "from PIL import ImageFont; import pathlib; [ImageFont.truetype(str(p), 32) for p in pathlib.Path('assets/fonts').rglob('*.ttf')]; print('ok')"` — remove any file that fails and note it.

- [ ] **Step 2: Write the failing tests** — append to `tests/test_font_matcher.py`:

```python
def test_default_font_dirs_include_bundle_and_system():
    from magicat.modules.captions.font_dirs import default_font_dirs
    dirs = default_font_dirs()
    assert any("assets" in d for d in dirs)
    assert any("Fonts" in d for d in dirs)


def test_bundled_fonts_load_if_present():
    from magicat.modules.captions.font_dirs import bundle_dirs
    m = FontMatcher.from_dirs(bundle_dirs())
    # bundle may be partial (network at build time) - whatever exists loads
    for key, path in m.fonts.items():
        from PIL import ImageFont
        ImageFont.truetype(path, 32)   # raises on a broken file
```

- [ ] **Step 3: Implement** — create `magicat/modules/captions/font_dirs.py`:

```python
# magicat/modules/captions/font_dirs.py
"""Font search locations: bundled OFL fonts + the OS font dir + user dirs.

Bundled fonts are free proxies for the short-form universe (the real
TikTok 'Classic' face is the commercial Proxima Nova - Montserrat is the
documented closest free substitute). Users add their own via the
MAGICAT_FONT_DIRS env var (os.pathsep-separated), which wins on collision.
"""
from __future__ import annotations

import sys
from pathlib import Path

ASSETS_FONTS = Path(__file__).resolve().parents[3] / "assets" / "fonts"


def bundle_dirs() -> list[str]:
    if not ASSETS_FONTS.is_dir():
        return []
    # the root is always included (it exists in-repo via its README even
    # when every font download failed), then any family subdirectories
    return [str(ASSETS_FONTS)] + [
        str(d) for d in sorted(ASSETS_FONTS.iterdir()) if d.is_dir()]


def system_font_dirs() -> list[str]:
    if sys.platform == "win32":
        return ["C:/Windows/Fonts"]
    return ["/usr/share/fonts", "/System/Library/Fonts"]


def default_font_dirs() -> list[str]:
    # bundle first, system second: a system font with the same stem wins,
    # and MAGICAT_FONT_DIRS (applied by FontMatcher.from_dirs) wins over both
    return bundle_dirs() + system_font_dirs()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_font_matcher.py -v`
Expected: 8 PASS. Full suite: 123 passed, 2 skipped.

- [ ] **Step 5: Commit** (note: committing binary font files + licenses is intended)

```bash
git add assets magicat/modules/captions/font_dirs.py tests/test_font_matcher.py
git commit -m "feat: OFL font bundle and default font search dirs"
```

---

### Task 6: Wire font identification into CaptionAnalyzer

**Files:**
- Modify: `magicat/modules/captions/analyzer.py`
- Test: `tests/test_font_wiring.py` (new)

- [ ] **Step 1: Write the failing tests** — create `tests/test_font_wiring.py`:

```python
# tests/test_font_wiring.py
from pathlib import Path

import pytest

from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import Manifest, Source
from magicat.modules.captions.analyzer import CaptionAnalyzer
from magicat.modules.captions.ocr import OcrLine
from magicat.modules.ingest import IngestAnalyzer


from magicat.modules.captions.font_matcher import FontMatcher

WIN_FONTS = "C:/Windows/Fonts"
FIVE = ["arial", "ariblk", "impact", "comic", "bahnschrift"]
# the matcher mis-prefers condensed siblings on real OCR crops (verified:
# Arial Narrow outranks base Arial) - any of the family counts as a hit
ARIAL_FAMILY = {"arial", "arialbd", "arialn", "arialnb", "ariblk"}


def five_font_matcher() -> FontMatcher:
    m = FontMatcher.from_dirs([WIN_FONTS])
    m.fonts = {k: m.fonts[k] for k in FIVE if k in m.fonts}
    return m


class RealTextEngine:
    """Reports the caption_video's real burned-in caption region, so the
    persisted crops contain ACTUAL Arial glyphs (fixture burns
    'HELLO WORLD' at t=1-3, bottom-center, 480x854)."""

    def read(self, image):
        n = int(image.stem.split("_")[1])
        if 7 <= n <= 14:                       # safely inside t=1..3
            return [OcrLine(text="HELLO WORLD",
                            bbox=(0.20, 0.79, 0.60, 0.07), confidence=0.95)]
        return []


@pytest.fixture()
def analyzed(caption_video, tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(caption_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    analyzer = CaptionAnalyzer()
    monkeypatch.setattr(analyzer, "engine_factory",
                        lambda: RealTextEngine())
    monkeypatch.setattr(analyzer, "matcher_factory", five_font_matcher)
    return analyzer.run(m, ws), m


def test_font_candidates_populated(analyzed):
    patch, m = analyzed
    style = patch["captions"]["segments"][0]["style"]
    cands = style["font_candidates"]
    assert 1 <= len(cands) <= 3
    for c in cands:
        assert c["name"]
        assert 0.0 < c["confidence"] <= 1.0    # real text -> nonzero scores
    scores = [c["confidence"] for c in cands]
    assert scores == sorted(scores, reverse=True)
    # real Arial glyphs against the 5-font set: arial family must lead
    assert cands[0]["name"] in ARIAL_FAMILY
    m2 = apply_patch(m, patch)   # round-trips the schema
    assert m2.captions.segments[0].style.font_candidates


def test_font_family_only_when_confident(analyzed):
    patch, _ = analyzed
    style = patch["captions"]["segments"][0]["style"]
    if style["font_family"] is not None:
        assert style["font_family"] == style["font_candidates"][0]["name"]


def test_real_arial_caption_with_real_ocr(caption_video, tmp_path):
    # full path: real OCR + full system font dir. Verified behavior: the
    # winner is in the Arial family but is typically Arial NARROW (the
    # width-fit normalization favors condensed siblings; base arial ranks
    # ~#6 of 338). The assertion is therefore family-level, top-3.
    pytest.importorskip("rapidocr", reason="rapidocr not installed")
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(caption_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    patch = CaptionAnalyzer().run(m, ws)
    style = patch["captions"]["segments"][0]["style"]
    top3 = [c["name"] for c in style["font_candidates"]]
    assert any(name in ARIAL_FAMILY for name in top3), top3


def test_blank_crop_emits_no_candidates(fixture_video, tmp_path,
                                        monkeypatch):
    # color-bar fixture has no glyphs in the reported bbox: every score is
    # ~0 and the analyzer's MIN_FONT_SCORE floor suppresses the noise
    class BlankEngine:
        def read(self, image):
            n = int(image.stem.split("_")[1])
            if 6 <= n <= 15:
                return [OcrLine(text="FONT TEST",
                                bbox=(0.25, 0.8, 0.5, 0.06),
                                confidence=0.95)]
            return []

    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    analyzer = CaptionAnalyzer()
    monkeypatch.setattr(analyzer, "engine_factory", lambda: BlankEngine())
    monkeypatch.setattr(analyzer, "matcher_factory", five_font_matcher)
    patch = analyzer.run(m, ws)
    style = patch["captions"]["segments"][0]["style"]
    assert style["font_candidates"] == []
    assert style["font_family"] is None


def test_no_fonts_available_degrades_gracefully(caption_video, tmp_path,
                                                monkeypatch):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(caption_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    analyzer = CaptionAnalyzer()
    monkeypatch.setattr(analyzer, "engine_factory",
                        lambda: RealTextEngine())
    monkeypatch.setattr(analyzer, "matcher_factory",
                        lambda: FontMatcher(fonts={}))
    patch = analyzer.run(m, ws)
    style = patch["captions"]["segments"][0]["style"]
    assert style["font_candidates"] == []
    assert style["font_family"] is None
    assert patch["layers_status"] == {"captions": "ok"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_font_wiring.py -v`
Expected: FAIL — KeyError `font_candidates` / AttributeError `matcher_factory`.

- [ ] **Step 3: Implement** — in `magicat/modules/captions/analyzer.py`:

Add imports:

```python
import logging

from PIL import Image as PILImage

from magicat.modules.captions.font_dirs import default_font_dirs
from magicat.modules.captions.font_matcher import FontMatcher

log = logging.getLogger(__name__)
```

Add to the class, under `engine_factory`:

```python
    matcher_factory = staticmethod(
        lambda: FontMatcher.from_dirs(default_font_dirs()))  # injectable
```

In `run()`, after the crops/style loop, add font identification per segment (uses the FIRST crop and the segment's text; failures degrade to no candidates — fonts must never fail the captions layer):

```python
MIN_FONT_SCORE = 0.05   # below this, the "match" is noise (blank crops)
```

(module level, near SAMPLE_FPS), and in `run()`:

```python
        matcher = self.matcher_factory()
        for seg in segments:
            seg["style"]["font_family"] = None
            seg["style"]["font_candidates"] = []
            if not seg["crops"]:
                continue
            try:
                with PILImage.open(seg["crops"][0]) as crop:
                    result = matcher.identify(crop, seg["text"])
            except Exception as exc:
                log.warning("font identification failed: %s", exc)
                continue
            if result.score < MIN_FONT_SCORE:
                continue   # blank/garbage crop: no candidates beat noise
            seg["style"]["font_candidates"] = [
                {"name": name, "confidence": round(score, 4)}
                for name, score in result.ranked[:3]]
            if result.confident:
                seg["style"]["font_family"] = result.font_key
```

(The `style` dict already exists from the fill/size/alignment block; these keys extend it. `FontCandidate.confidence` carries the raw IoU score 0..1 — documented as a similarity, not a probability.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_font_wiring.py tests/test_caption_analyzer.py tests/test_caption_crops.py -v`
Expected: all PASS. Only `test_real_arial_caption_with_real_ocr` runs the full ~338-font system matcher (measured ~2.4 s for identify() + ~6 s OCR — well under any timeout); every other wiring test uses the narrowed 5-font matcher. NOTE: the pre-existing caption-analyzer e2e tests (test_caption_analyzer.py) now ALSO run the full matcher via the default matcher_factory — expect them to slow by a few seconds each, still fine. Full suite: 128 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat/modules/captions/analyzer.py tests/test_font_wiring.py
git commit -m "feat: caption font identification wired into analyzer"
```

---

### Task 7: Single-pass preview render with music

**Files:**
- Modify: `magicat/modules/render_preview.py`
- Test: `tests/test_render_music.py` (new); existing `tests/test_render.py` must keep passing unchanged

- [ ] **Step 1: Write the failing tests** — create `tests/test_render_music.py`:

```python
# tests/test_render_music.py
from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import Manifest, Source
from magicat.modules.cuts_pyscenedetect import CutDetector
from magicat.modules.ingest import IngestAnalyzer
from magicat.modules.render_preview import PreviewRenderer, build_filtergraph
from tests.conftest import probe_duration, run_ffmpeg


def test_filtergraph_no_music():
    fg = build_filtergraph([(0.0, 2.0), (2.0, 4.0)], with_music=False)
    assert "concat=n=2:v=1:a=1[vout][aout]" in fg
    assert "amix" not in fg


def test_filtergraph_with_music():
    fg = build_filtergraph([(0.0, 2.0)], with_music=True,
                           music_offset_s=1.5, music_volume=0.8)
    assert "adelay=1500:all=1" in fg
    assert "amix=inputs=2:duration=first:normalize=0" in fg
    assert "[vout][aconcat]" in fg


def music_fixture(tmp_path):
    p = tmp_path / "music.mp3"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=880:duration=30",
                "-c:a", "libmp3lame", str(p)])
    return p


def analyzed_manifest(fixture_video, ws):
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    m = apply_patch(m, CutDetector().run(m, ws))
    return m


def test_preview_with_music_keeps_video_duration(fixture_video, tmp_path):
    # 30s music on a 6s video: duration=first must clamp to 6s (the old
    # two-pass approach would have drifted/overrun)
    ws = Workspace(tmp_path / "job")
    m = analyzed_manifest(fixture_video, ws)
    music = music_fixture(tmp_path)
    audio = m.audio.model_dump(mode="json")
    audio["music"] = {
        "detected": True, "title": "T", "artist": "A",
        "timeline_offset": 1.0,
        "song_segment": {"start_in_song": 0.0, "duration": 5.0},
        "acquisition": {"status": "acquired", "file": str(music),
                        "links": {}},
    }
    m = apply_patch(m, {"audio": audio})
    out = PreviewRenderer().export(m, ws)
    assert abs(probe_duration(out) - (m.source.duration or 0)) < 0.3


def test_preview_without_music_unchanged(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    m = analyzed_manifest(fixture_video, ws)
    out = PreviewRenderer().export(m, ws)
    assert abs(probe_duration(out) - (m.source.duration or 0)) < 0.3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_render_music.py -v`
Expected: FAIL — ImportError (`build_filtergraph` doesn't exist).

- [ ] **Step 3: Rewrite** `magicat/modules/render_preview.py` (full replacement — the per-segment-encode + concat-demuxer approach goes away):

```python
# magicat/modules/render_preview.py
"""Preview exporter: single-pass filter_complex render (verified shape).

One ffmpeg invocation cuts every shot (trim/atrim + PTS reset), concats
them, and - when music was acquired - mixes it in at timeline_offset.
Single-pass eliminates the per-segment AAC priming drift of the old
two-pass approach (~80ms over 3 segments) that would desync music.

Load-bearing flags (empirically verified):
  amix duration=first  - default 'longest' overruns video length
  amix normalize=0     - default halves source dialog everywhere
  adelay ...:all=1     - one value for every channel, any channel count
"""
from __future__ import annotations

from pathlib import Path

from magicat.core.ffmpeg import run_ffmpeg
from magicat.core.registry import register_exporter
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest

MUSIC_VOLUME = 0.8


def build_filtergraph(segments: list[tuple[float, float]], with_music: bool,
                      music_offset_s: float = 0.0,
                      music_volume: float = MUSIC_VOLUME) -> str:
    parts: list[str] = []
    concat_inputs: list[str] = []
    for i, (start, end) in enumerate(segments):
        parts.append(f"[0:v]trim=start={start:.3f}:end={end:.3f},"
                     f"setpts=PTS-STARTPTS[v{i}]")
        parts.append(f"[0:a]atrim=start={start:.3f}:end={end:.3f},"
                     f"asetpts=PTS-STARTPTS[a{i}]")
        concat_inputs.append(f"[v{i}][a{i}]")
    n = len(segments)
    if with_music:
        parts.append("".join(concat_inputs)
                     + f"concat=n={n}:v=1:a=1[vout][aconcat]")
        delay_ms = int(round(music_offset_s * 1000))
        parts.append(f"[1:a]volume={music_volume},"
                     f"adelay={delay_ms}:all=1[music]")
        parts.append("[aconcat][music]"
                     "amix=inputs=2:duration=first:normalize=0[aout]")
    else:
        parts.append("".join(concat_inputs)
                     + f"concat=n={n}:v=1:a=1[vout][aout]")
    return ";".join(parts)


@register_exporter
class PreviewRenderer:
    format = "preview_mp4"

    def export(self, manifest: Manifest, ws: Workspace) -> Path:
        if not manifest.shots:
            raise ValueError("no shots in manifest - nothing to render")
        source = Path(manifest.source.file)

        music = manifest.audio.music
        music_file = (music.acquisition.file
                      if music.detected and music.acquisition.file else None)

        segments = [(shot.start, shot.end) for shot in manifest.shots]
        out = ws.exports_dir / "preview.mp4"
        args = ["-i", str(source)]
        if music_file:
            args += ["-i", music_file]
        args += [
            "-filter_complex",
            build_filtergraph(segments, with_music=music_file is not None,
                              music_offset_s=music.timeline_offset),
            "-map", "[vout]", "-map", "[aout]",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
            str(out),
        ]
        run_ffmpeg(args)
        return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_render_music.py tests/test_render.py tests/test_pipeline.py -v`
Expected: all PASS — the existing render tests (full-timeline duration, no-shots guard, relative-workdir) hold against the rewrite; pipeline e2e holds. Full suite: 132 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat/modules/render_preview.py tests/test_render_music.py
git commit -m "feat: single-pass preview render with music mixed at timeline offset"
```

---

### Task 8: SRT writer + xmeml v5 generator

**Files:**
- Create: `magicat/modules/export/__init__.py` (empty), `magicat/modules/export/srt.py`, `magicat/modules/export/fcp7.py`
- Modify: `pyproject.toml` (dev extra)
- Test: `tests/test_export_srt.py`, `tests/test_export_fcp7.py`

- [ ] **Step 1: Add the dev validator dependency** — in `pyproject.toml`:

```toml
dev = ["pytest>=8", "opentimelineio>=0.18", "otio-fcp-adapter>=1.0"]
```

Run `.venv/Scripts/python -m pip install -e .[dev]`. (OTIO 0.18.1 ships a cp312 wheel; otio-fcp-adapter 1.0.0 is pure-python. They are TEST-ONLY — production code must never import them.)

- [ ] **Step 2: Write the failing tests** — create `tests/test_export_srt.py`:

```python
# tests/test_export_srt.py
from magicat.manifest.schema import Manifest
from magicat.modules.export.srt import srt_timestamp, to_srt


def test_srt_timestamp():
    assert srt_timestamp(0.0) == "00:00:00,000"
    assert srt_timestamp(1.5) == "00:00:01,500"
    assert srt_timestamp(3661.25) == "01:01:01,250"


def test_to_srt_two_segments():
    m = Manifest(job_id="j", captions={"segments": [
        {"text": "HELLO WORLD", "t_start": 1.0, "t_end": 3.0},
        {"text": "SECOND LINE", "t_start": 3.5, "t_end": 5.2},
    ]})
    srt = to_srt(m)
    blocks = srt.strip().split("\n\n")
    assert len(blocks) == 2
    assert blocks[0].splitlines() == [
        "1", "00:00:01,000 --> 00:00:03,000", "HELLO WORLD"]
    assert blocks[1].splitlines() == [
        "2", "00:00:03,500 --> 00:00:05,200", "SECOND LINE"]
    assert srt.endswith("\n")


def test_to_srt_empty():
    assert to_srt(Manifest(job_id="j")) == ""
```

And create `tests/test_export_fcp7.py`:

```python
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
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_export_srt.py tests/test_export_fcp7.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 4: Implement** — create `magicat/modules/export/srt.py`:

```python
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
```

And create `magicat/modules/export/fcp7.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_export_srt.py tests/test_export_fcp7.py -v`
Expected: 3 + 7 PASS (the OTIO round-trip runs because Step 1 installed the dev validators). Full suite: 142 passed, 2 skipped.

- [ ] **Step 6: Commit**

```bash
git add magicat/modules/export pyproject.toml tests/test_export_srt.py tests/test_export_fcp7.py
git commit -m "feat: SRT writer and hand-rolled xmeml v5 generator"
```

---

### Task 9: Report builder + HTML exporter + pipeline report patch

**Files:**
- Create: `magicat/modules/report.py`
- Modify: `magicat/core/pipeline.py`
- Test: `tests/test_report.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_report.py`:

```python
# tests/test_report.py
from magicat.core.workspace import Workspace
from magicat.manifest.schema import LayerState, Manifest, Shot, Source
from magicat.modules.report import ReportExporter, build_report


def rich_manifest() -> Manifest:
    return Manifest(
        job_id="job12345678",
        source=Source(file="C:/m/source.mp4", url="https://t.example/v/1",
                      platform="tiktok", fps=30.0, resolution="480x854",
                      duration=6.0),
        shots=[Shot(id="shot_000", start=0.0, end=2.0,
                    keyframes=["C:/m/kf0.jpg"]),
               Shot(id="shot_001", start=2.0, end=6.0,
                    keyframes=["C:/m/kf1.jpg"])],
        audio={"music": {
            "detected": True, "title": "Around the World",
            "artist": "Daft Punk", "provider": "audd",
            "timeline_offset": 1.0,
            "song_segment": {"start_in_song": 30.0, "duration": 5.0},
            "acquisition": {"status": "acquired", "file": "C:/m/music.mp3",
                            "links": {"spotify": "https://sp/x",
                                      "soundcloud": "https://sc/y"}},
        }},
        captions={"segments": [{
            "text": "HELLO WORLD", "t_start": 1.0, "t_end": 3.0,
            "style": {"font_family": "arial",
                      "font_candidates": [
                          {"name": "arial", "confidence": 0.91},
                          {"name": "calibri", "confidence": 0.62}],
                      "fill": "#FDFDFD", "alignment": "center"},
        }]},
        layers_status={"source": LayerState.OK, "shots": LayerState.OK,
                       "music": LayerState.OK,
                       "captions": LayerState.OK},
    )


def test_build_report_dict():
    report = build_report(rich_manifest())
    assert report["job_id"] == "job12345678"
    assert report["source"]["platform"] == "tiktok"
    assert report["shots"]["count"] == 2
    music = report["music"]
    assert music["title"] == "Around the World"
    assert music["identified_by"] == "audd"
    assert music["links"]["spotify"] == "https://sp/x"
    assert music["used_segment"] == {"start_in_song": 30.0, "duration": 5.0}
    caps = report["captions"]
    assert caps["count"] == 1
    assert caps["fonts"] == ["arial"]
    assert caps["transcript"] == ["HELLO WORLD"]
    assert report["layers"]["music"] == "ok"


def test_build_report_no_music_no_captions():
    report = build_report(Manifest(job_id="j"))
    assert report["music"]["detected"] is False
    assert report["captions"]["count"] == 0
    assert report["captions"]["fonts"] == []


def test_html_report_renders(tmp_path):
    ws = Workspace(tmp_path / "job")
    out = ReportExporter().export(rich_manifest(), ws)
    assert out.name == "report.html"
    html = out.read_text(encoding="utf-8")
    assert "Around the World" in html
    assert "Daft Punk" in html
    assert "HELLO WORLD" in html
    assert "arial" in html
    assert "https://sp/x" in html
    assert "2 shots detected" in html


def test_html_escapes_user_text(tmp_path):
    m = Manifest(job_id="j", captions={"segments": [{
        "text": "<script>alert(1)</script>", "t_start": 0.0, "t_end": 1.0,
    }]})
    ws = Workspace(tmp_path / "job")
    html = ReportExporter().export(m, ws).read_text(encoding="utf-8")
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;" in html
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_report.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement** — create `magicat/modules/report.py`:

```python
# magicat/modules/report.py
"""User-facing summary (spec section 9): a dict in manifest.report and a
standalone report.html included in every export package."""
from __future__ import annotations

import html
from pathlib import Path
from string import Template
from typing import Any

from magicat.core.registry import register_exporter
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest


def build_report(manifest: Manifest) -> dict[str, Any]:
    music = manifest.audio.music
    fonts = sorted({seg.style.font_family
                    for seg in manifest.captions.segments
                    if seg.style.font_family})
    return {
        "job_id": manifest.job_id,
        "source": {
            "url": manifest.source.url,
            "platform": manifest.source.platform,
            "duration": manifest.source.duration,
            "resolution": manifest.source.resolution,
        },
        "shots": {
            "count": len(manifest.shots),
            "keyframes": [shot.keyframes[0] for shot in manifest.shots
                          if shot.keyframes],
        },
        "music": {
            "detected": music.detected,
            "title": music.title,
            "artist": music.artist,
            "identified_by": music.provider,
            "links": dict(music.acquisition.links),
            "used_segment": {
                "start_in_song": music.song_segment.start_in_song,
                "duration": music.song_segment.duration,
            } if music.detected else None,
            "acquisition_status": music.acquisition.status,
        },
        "captions": {
            "count": len(manifest.captions.segments),
            "fonts": fonts,
            "font_candidates": [
                [c.model_dump() for c in seg.style.font_candidates]
                for seg in manifest.captions.segments],
            "transcript": [seg.text for seg in manifest.captions.segments],
        },
        "layers": {k: v.value for k, v in manifest.layers_status.items()},
    }


_PAGE = Template("""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Magicat report $job_id</title>
<style>
body{font-family:system-ui,sans-serif;max-width:720px;margin:2rem auto;
     padding:0 1rem;color:#222}
h1{font-size:1.4rem} h2{font-size:1.1rem;margin-top:1.5rem}
table{border-collapse:collapse;width:100%}
td,th{text-align:left;padding:.3rem .6rem;border-bottom:1px solid #eee}
.tag{display:inline-block;padding:.1rem .5rem;border-radius:.5rem;
     background:#eef;margin-right:.3rem;font-size:.85rem}
</style></head><body>
<h1>Magicat reconstruction report</h1>
<p>Job <code>$job_id</code> &middot; $platform &middot; $duration s
&middot; $resolution</p>
<h2>Scenes</h2><p>$shot_count shots detected.</p>
<h2>Music</h2>$music_html
<h2>Captions</h2>$captions_html
<h2>Layer status</h2><table>$layers_rows</table>
</body></html>
""")


def _esc(value: Any) -> str:
    return html.escape(str(value if value is not None else "-"))


def _render_html(report: dict[str, Any]) -> str:
    music = report["music"]
    if music["detected"]:
        links = " ".join(
            f'<a class="tag" href="{_esc(url)}">{_esc(name)}</a>'
            for name, url in music["links"].items())
        seg = music["used_segment"] or {}
        music_html = (
            f"<p><strong>{_esc(music['title'])}</strong> by "
            f"{_esc(music['artist'])} (identified by "
            f"{_esc(music['identified_by'])})<br>"
            f"Segment used: {_esc(seg.get('start_in_song'))}s for "
            f"{_esc(seg.get('duration'))}s &middot; acquisition: "
            f"{_esc(music['acquisition_status'])}<br>{links}</p>")
    else:
        music_html = "<p>No music detected.</p>"

    caps = report["captions"]
    if caps["count"]:
        fonts = ", ".join(_esc(f) for f in caps["fonts"]) or "uncertain"
        lines = "".join(f"<li>{_esc(t)}</li>" for t in caps["transcript"])
        captions_html = (f"<p>{caps['count']} caption(s); font: {fonts}</p>"
                         f"<ul>{lines}</ul>")
    else:
        captions_html = "<p>No captions detected.</p>"

    layers_rows = "".join(
        f"<tr><td>{_esc(layer)}</td><td>{_esc(state)}</td></tr>"
        for layer, state in report["layers"].items())

    return _PAGE.substitute(
        job_id=_esc(report["job_id"]),
        platform=_esc(report["source"]["platform"]),
        duration=_esc(report["source"]["duration"]),
        resolution=_esc(report["source"]["resolution"]),
        shot_count=report["shots"]["count"],
        music_html=music_html,
        captions_html=captions_html,
        layers_rows=layers_rows,
    )


@register_exporter
class ReportExporter:
    format = "report_html"

    def export(self, manifest: Manifest, ws: Workspace) -> Path:
        out = ws.exports_dir / "report.html"
        out.write_text(_render_html(build_report(manifest)),
                       encoding="utf-8")
        return out
```

- [ ] **Step 4: Wire into the pipeline** — in `magicat/core/pipeline.py`:

```python
EXPORTERS = ["preview_mp4", "report_html"]   # Task 10 appends the zip
```

Add `import magicat.modules.report  # noqa: F401` to `load_builtin_modules` (alphabetical). Then TWO report patches in `run_job` (put `from magicat.modules.report import build_report` at the top of pipeline.py with the other imports — report.py only imports core+manifest, no circularity):

1. Right BEFORE the exporter loop — so the html exporter renders a filled report (this snapshot covers ANALYSIS layers only; exporter statuses don't exist yet):

```python
    manifest = apply_patch(manifest, {"report": build_report(manifest)})
```

2. Right AFTER the exporter loop, before `ws.save_manifest(manifest)` — so the PERSISTED manifest.report also carries every exporter's layer status (the spec §9 "what was recovered" contract; the html inside the zip intentionally predates the zip itself):

```python
    manifest = apply_patch(manifest, {"report": build_report(manifest)})
```

Add this assertion to `test_pipeline_produces_zip` in Task 10's test file (it pins the post-exporter re-patch):

```python
    assert manifest.report["layers"]["premiere_resolve_zip"] == "ok"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_report.py tests/test_pipeline.py -v`
Expected: all PASS; pipeline e2e now also produces exports/report.html (existing assertions unaffected; layers_status gains `report_html: ok`). Full suite: 146 passed, 2 skipped.

- [ ] **Step 6: Commit**

```bash
git add magicat/modules/report.py magicat/core/pipeline.py tests/test_report.py
git commit -m "feat: report builder with HTML exporter, report dict in manifest"
```

---

### Task 10: Premiere/Resolve zip package + wiring + e2e + README

**Files:**
- Create: `magicat/modules/export/package.py`
- Modify: `magicat/core/pipeline.py`, `README.md`
- Test: `tests/test_export_package.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_export_package.py`:

```python
# tests/test_export_package.py
import zipfile

from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import Manifest, Source
from magicat.modules.cuts_pyscenedetect import CutDetector
from magicat.modules.export.package import PremiereResolvePackage
from magicat.modules.ingest import IngestAnalyzer
from magicat.modules.report import ReportExporter
from tests.conftest import run_ffmpeg


def prepared(fixture_video, tmp_path, with_music: bool):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    m = apply_patch(m, CutDetector().run(m, ws))
    m = apply_patch(m, {"captions": {"segments": [
        {"text": "HELLO", "t_start": 1.0, "t_end": 2.0}]}})
    if with_music:
        music = tmp_path / "music.mp3"
        run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=660:duration=10",
                    "-c:a", "libmp3lame", str(music)])
        audio = m.audio.model_dump(mode="json")
        audio["music"] = {
            "detected": True, "title": "T", "artist": "A",
            "timeline_offset": 1.0,
            "song_segment": {"start_in_song": 0.0, "duration": 5.0},
            "acquisition": {"status": "acquired", "file": str(music),
                            "links": {}},
        }
        m = apply_patch(m, {"audio": audio})
    ReportExporter().export(m, ws)   # zip includes the report
    return m, ws


def test_zip_contains_project_files(fixture_video, tmp_path):
    m, ws = prepared(fixture_video, tmp_path, with_music=True)
    out = PremiereResolvePackage().export(m, ws)
    assert out.name == "premiere_resolve.zip"
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "project.xml" in names
        assert "captions.srt" in names
        assert "report.html" in names
        assert "IMPORT_INSTRUCTIONS.txt" in names
        assert "media/source.mp4" in names
        assert "media/music.mp3" in names
        xml = zf.read("project.xml").decode("utf-8")
        # pathurls inside the zip are RELATIVE ("media/<name>") so the
        # project resolves against wherever the zip is extracted - an
        # absolute path into the (deleted) staging dir would dangle
        assert "<pathurl>media/source.mp4</pathurl>" in xml
        assert "exports/package" not in xml
        srt = zf.read("captions.srt").decode("utf-8")
        assert "HELLO" in srt


def test_zip_without_music_or_captions(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    m = apply_patch(m, CutDetector().run(m, ws))
    ReportExporter().export(m, ws)
    out = PremiereResolvePackage().export(m, ws)
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "project.xml" in names
        assert "captions.srt" not in names      # nothing to caption
        assert "media/music.mp3" not in names


def test_pipeline_produces_zip(fixture_video, tmp_path):
    from magicat.core.pipeline import run_job
    manifest = run_job(str(fixture_video), tmp_path / "job")
    assert any(e.format == "premiere_resolve_zip"
               for e in manifest.exports)
    assert (tmp_path / "job" / "exports" / "premiere_resolve.zip").is_file()
    assert manifest.report["shots"]["count"] == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_export_package.py -v`
Expected: FAIL — ModuleNotFoundError.

- [ ] **Step 3: Implement** — create `magicat/modules/export/package.py`:

```python
# magicat/modules/export/package.py
"""The deliverable: one zip containing the xmeml project, SRT captions,
report, import instructions, and the media files the project references.

The xmeml inside the zip is REWRITTEN against a manifest whose media paths
point at the zip's own media/ folder (relative pathurls survive extraction
anywhere); the original manifest is not modified.
"""
from __future__ import annotations

import shutil
import zipfile
from pathlib import Path, PureWindowsPath

from magicat.core.registry import register_exporter
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest
from magicat.modules.export.fcp7 import to_xmeml
from magicat.modules.export.srt import to_srt

INSTRUCTIONS = """Magicat project import
=======================

Adobe Premiere:
  1. Extract this zip somewhere permanent (the project references media/).
  2. File > Import... > project.xml
  3. If media appears offline, right-click > Link Media to the media/ folder.
  4. File > Import... > captions.srt for the caption track (style specs
     are in report.html - font, fill color, position).

DaVinci Resolve:
  1. Extract this zip somewhere permanent.
  2. File > Import > Timeline... > project.xml
  3. Captions are NOT in the xml: File > Import > Subtitle... >
     captions.srt (Resolve will not auto-load it).
  4. Relink media to the media/ folder if prompted.

report.html summarizes everything that was recovered (song, links, fonts).
"""


@register_exporter
class PremiereResolvePackage:
    format = "premiere_resolve_zip"

    def export(self, manifest: Manifest, ws: Workspace) -> Path:
        staging = ws.exports_dir / "package"
        if staging.is_dir():
            shutil.rmtree(staging)
        media_dir = staging / "media"
        media_dir.mkdir(parents=True)

        # bundle media + a manifest copy whose paths are RELATIVE to the
        # zip root ("media/<name>") - an absolute path into this staging
        # dir would dangle the moment we rmtree it below, leaving the
        # imported project permanently offline (panel-review finding)
        bundled = manifest.model_copy(deep=True)
        src = Path(manifest.source.file)
        shutil.copy2(src, media_dir / src.name)
        bundled.source.file = f"media/{src.name}"
        music = manifest.audio.music
        if music.detected and music.acquisition.file:
            mfile = Path(music.acquisition.file)
            if mfile.is_file():
                shutil.copy2(mfile, media_dir / mfile.name)
                bundled.audio.music.acquisition.file = f"media/{mfile.name}"

        (staging / "project.xml").write_text(to_xmeml(bundled),
                                             encoding="utf-8")
        srt = to_srt(manifest)
        if srt:
            (staging / "captions.srt").write_text(srt, encoding="utf-8")
        report = ws.exports_dir / "report.html"
        if report.is_file():
            shutil.copy2(report, staging / "report.html")
        (staging / "IMPORT_INSTRUCTIONS.txt").write_text(
            INSTRUCTIONS, encoding="utf-8")

        out = ws.exports_dir / "premiere_resolve.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    zf.write(path, path.relative_to(staging).as_posix())
        shutil.rmtree(staging)
        return out
```

- [ ] **Step 4: Wire into the pipeline** — in `magicat/core/pipeline.py`:

```python
EXPORTERS = ["preview_mp4", "report_html", "premiere_resolve_zip"]
```

(report before zip: the zip picks up exports/report.html) and add `import magicat.modules.export.package  # noqa: F401` to `load_builtin_modules` (alphabetical).

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_export_package.py tests/test_pipeline.py -v`
Expected: all PASS. Full suite: 149 passed, 2 skipped.

- [ ] **Step 6: README + smoke test.** Replace the README Status line with:

```markdown
**Status:** M3 — full v1 deliverable: cut detection, music identification +
acquisition, caption OCR + font identification, Premiere/Resolve project
export, preview with music, HTML report.
```

Append after the Music & captions section:

```markdown
## Project export (M3)

Every job now produces `exports/premiere_resolve.zip`: an FCP7-XML project
(imports into Adobe Premiere AND DaVinci Resolve), `captions.srt`,
`report.html`, import instructions, and the referenced media. Captions
travel as SRT (xmeml titles are not portable); font + style specs are in
the report. Bundled OFL fonts under `assets/fonts/` extend the font
matcher; add your own via `$env:MAGICAT_FONT_DIRS`.
```

Smoke: generate a clip and run the CLI end-to-end:

```
ffmpeg -y -hide_banner -loglevel error -f lavfi -i color=c=red:s=320x640:r=30:d=4 -f lavfi -i sine=frequency=440:duration=4 -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest smoke.mp4
.venv/Scripts/magicat run smoke.mp4
```
Verify: summary lists `report_html: ok` and `premiere_resolve_zip: ok`; the printed workdir contains exports/{preview.mp4, report.html, premiere_resolve.zip}. Delete smoke.mp4 and the jobs/ dir contents afterwards (gitignored).

- [ ] **Step 7: Commit**

```bash
git add magicat/modules/export/package.py magicat/core/pipeline.py README.md tests/test_export_package.py
git commit -m "feat: Premiere/Resolve zip package exporter; M3 README"
```

---

## Out of Scope (M3) — deliberate deferrals

- CNN-trained font classifier (spec 6.5 step 4 "train"): M3 ships the verified render-and-compare matcher instead — recorded deviation; revisit only if real-world accuracy demands it. Heavy-outline captions are known-fragile (top-3 + confident flag is the contract).
- Stroke/shadow style extraction (M2 deferral stands).
- Burned-caption preview overlay (captions stay SRT + report; burning into preview.mp4 needs the ASS/fontfile path — M4 polish if demanded).
- `.drp` native Resolve project, `.prproj` native Premiere — xmeml covers both importers.
- CapCut export + reverse search (M5); SaaS shell (M4).
- OTIO as a runtime dependency (adapter is frozen; test-only validator).
- `caption_video` fixture font fallback for non-Windows CI: deferred to M4 (Linux CI lands then); M3 caption/font tests skip cleanly when `C:/Windows/Fonts` is absent (skipif guards already in place).
- Carried to M4+ (recorded in `memory/magicat-m2-backlog.md`, not forgotten): TransNetV2 API verification on first `[transnet]` install; `detect_platform` urlparse hardening; real `shot.confidence` values for detector disagreement; the M4 prod image must install `[separation]`.

## Self-Review Notes (already applied)

- **Spec coverage:** §6.5 step 4 → Tasks 4/5/6 (matcher deviation recorded above); §7 Premiere/Resolve row → Tasks 8/10 (xmeml + SRT + zip; OTIO swapped out with recorded rationale); §7 preview row → Task 7 (music mixed at offset — also closes the M1 AAC-drift backlog); §9 → Task 9. Backlog items → Tasks 1/2/3 (each test names its item).
- **Ordering:** Task 9 (report) before Task 10 (zip) because the zip bundles report.html; EXPORTERS order encodes the same dependency at runtime.
- **Type consistency:** `CaptionSegment.crops` (T1) feeds `save_crop` (T3) feeds `matcher.identify` (T6); `Music.provider` (T1) feeds `build_report.identified_by` (T9); `build_filtergraph` signature consistent between tests and module (T7); `to_xmeml`/`to_srt` consumed by package (T10) exactly as defined (T8).
- **Counts traced:** 103→110 (T1, +7) →114 (T2, +4 new, 1 replaced in place) →115 (T3, +1) →121 (T4, +6) →123 (T5, +2) →128 (T6, +5) →132 (T7, +4) →142 (T8, +10) →146 (T9, +4) →149 (T10, +3); skips stay 2 (transnet, demucs). The OTIO round-trip counts as passed (dev extra installed in T8 Step 1).
- **Panel-review fixes applied (wf_06465071):** symmetric font-matcher pipeline (raw `render_sample` queries; candidates `prep_crop`'d inside `identify` — re-verified 15/15 locally incl. Impact across clean/degraded/no-cv2 regimes); relative zip pathurls (absolute staging paths would dangle after rmtree); bundle root guaranteed by committed README; `MIN_FONT_SCORE` floor suppresses blank-crop noise; wiring tests use real-text crops on the narrowed 5-font set; arial-FAMILY allowlist on the OCR e2e (Arial Narrow legitimately outranks base Arial); non-tautological t_end clamp test (pinned duration 5.9); dual report patch (pre-exporters for the bundled html, post-exporters for the persisted manifest).





