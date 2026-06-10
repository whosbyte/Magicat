# Magicat M2 — Audio & Captions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect the background song (which song, which part, where it sits in the video), acquire it per the configured policy, and extract burned-in captions with text/timing/position — all as manifest patches behind the existing plugin contracts.

**Architecture:** Two new analyzer packages (`magicat/modules/audio/`, `magicat/modules/captions/`) plus one acquisition analyzer, all registered in the existing registry and orchestrated by the existing sequential pipeline. External services (AudD/ACRCloud) sit behind a `MusicIdProvider` protocol so tests run fully offline with fakes; heavy ML (demucs separation) is an optional extra mirroring the M1 TransNetV2 pattern. A small pipeline-hardening task first pays down M1 review debt that M2 needs (analyzer-declared layers, exports append semantics, exporter status reporting).

**Tech Stack:** requests (AudD + ACRCloud raw HTTP), RapidOCR + onnxruntime (caption OCR — PP-OCRv4 ONNX models, pip-only on Windows), yt-dlp ≥ 2026.6.9 (SoundCloud/YouTube acquisition; first stable with the SoundCloud 404 fix), demucs-infer ≥ 4.1.3 + soundfile (optional `[separation]` extra — the only maintained Demucs that works with torch ≥ 2.6), ffmpeg/ffprobe, pytest.

**Spec:** `docs/superpowers/specs/2026-06-09-magicat-framework-design.md` §6.3 (audio analysis), §6.4 (acquisition), §6.5 steps 1–2 + partial 3 + partial 5 (font classification, full in-scene-text discrimination, and stroke/shadow extraction are M3), §5 (pipeline policy).

**Research basis (verified against official docs/PyPI 2026-06-10):**
- AudD: `POST https://api.audd.io/`, multipart `file` + `api_token` form fields; errors come back as HTTP 200 with `{"status":"error"}`; no-match is `{"status":"success","result":null}`; `result.timecode` ("MM:SS" string) = position in the recognized song where the clip plays. ≤20 s clips recommended, 10 MB cap. Free tier: one-time 300 requests.
- ACRCloud: `POST https://{host}/v1/identify` with base64(HMAC-SHA1) signature over `"POST\n/v1/identify\n{key}\n{data_type}\n1\n{timestamp}"`; `status.code` 0=hit, 1001=no match; song position at clip start = `db_begin_time_offset_ms − sample_begin_time_offset_ms`. `pyacrcloud` has NO Windows wheels — raw HTTP only. Clips ≤15 s, <5 MB.
- RapidOCR 3.8.x: `pip install rapidocr onnxruntime` (onnxruntime deliberately NOT auto-installed). `RapidOCR()` constructor downloads ~15 MB of models (network needed once); `engine(img)` → `RapidOCROutput` with `.boxes` (N,4,2 pixel quads TL,TR,BR,BL), `.txts` tuple, `.scores` tuple; ALL THREE ARE `None` for empty frames. ~0.2 s/frame CPU.
- yt-dlp: `scsearch1:`/`ytsearch1:` prefixes; preview-only SoundCloud tracks have every `format_id` ending `_preview`; post-processed file path = `info['requested_downloads'][0]['filepath']`; `license` field: SoundCloud `'all-rights-reserved'`/`'cc-by-*'`, YouTube `None`/`'Standard YouTube License'`/`'Creative Commons Attribution license (reuse allowed)'` — gate on a `"creative commons"`/`"cc-"` substring, never on None-vs-CC.
- demucs-infer 4.1.3: import namespace `demucs_infer`; `get_model("htdemucs")` (~80 MB weight download), `apply_model(model, mix[None], device="cpu", shifts=0, split=True, overlap=0.25)` — do NOT pass `segment` (htdemucs has a trained-segment cap; omitting falls back to the model's own). Stems order `['drums','bass','other','vocals']` at 44100 Hz stereo.

**Environment variables (all optional — pipeline degrades per layer without them):**

| Var | Meaning |
|---|---|
| `AUDD_API_TOKEN` | enables AudD provider |
| `ACR_HOST`, `ACR_ACCESS_KEY`, `ACR_ACCESS_SECRET` | enables ACRCloud provider |
| `MAGICAT_MUSIC_PROVIDER` | `auto` (default) \| `audd` \| `acrcloud` \| `none` |
| `MAGICAT_ACQUISITION_POLICY` | `always` (default, CEO decision — legal review flagged) \| `licensed_only` \| `link_only` |
| `MAGICAT_USE_SEPARATION` | `auto` (default: use if installed) \| `never` |

No keys → `music` layer = `skipped`; captions still run (no keys needed).

---

## File Structure

```
magicat/
  config.py                       # env-driven settings (acquisition policy) - provider-AGNOSTIC,
                                  # never imports from modules/ (plugin contract)
  manifest/schema.py              # MODIFY: Music gains duration_s field
  manifest/patch.py               # MODIFY: exports gains append-merge semantics
  core/interfaces.py              # MODIFY: Analyzer protocol gains `layer: str`
  core/pipeline.py                # MODIFY: analyzer.layer, exporter status, new ANALYZERS
  modules/
    ingest.py                     # MODIFY: + layer = "source"
    cuts_pyscenedetect.py         # MODIFY: + layer = "shots"
    cuts_transnetv2.py            # MODIFY: + layer = "shots"
    audio/
      __init__.py
      extract.py                  # WAV extraction + sliding windows (ffmpeg)
      providers.py                # SongMatch, MusicIdProvider protocol, AudD, ACRCloud,
                                  # providers_from_env() ordered fallback chain
      identify.py                 # window recognition + offset alignment (pure logic)
      separation.py               # OPTIONAL demucs-infer music-bed split (lazy import)
      analyzer.py                 # AudioAnalyzer ("audio_analysis", layer "music")
      acquire.py                  # resolver chain + policy + trim ("music_acquisition")
    captions/
      __init__.py
      sampling.py                 # frame sampling at 5 fps (ffmpeg)
      ocr.py                      # OcrLine, OcrEngine protocol, RapidOcrEngine
      clustering.py               # temporal clustering + fill-color estimation (pure)
      analyzer.py                 # CaptionAnalyzer ("caption_analysis", layer "captions")
tests/
  test_patch.py                   # MODIFY: exports append test
  test_pipeline.py                # MODIFY: exporter status, new layer expectations
  test_audio_extract.py
  test_audio_providers.py
  test_audio_identify.py
  test_audio_analyzer.py
  test_audio_separation.py
  test_audio_acquire.py
  test_caption_sampling.py
  test_caption_ocr.py
  test_caption_clustering.py
  test_caption_analyzer.py
  conftest.py                     # MODIFY: + autouse env-isolation fixture (Task 1),
                                  # + long_wav fixture (Task 2), + caption_video fixture (Task 9)
```

Every module still touches only `manifest.schema` + `core` + its own package. The two analyzers never import each other.

---

### Task 1: Dependencies + pipeline hardening (M1 review debt M2 needs)

**Files:**
- Modify: `pyproject.toml`, `magicat/core/interfaces.py`, `magicat/manifest/patch.py`, `magicat/core/pipeline.py`, `magicat/modules/ingest.py`, `magicat/modules/cuts_pyscenedetect.py`, `magicat/modules/cuts_transnetv2.py`
- Test: `tests/test_patch.py`, `tests/test_pipeline.py`, `tests/test_registry.py`

- [ ] **Step 1: Update `pyproject.toml` dependencies**

Replace the `dependencies` and `optional-dependencies` sections with:

```toml
dependencies = [
    "pydantic>=2.7",
    "typer>=0.12",
    "yt-dlp>=2026.6.9",
    "scenedetect[opencv]>=0.6.4",
    "requests>=2.32",
    "rapidocr>=3.8",
    "onnxruntime>=1.20",
    "pillow>=10",
    "numpy>=1.26",
]

[project.optional-dependencies]
dev = ["pytest>=8"]
transnet = ["transnetv2-pytorch>=1.0"]
separation = ["demucs-infer>=4.1.3", "soundfile>=0.12"]
```

Run: `.venv/Scripts/python -m pip install -e .[dev]`
Expected: installs rapidocr, onnxruntime, requests, upgrades yt-dlp. (rapidocr does NOT pull onnxruntime on its own — that is why it is listed explicitly.)

- [ ] **Step 2: Write the failing tests**

First, append an autouse environment-isolation fixture to `tests/conftest.py` — from Task 6 onward the pipeline consults provider/policy env vars, and WITHOUT this fixture a machine with real API keys would make paid AudD/ACRCloud calls (and, under the `always` policy, download real audio) during pytest:

```python
@pytest.fixture(autouse=True)
def _isolated_magicat_env(monkeypatch):
    """Tests never see ambient Magicat/provider configuration; tests that
    need a var set it explicitly via monkeypatch.setenv (composes fine)."""
    for var in ("AUDD_API_TOKEN", "ACR_HOST", "ACR_ACCESS_KEY",
                "ACR_ACCESS_SECRET", "MAGICAT_MUSIC_PROVIDER",
                "MAGICAT_ACQUISITION_POLICY", "MAGICAT_USE_SEPARATION"):
        monkeypatch.delenv(var, raising=False)
```

Then append to `tests/test_patch.py`:

```python
def test_exports_append_instead_of_replacing():
    m = Manifest(job_id="j", exports=[
        {"format": "preview_mp4", "artifact": "a.mp4"}])
    m2 = apply_patch(m, {"exports": [
        {"format": "premiere_zip", "artifact": "b.zip"}]})
    assert [e.format for e in m2.exports] == ["preview_mp4", "premiere_zip"]
```

In `tests/test_pipeline.py`, REPLACE `test_run_job_end_to_end` and `test_analyzer_failure_degrades_gracefully` with:

```python
def test_run_job_end_to_end(fixture_video, tmp_path):
    workdir = tmp_path / "job"
    manifest = run_job(str(fixture_video), workdir)

    assert manifest.layers_status["source"] == LayerState.OK
    assert manifest.layers_status["shots"] == LayerState.OK
    assert manifest.layers_status["preview_mp4"] == LayerState.OK
    assert len(manifest.shots) == 3
    assert any(e.format == "preview_mp4" for e in manifest.exports)
    assert (workdir / "manifest.json").is_file()
    assert (workdir / "exports" / "preview.mp4").is_file()


def test_analyzer_failure_degrades_gracefully(fixture_video, tmp_path,
                                              monkeypatch):
    from magicat.core import registry

    def boom(manifest, ws):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(registry.get_analyzer("cut_detection"), "run", boom)
    manifest = run_job(str(fixture_video), tmp_path / "job")
    assert manifest.layers_status["shots"] == LayerState.FAILED
    # exporter cannot render without shots and must say so now
    assert manifest.layers_status["preview_mp4"] == LayerState.FAILED
    assert (tmp_path / "job" / "manifest.json").is_file()
```

Append to `tests/test_registry.py` (inside the file, uses the existing fixture):

```python
def test_analyzer_layer_attribute():
    @registry.register_analyzer
    class WithLayer:
        name = "layered"
        layer = "mylayer"
        needs_gpu = False

        def run(self, manifest, ws):
            return {}

    assert registry.get_analyzer("layered").layer == "mylayer"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_patch.py tests/test_pipeline.py tests/test_registry.py -v`
Expected: `test_exports_append_instead_of_replacing` FAILS (exports replaced, len 1); BOTH replaced pipeline tests FAIL — the pipeline does not yet write a `preview_mp4` key into `layers_status`, so each `manifest.layers_status["preview_mp4"]` assertion raises `KeyError`; `test_analyzer_layer_attribute` PASSES already (duck typing) — that is fine, it documents the contract.

- [ ] **Step 4: Implement**

`magicat/manifest/schema.py` — add one field to the `Music` model (additive, old manifests still validate); this carries the full-song duration that ACRCloud reports, which Task 8's candidate validation needs:

```python
class Music(StrictModel):
    detected: bool = False
    title: str | None = None
    artist: str | None = None
    duration_s: float | None = None   # full song duration when the provider knows it
    provider_ids: dict[str, str] = Field(default_factory=dict)
    song_segment: SongSegment = Field(default_factory=SongSegment)
    timeline_offset: float = 0.0
    acquisition: Acquisition = Field(default_factory=Acquisition)
```

`magicat/manifest/patch.py` — in `apply_patch`, add an `exports` branch, and update the module docstring's exception list to read: "...except `layers_status`, which is merged, and `exports`, which appends.":

```python
def apply_patch(manifest: Manifest, patch: ManifestPatch) -> Manifest:
    data = manifest.model_dump(mode="json")
    for key, value in patch.items():
        if key == "layers_status":
            data["layers_status"] = {**data["layers_status"], **value}
        elif key == "exports":
            # exports accumulate across exporters; a patch appends, never replaces
            data["exports"] = [*data["exports"], *value]
        else:
            data[key] = value
    return Manifest.model_validate(data)
```

`magicat/core/interfaces.py` — add `layer` to the Analyzer protocol:

```python
@runtime_checkable
class Analyzer(Protocol):
    name: str
    layer: str          # layers_status key this analyzer owns (marked failed on crash)
    needs_gpu: bool

    def run(self, manifest: Manifest, ws: Workspace) -> ManifestPatch: ...
```

Add `layer` class attributes: `IngestAnalyzer.layer = "source"` (in `magicat/modules/ingest.py`), `CutDetector.layer = "shots"` (in `cuts_pyscenedetect.py`), `TransNetV2Detector.layer = "shots"` (in `cuts_transnetv2.py`) — one line each, right under `name`.

`magicat/core/pipeline.py` — delete `LAYER_OF_ANALYZER`; use `analyzer.layer`; record exporter status:

```python
ANALYZERS = ["cut_detection"]          # Tasks 6/8/12 append audio/captions/acquisition
EXPORTERS = ["preview_mp4"]


def run_job(input_arg: str, workdir: Path) -> Manifest:
    load_builtin_modules()
    ws = Workspace(Path(workdir).resolve())

    if input_arg.startswith(("http://", "https://")):
        source = Source(url=input_arg)
    else:
        source = Source(file=str(Path(input_arg).resolve()))
    manifest = Manifest(job_id=uuid.uuid4().hex, source=source)

    # ingest is fatal on failure
    manifest = apply_patch(manifest, registry.get_analyzer("ingest")
                           .run(manifest, ws))

    for name in ANALYZERS:
        analyzer = registry.get_analyzer(name)
        try:
            manifest = apply_patch(manifest, analyzer.run(manifest, ws))
        except Exception:
            log.exception("analyzer %s failed", name)
            manifest = apply_patch(
                manifest, {"layers_status": {analyzer.layer: "failed"}})

    for fmt in EXPORTERS:
        exporter = registry.get_exporter(fmt)
        try:
            artifact = exporter.export(manifest, ws)
            manifest = apply_patch(manifest, {
                "exports": [{"format": fmt, "artifact": str(artifact)}],
                "layers_status": {fmt: "ok"},
            })
        except Exception:
            log.exception("exporter %s failed", fmt)
            manifest = apply_patch(
                manifest, {"layers_status": {fmt: "failed"}})

    ws.save_manifest(manifest)
    return manifest
```

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python -m pytest -v`
Expected: 36 passed, 1 skipped (34 prior + 1 patch test + 1 registry test; 2 pipeline tests replaced in place).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml magicat tests
git commit -m "feat: analyzer-declared layers, exports append semantics, exporter status"
```

(The commit includes `tests/conftest.py` with the env-isolation fixture and the `Music.duration_s` schema addition.)

---

### Task 2: Audio extraction + sliding windows

**Files:**
- Create: `magicat/modules/audio/__init__.py` (empty), `magicat/modules/audio/extract.py`
- Modify: `tests/conftest.py` (add `long_wav` fixture)
- Test: `tests/test_audio_extract.py`

- [ ] **Step 1: Add a long-audio fixture to `tests/conftest.py`**

Append:

```python
@pytest.fixture(scope="session")
def long_wav(tmp_path_factory) -> Path:
    """25s mono sine WAV - long enough to produce 3 sliding windows."""
    out = tmp_path_factory.mktemp("audio") / "long.wav"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=25",
                "-ac", "1", str(out)])
    return out
```

- [ ] **Step 2: Write the failing tests** — create `tests/test_audio_extract.py`:

```python
# tests/test_audio_extract.py
from magicat.modules.audio.extract import cut_windows, extract_wav, wav_duration
from tests.conftest import probe_duration


def test_extract_wav_from_video(fixture_video, tmp_path):
    wav = extract_wav(fixture_video, tmp_path / "audio.wav")
    assert wav.is_file()
    assert abs(probe_duration(wav) - 6.0) < 0.3


def test_wav_duration(long_wav):
    assert abs(wav_duration(long_wav) - 25.0) < 0.1


def test_cut_windows_short_audio_single_window(fixture_video, tmp_path):
    wav = extract_wav(fixture_video, tmp_path / "audio.wav")
    windows = cut_windows(wav, tmp_path / "win")
    assert len(windows) == 1
    assert windows[0].t_start == 0.0
    assert abs(probe_duration(windows[0].path) - 6.0) < 0.3  # min(12, remaining)


def test_cut_windows_long_audio(long_wav, tmp_path):
    windows = cut_windows(long_wav, tmp_path / "win")
    assert [w.t_start for w in windows] == [0.0, 10.0, 20.0]
    assert abs(probe_duration(windows[0].path) - 12.0) < 0.2
    assert abs(probe_duration(windows[2].path) - 5.0) < 0.3  # 25-20 remaining


def test_cut_windows_respects_max(long_wav, tmp_path):
    windows = cut_windows(long_wav, tmp_path / "win", max_windows=2)
    assert len(windows) == 2
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_audio_extract.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'magicat.modules.audio'`

- [ ] **Step 4: Implement** — create `magicat/modules/audio/extract.py`:

```python
# magicat/modules/audio/extract.py
"""Audio extraction and sliding-window cutting for music fingerprinting.

Windows are 12s every 10s (2s overlap): AudD analyzes <=~20s and ACRCloud
recommends <=15s, so 12s mono WAV clips sit comfortably inside both caps.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


def _ffmpeg(args: list[str]) -> None:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args],
        check=True, capture_output=True,
    )


def wav_duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return float(json.loads(out)["format"]["duration"])


def extract_wav(video: Path, dest: Path, sample_rate: int = 44100) -> Path:
    """Full audio track as stereo WAV (44.1kHz - what demucs expects too)."""
    _ffmpeg(["-i", str(video), "-vn", "-ac", "2", "-ar", str(sample_rate),
             str(dest)])
    return dest


@dataclass
class AudioWindow:
    t_start: float          # seconds into the video
    path: Path


def cut_windows(wav: Path, out_dir: Path, window_s: float = 12.0,
                stride_s: float = 10.0, max_windows: int = 5,
                min_window_s: float = 3.0) -> list[AudioWindow]:
    """Cut mono fingerprinting windows. Mono halves upload size; both
    providers fingerprint mono fine."""
    out_dir.mkdir(parents=True, exist_ok=True)
    duration = wav_duration(wav)
    windows: list[AudioWindow] = []
    t = 0.0
    while t < duration and len(windows) < max_windows:
        remaining = duration - t
        if remaining < min_window_s and windows:
            break  # tail too short to fingerprint reliably
        clip = out_dir / f"win_{int(t):04d}.wav"
        _ffmpeg(["-ss", f"{t:.3f}", "-t", f"{window_s:.3f}", "-i", str(wav),
                 "-ac", "1", str(clip)])
        windows.append(AudioWindow(t_start=t, path=clip))
        t += stride_s
    return windows
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_audio_extract.py -v`
Expected: 5 PASS. Full suite: 41 passed, 1 skipped.

- [ ] **Step 6: Commit**

```bash
git add magicat/modules/audio tests/test_audio_extract.py tests/conftest.py
git commit -m "feat: audio extraction and fingerprinting windows"
```

---

### Task 3: Music-ID provider protocol + AudD provider

**Files:**
- Create: `magicat/modules/audio/providers.py`
- Test: `tests/test_audio_providers.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_audio_providers.py`:

```python
# tests/test_audio_providers.py
from pathlib import Path

import pytest

from magicat.modules.audio import providers
from magicat.modules.audio.providers import (
    AudDProvider,
    ProviderError,
    SongMatch,
    parse_timecode,
)


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


@pytest.fixture()
def clip(tmp_path) -> Path:
    p = tmp_path / "clip.wav"
    p.write_bytes(b"RIFFfake")
    return p


def test_parse_timecode():
    assert parse_timecode("02:32") == 152.0
    assert parse_timecode("1:02:03") == 3723.0
    assert parse_timecode("00:00") == 0.0


def test_audd_success(monkeypatch, clip):
    payload = {"status": "success", "result": {
        "artist": "Imagine Dragons", "title": "Warriors",
        "album": "Warriors", "timecode": "02:32",
        "song_link": "https://lis.tn/Warriors",
        "spotify": {"external_urls": {
            "spotify": "https://open.spotify.com/track/abc"}, "id": "abc"},
    }}
    captured = {}

    def fake_post(url, data=None, files=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        return FakeResponse(payload)

    monkeypatch.setattr(providers.requests, "post", fake_post)
    match = AudDProvider(api_token="tok").identify(clip)
    assert captured["url"] == "https://api.audd.io/"
    assert captured["data"]["api_token"] == "tok"
    assert match.title == "Warriors"
    assert match.artist == "Imagine Dragons"
    assert match.song_offset_s == 152.0
    assert match.provider == "audd"
    assert match.links["song_link"] == "https://lis.tn/Warriors"
    assert match.links["spotify"] == "https://open.spotify.com/track/abc"
    assert match.provider_ids["spotify"] == "abc"


def test_audd_no_match_returns_none(monkeypatch, clip):
    monkeypatch.setattr(
        providers.requests, "post",
        lambda *a, **k: FakeResponse({"status": "success", "result": None}))
    assert AudDProvider(api_token="tok").identify(clip) is None


def test_audd_api_error_raises(monkeypatch, clip):
    payload = {"status": "error",
               "error": {"error_code": 901, "error_message": "limit reached"}}
    monkeypatch.setattr(providers.requests, "post",
                        lambda *a, **k: FakeResponse(payload))
    with pytest.raises(ProviderError, match="901"):
        AudDProvider(api_token="tok").identify(clip)


def test_song_match_defaults():
    m = SongMatch(title="T", artist="A", song_offset_s=1.0, provider="x")
    assert m.provider_ids == {}
    assert m.links == {}
    assert m.duration_s is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_audio_providers.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement** — create `magicat/modules/audio/providers.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_audio_providers.py -v`
Expected: 5 PASS. Full suite: 46 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat/modules/audio/providers.py tests/test_audio_providers.py
git commit -m "feat: music-id provider protocol with AudD implementation"
```

---

### Task 4: ACRCloud provider

**Files:**
- Modify: `magicat/modules/audio/providers.py` (append class)
- Test: `tests/test_audio_providers.py` (append tests)

- [ ] **Step 1: Write the failing tests** — append to `tests/test_audio_providers.py`:

```python
ACR_HIT = {
    "status": {"code": 0, "msg": "Success", "version": "1.0"},
    "metadata": {"music": [{
        "title": "Around the World",
        "artists": [{"name": "Daft Punk"}],
        "score": 95,
        "play_offset_ms": 45000,
        "duration_ms": 428000,
        "sample_begin_time_offset_ms": 1000,
        "db_begin_time_offset_ms": 34000,
        "external_ids": {"isrc": "GBDUW0600746"},
        "external_metadata": {"spotify": {"track": {"id": "spt1"}}},
    }]},
}


def test_acrcloud_signature_string_format():
    from magicat.modules.audio.providers import ACRCloudProvider
    p = ACRCloudProvider(host="h.example.com", access_key="key",
                         access_secret="secret")
    assert p._string_to_sign("12345") == (
        "POST\n/v1/identify\nkey\naudio\n1\n12345")


def test_acrcloud_success(monkeypatch, clip):
    from magicat.modules.audio.providers import ACRCloudProvider
    captured = {}

    def fake_post(url, files=None, data=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        return FakeResponse(ACR_HIT)

    monkeypatch.setattr(providers.requests, "post", fake_post)
    p = ACRCloudProvider(host="h.example.com", access_key="key",
                         access_secret="secret")
    match = p.identify(clip)
    assert captured["url"] == "https://h.example.com/v1/identify"
    assert captured["data"]["access_key"] == "key"
    assert captured["data"]["signature"]          # present and non-empty
    # song position at clip start = (db_begin - sample_begin) / 1000
    assert match.song_offset_s == 33.0
    assert match.title == "Around the World"
    assert match.artist == "Daft Punk"
    assert match.score == 95
    assert match.duration_s == 428.0
    assert match.provider_ids["isrc"] == "GBDUW0600746"
    assert match.provider_ids["spotify"] == "spt1"
    assert match.links["spotify"] == "https://open.spotify.com/track/spt1"


def test_acrcloud_no_match(monkeypatch, clip):
    from magicat.modules.audio.providers import ACRCloudProvider
    payload = {"status": {"code": 1001, "msg": "No result", "version": "1.0"}}
    monkeypatch.setattr(providers.requests, "post",
                        lambda *a, **k: FakeResponse(payload))
    p = ACRCloudProvider(host="h", access_key="k", access_secret="s")
    assert p.identify(clip) is None


def test_acrcloud_error_raises(monkeypatch, clip):
    from magicat.modules.audio.providers import ACRCloudProvider
    payload = {"status": {"code": 3001, "msg": "wrong access key",
                          "version": "1.0"}}
    monkeypatch.setattr(providers.requests, "post",
                        lambda *a, **k: FakeResponse(payload))
    p = ACRCloudProvider(host="h", access_key="k", access_secret="s")
    with pytest.raises(ProviderError, match="3001"):
        p.identify(clip)


def test_acrcloud_negative_offset_clamped(monkeypatch, clip):
    from magicat.modules.audio.providers import ACRCloudProvider
    hit = {"status": {"code": 0, "msg": "Success", "version": "1.0"},
           "metadata": {"music": [{
               "title": "T", "artists": [{"name": "A"}], "score": 80,
               "play_offset_ms": 100,
               "sample_begin_time_offset_ms": 5000,
               "db_begin_time_offset_ms": 2000,
           }]}}
    monkeypatch.setattr(providers.requests, "post",
                        lambda *a, **k: FakeResponse(hit))
    p = ACRCloudProvider(host="h", access_key="k", access_secret="s")
    assert p.identify(clip).song_offset_s == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_audio_providers.py -v`
Expected: new tests FAIL with `ImportError: cannot import name 'ACRCloudProvider'`

- [ ] **Step 3: Implement** — append to `magicat/modules/audio/providers.py`:

```python
import base64
import hashlib
import hmac
import time


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

        best = data["metadata"]["music"][0]   # best match first
        offset_s = max(0.0, (best["db_begin_time_offset_ms"]
                             - best["sample_begin_time_offset_ms"]) / 1000.0)

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
            title=best["title"],
            artist=", ".join(a["name"] for a in best.get("artists", [])),
            song_offset_s=offset_s,
            provider=self.name,
            score=float(best.get("score", 0)),
            duration_s=duration_ms / 1000.0 if duration_ms else None,
            provider_ids=provider_ids,
            links=links,
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_audio_providers.py -v`
Expected: 10 PASS. Full suite: 51 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat/modules/audio/providers.py tests/test_audio_providers.py
git commit -m "feat: ACRCloud provider via raw HMAC-signed HTTP"
```

---

### Task 5: Window recognition + offset alignment (pure logic)

**Files:**
- Create: `magicat/modules/audio/identify.py`
- Test: `tests/test_audio_identify.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_audio_identify.py`:

```python
# tests/test_audio_identify.py
from pathlib import Path

from magicat.modules.audio.extract import AudioWindow
from magicat.modules.audio.identify import align, recognize_windows
from magicat.modules.audio.providers import ProviderError, SongMatch


def w(t: float) -> AudioWindow:
    return AudioWindow(t_start=t, path=Path(f"win_{int(t)}.wav"))


def m(offset: float, title: str = "Song", artist: str = "Artist",
      **kw) -> SongMatch:
    return SongMatch(title=title, artist=artist, song_offset_s=offset,
                     provider="fake", **kw)


class FakeProvider:
    name = "fake"

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def identify(self, clip):
        result = self.results[self.calls]
        self.calls += 1
        if isinstance(result, Exception):
            raise result
        return result


def test_recognize_windows_maps_provider_errors_to_none():
    provider = FakeProvider([m(30.0), ProviderError("quota"), None])
    windows = [w(0.0), w(10.0), w(20.0)]
    results = recognize_windows(windows, provider)
    assert results[0].song_offset_s == 30.0
    assert results[1] is None
    assert results[2] is None


def test_align_consistent_windows():
    # song plays from its 30s mark starting at video t=0
    windows = [w(0.0), w(10.0), w(20.0)]
    matches = [m(30.0), m(40.0), m(50.0)]
    music = align(windows, matches, video_duration=25.0, window_s=12.0)
    assert music["detected"] is True
    assert music["title"] == "Song"
    assert music["timeline_offset"] == 0.0
    assert music["song_segment"]["start_in_song"] == 30.0
    # matched span: first window start -> min(last start + 12, video end)
    assert music["song_segment"]["duration"] == 25.0


def test_align_rejects_outlier_window():
    windows = [w(0.0), w(10.0), w(20.0)]
    matches = [m(30.0), m(95.0), m(50.0)]   # middle anchor disagrees by 55s
    music = align(windows, matches, video_duration=25.0, window_s=12.0)
    assert music["detected"] is True
    assert music["song_segment"]["start_in_song"] == 30.0


def test_align_majority_vote_on_identity():
    windows = [w(0.0), w(10.0), w(20.0)]
    matches = [m(30.0), m(40.0), m(10.0, title="Other Song")]
    music = align(windows, matches, video_duration=25.0, window_s=12.0)
    assert music["title"] == "Song"


def test_align_music_starts_mid_video():
    # nothing matches at t=0; song matched from t=10 at its very start
    windows = [w(0.0), w(10.0), w(20.0)]
    matches = [None, m(0.0), m(10.0)]
    music = align(windows, matches, video_duration=25.0, window_s=12.0)
    assert music["timeline_offset"] == 10.0
    assert music["song_segment"]["start_in_song"] == 0.0
    assert music["song_segment"]["duration"] == 15.0


def test_align_no_matches_returns_none():
    assert align([w(0.0)], [None], video_duration=6.0, window_s=12.0) is None


def test_align_carries_metadata_from_best_match():
    windows = [w(0.0), w(10.0)]
    matches = [m(30.0, provider_ids={"spotify": "x"},
                 links={"spotify": "url"}, duration_s=200.0),
               m(40.0)]
    music = align(windows, matches, video_duration=22.0, window_s=12.0)
    assert music["provider_ids"] == {"spotify": "x"}
    assert music["acquisition"]["links"] == {"spotify": "url"}
    assert music["duration_s"] == 200.0   # consumed by acquisition validation
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_audio_identify.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement** — create `magicat/modules/audio/identify.py`:

```python
# magicat/modules/audio/identify.py
"""Turn per-window provider matches into one music description.

Alignment model: every matched window i gives an "anchor" a_i =
window_start_i - song_offset_i = the video time where the song's 0:00 would
land. A steadily-playing song gives identical anchors; we take the median
and drop windows deviating > ANCHOR_TOLERANCE_S (re-fingerprint noise,
DJ edits, lyric repeats). The matched span approximates where the music
plays in the video - M2 precision is window-level (~10s), refined later.
"""
from __future__ import annotations

import logging
import statistics
from typing import Any

from magicat.modules.audio.extract import AudioWindow
from magicat.modules.audio.providers import (
    MusicIdProvider,
    ProviderError,
    SongMatch,
)

log = logging.getLogger(__name__)

ANCHOR_TOLERANCE_S = 5.0


def recognize_windows(windows: list[AudioWindow],
                      provider: MusicIdProvider) -> list[SongMatch | None]:
    """Identify every window; provider errors degrade to no-match."""
    results: list[SongMatch | None] = []
    for window in windows:
        try:
            results.append(provider.identify(window.path))
        except (ProviderError, OSError) as exc:
            log.warning("window at %.1fs failed: %s", window.t_start, exc)
            results.append(None)
    return results


def align(windows: list[AudioWindow], matches: list[SongMatch | None],
          video_duration: float, window_s: float) -> dict[str, Any] | None:
    """Build the manifest's audio.music dict from window matches."""
    hits = [(w, m) for w, m in zip(windows, matches) if m is not None]
    if not hits:
        return None

    # majority vote on song identity
    def identity(match: SongMatch) -> tuple[str, str]:
        return (match.title.strip().lower(), match.artist.strip().lower())

    counts: dict[tuple[str, str], int] = {}
    for _, match in hits:
        counts[identity(match)] = counts.get(identity(match), 0) + 1
    winner = max(counts, key=lambda k: counts[k])
    hits = [(w, m) for w, m in hits if identity(m) == winner]

    # consensus anchor (video time of the song's 0:00), outliers dropped
    anchors = [w.t_start - m.song_offset_s for w, m in hits]
    consensus = statistics.median(anchors)
    inliers = [(w, m) for (w, m), a in zip(hits, anchors)
               if abs(a - consensus) <= ANCHOR_TOLERANCE_S]
    if not inliers:
        inliers = hits  # all "outliers": fall back rather than drop the song
    consensus = statistics.median(
        w.t_start - m.song_offset_s for w, m in inliers)

    first_w = inliers[0][0]
    last_w = inliers[-1][0]
    timeline_offset = first_w.t_start
    start_in_song = max(0.0, timeline_offset - consensus)
    span_end = min(video_duration, last_w.t_start + window_s)
    best = inliers[0][1]

    return {
        "detected": True,
        "title": best.title,
        "artist": best.artist,
        "duration_s": best.duration_s,
        "provider_ids": best.provider_ids,
        "song_segment": {
            "start_in_song": start_in_song,
            "duration": span_end - timeline_offset,
        },
        "timeline_offset": timeline_offset,
        "acquisition": {
            "status": "skipped",
            "links": best.links,
        },
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_audio_identify.py -v`
Expected: 7 PASS. Full suite: 58 passed, 1 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat/modules/audio/identify.py tests/test_audio_identify.py
git commit -m "feat: window recognition and song offset alignment"
```

---

### Task 6: Config + provider chain + AudioAnalyzer + pipeline wiring

Contract note: `magicat/config.py` is shared by multiple modules, so it must stay **provider-agnostic** (it never imports from `magicat/modules/`). Provider construction lives in the audio package (`providers_from_env()`), which returns an ORDERED chain — AudD primary, ACRCloud fallback (spec §6.3 step 2): if the first provider yields no usable result across all windows, the analyzer tries the next.

**Files:**
- Create: `magicat/config.py`, `magicat/modules/audio/analyzer.py`
- Modify: `magicat/modules/audio/providers.py` (append `providers_from_env`), `magicat/core/pipeline.py` (ANALYZERS + loader)
- Test: `tests/test_audio_analyzer.py`, `tests/test_audio_providers.py` (append)

- [ ] **Step 1: Write the failing tests** — create `tests/test_audio_analyzer.py`:

```python
# tests/test_audio_analyzer.py
import pytest

from magicat import config
from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import LayerState, Manifest, Source
from magicat.modules.audio.analyzer import AudioAnalyzer
from magicat.modules.audio.providers import SongMatch
from magicat.modules.ingest import IngestAnalyzer


class OneSongProvider:
    name = "fake"

    def identify(self, clip):
        return SongMatch(title="Fixture Song", artist="Fixture Artist",
                         song_offset_s=30.0, provider="fake",
                         links={"song_link": "https://example.com/s"})


@pytest.fixture()
def ingested(fixture_video, tmp_path):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    return m, ws


def test_no_providers_configured_skips_layer(ingested, monkeypatch):
    m, ws = ingested
    analyzer = AudioAnalyzer()
    monkeypatch.setattr(analyzer, "provider_factory", lambda: [])
    patch = analyzer.run(m, ws)
    assert patch == {"layers_status": {"music": "skipped"}}


def test_detects_song_and_fills_music_section(ingested, monkeypatch):
    m, ws = ingested
    analyzer = AudioAnalyzer()
    monkeypatch.setattr(analyzer, "provider_factory",
                        lambda: [OneSongProvider()])
    patch = analyzer.run(m, ws)
    music = patch["audio"]["music"]
    assert music["detected"] is True
    assert music["title"] == "Fixture Song"
    assert music["timeline_offset"] == 0.0
    assert music["song_segment"]["start_in_song"] == 30.0
    assert music["acquisition"]["links"]["song_link"] == "https://example.com/s"
    assert patch["layers_status"] == {"music": "ok"}
    # validates against the schema
    m2 = apply_patch(m, patch)
    assert m2.audio.music.detected is True


class NoMatchProvider:
    name = "nomatch"

    def identify(self, clip):
        return None


def test_no_match_is_ok_with_detected_false(ingested, monkeypatch):
    m, ws = ingested
    analyzer = AudioAnalyzer()
    monkeypatch.setattr(analyzer, "provider_factory",
                        lambda: [NoMatchProvider()])
    patch = analyzer.run(m, ws)
    assert patch["audio"]["music"]["detected"] is False
    assert patch["layers_status"] == {"music": "ok"}


def test_fallback_to_second_provider(ingested, monkeypatch):
    # spec 6.3 step 2: AudD primary, ACRCloud fallback - when the primary
    # finds nothing across all windows, the next provider in the chain runs
    m, ws = ingested
    analyzer = AudioAnalyzer()
    monkeypatch.setattr(analyzer, "provider_factory",
                        lambda: [NoMatchProvider(), OneSongProvider()])
    patch = analyzer.run(m, ws)
    assert patch["audio"]["music"]["detected"] is True
    assert patch["audio"]["music"]["title"] == "Fixture Song"


def test_config_acquisition_policy(monkeypatch):
    assert config.acquisition_policy() == "always"
    monkeypatch.setenv("MAGICAT_ACQUISITION_POLICY", "link_only")
    assert config.acquisition_policy() == "link_only"
    monkeypatch.setenv("MAGICAT_ACQUISITION_POLICY", "bogus")
    with pytest.raises(ValueError):
        config.acquisition_policy()


def test_pipeline_includes_audio_analysis(fixture_video, tmp_path):
    # ambient provider env is cleared by the autouse fixture (Task 1)
    from magicat.core.pipeline import run_job
    manifest = run_job(str(fixture_video), tmp_path / "job")
    assert manifest.layers_status["music"] == LayerState.SKIPPED
```

And append to `tests/test_audio_providers.py`:

```python
def test_providers_from_env(monkeypatch):
    from magicat.modules.audio.providers import providers_from_env

    assert providers_from_env() == []          # env cleared by autouse fixture

    monkeypatch.setenv("AUDD_API_TOKEN", "tok")
    assert [p.name for p in providers_from_env()] == ["audd"]

    monkeypatch.setenv("ACR_HOST", "h")
    monkeypatch.setenv("ACR_ACCESS_KEY", "k")
    monkeypatch.setenv("ACR_ACCESS_SECRET", "s")
    # auto mode: AudD primary, ACRCloud fallback (spec 6.3 step 2)
    assert [p.name for p in providers_from_env()] == ["audd", "acrcloud"]

    monkeypatch.setenv("MAGICAT_MUSIC_PROVIDER", "acrcloud")
    assert [p.name for p in providers_from_env()] == ["acrcloud"]
    monkeypatch.setenv("MAGICAT_MUSIC_PROVIDER", "none")
    assert providers_from_env() == []
    monkeypatch.setenv("MAGICAT_MUSIC_PROVIDER", "bogus")
    import pytest as _pytest
    with _pytest.raises(ValueError):
        providers_from_env()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_audio_analyzer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement config** — create `magicat/config.py`:

```python
# magicat/config.py
"""Environment-driven settings. Read at call time (never import time) so
tests can monkeypatch the environment.

Contract: this module is shared infrastructure - it must NEVER import from
magicat.modules.* (modules import config, not the other way around).
"""
from __future__ import annotations

import os

ACQUISITION_POLICIES = ("always", "licensed_only", "link_only")


def acquisition_policy() -> str:
    policy = os.environ.get("MAGICAT_ACQUISITION_POLICY", "always")
    if policy not in ACQUISITION_POLICIES:
        raise ValueError(
            f"MAGICAT_ACQUISITION_POLICY must be one of "
            f"{ACQUISITION_POLICIES}, got {policy!r}")
    return policy
```

- [ ] **Step 3b: Implement the provider chain** — append to `magicat/modules/audio/providers.py`:

```python
import os


def providers_from_env() -> list[MusicIdProvider]:
    """Ordered provider chain from env: AudD primary, ACRCloud fallback
    (spec 6.3 step 2). MAGICAT_MUSIC_PROVIDER narrows: audd|acrcloud|none.
    """
    selection = os.environ.get("MAGICAT_MUSIC_PROVIDER", "auto")
    if selection == "none":
        return []
    if selection not in ("auto", "audd", "acrcloud"):
        raise ValueError(f"unknown MAGICAT_MUSIC_PROVIDER {selection!r}")

    chain: list[MusicIdProvider] = []
    token = os.environ.get("AUDD_API_TOKEN")
    if selection in ("auto", "audd") and token:
        chain.append(AudDProvider(api_token=token))
    host = os.environ.get("ACR_HOST")
    key = os.environ.get("ACR_ACCESS_KEY")
    secret = os.environ.get("ACR_ACCESS_SECRET")
    if selection in ("auto", "acrcloud") and host and key and secret:
        chain.append(ACRCloudProvider(host=host, access_key=key,
                                      access_secret=secret))
    return chain
```

- [ ] **Step 4: Implement analyzer** — create `magicat/modules/audio/analyzer.py`:

```python
# magicat/modules/audio/analyzer.py
"""Audio analysis: extract -> (optional separation, Task 7) -> windows ->
provider chain recognition -> offset alignment -> audio.music patch."""
from __future__ import annotations

from pathlib import Path

from magicat.core.registry import register_analyzer
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest
from magicat.modules.audio.extract import cut_windows, extract_wav
from magicat.modules.audio.identify import align, recognize_windows
from magicat.modules.audio.providers import providers_from_env

WINDOW_S = 12.0


@register_analyzer
class AudioAnalyzer:
    name = "audio_analysis"
    layer = "music"
    needs_gpu = False
    provider_factory = staticmethod(providers_from_env)  # injectable

    def run(self, manifest: Manifest, ws: Workspace) -> dict:
        providers = self.provider_factory()
        if not providers:
            return {"layers_status": {"music": "skipped"}}

        wav = extract_wav(Path(manifest.source.file),
                          ws.media_dir / "audio.wav")
        fingerprint_input = wav  # Task 7 swaps in the separated music bed

        windows = cut_windows(fingerprint_input,
                              ws.media_dir / "audio_windows",
                              window_s=WINDOW_S)
        music = None
        for provider in providers:   # primary first; fallback on no result
            matches = recognize_windows(windows, provider)
            music = align(windows, matches,
                          video_duration=manifest.source.duration or 0.0,
                          window_s=WINDOW_S)
            if music is not None:
                break

        audio = manifest.audio.model_dump(mode="json")
        if music is None:
            audio["music"]["detected"] = False
        else:
            audio["music"] = music
        return {"audio": audio, "layers_status": {"music": "ok"}}
```

- [ ] **Step 5: Wire into the pipeline** — in `magicat/core/pipeline.py`:

```python
ANALYZERS = ["cut_detection", "audio_analysis"]   # Task 8/12 extend further
```

and extend the loader:

```python
def load_builtin_modules() -> None:
    """Import modules for their @register side effects."""
    import magicat.modules.audio.analyzer  # noqa: F401
    import magicat.modules.cuts_pyscenedetect  # noqa: F401
    import magicat.modules.cuts_transnetv2  # noqa: F401
    import magicat.modules.ingest  # noqa: F401
    import magicat.modules.render_preview  # noqa: F401
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_audio_analyzer.py tests/test_audio_providers.py -v` — expect 6 + 11 PASS.
Full suite: 65 passed, 1 skipped. (The pipeline e2e tests still pass: the autouse env fixture guarantees no provider is configured, so the music layer is skipped, which the Task 1 assertions don't constrain.)

- [ ] **Step 7: Commit**

```bash
git add magicat/config.py magicat/modules/audio/analyzer.py magicat/modules/audio/providers.py magicat/core/pipeline.py tests/test_audio_analyzer.py tests/test_audio_providers.py
git commit -m "feat: audio analyzer with provider fallback chain, wired into pipeline"
```

---

### Task 7: Optional Demucs separation

**Files:**
- Create: `magicat/modules/audio/separation.py`
- Modify: `magicat/modules/audio/analyzer.py`
- Test: `tests/test_audio_separation.py`

- [ ] **Step 1: Write the tests** — create `tests/test_audio_separation.py`:

```python
# tests/test_audio_separation.py
import pytest

from magicat.modules.audio import separation


def test_available_reflects_install_state():
    import importlib.util
    expected = importlib.util.find_spec("demucs_infer") is not None
    assert separation.available() is expected


def test_separation_disabled_by_env(monkeypatch):
    monkeypatch.setenv("MAGICAT_USE_SEPARATION", "never")
    assert separation.enabled() is False


def test_split_music_bed(long_wav, tmp_path):
    pytest.importorskip(
        "demucs_infer", reason="optional separation extra not installed")
    music_bed, vocals = separation.split_music_bed(long_wav, tmp_path)
    assert music_bed.is_file()
    assert vocals.is_file()
```

- [ ] **Step 2: Run tests** — `.venv/Scripts/python -m pytest tests/test_audio_separation.py -v`
Expected: FAIL (ModuleNotFoundError for magicat module; the demucs test will SKIP once the module exists).

- [ ] **Step 3: Implement** — create `magicat/modules/audio/separation.py`:

```python
# magicat/modules/audio/separation.py
"""Optional music/speech separation via demucs-infer (htdemucs).

Install: pip install -e .[separation]. All heavy imports are inside
functions; importing this module costs nothing. Music bed = drums+bass+other
(vocals stem carries both sung vocals and voiceover speech).
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path


def available() -> bool:
    return importlib.util.find_spec("demucs_infer") is not None


def enabled() -> bool:
    mode = os.environ.get("MAGICAT_USE_SEPARATION", "auto")
    if mode == "never":
        return False
    return available()


def split_music_bed(wav: Path, out_dir: Path) -> tuple[Path, Path]:
    """Returns (music_bed_path, vocals_path). Caller guards with enabled()."""
    import soundfile as sf
    import torch
    from demucs_infer.apply import apply_model
    from demucs_infer.pretrained import get_model

    model = get_model("htdemucs")   # ~80 MB one-time weight download
    model.eval()

    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T)            # (channels, samples)
    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)              # mono -> stereo
    if sr != model.samplerate:
        import torchaudio
        audio = torchaudio.functional.resample(audio, sr, model.samplerate)

    ref = audio.mean(0)
    normalized = (audio - ref.mean()) / (ref.std() + 1e-8)
    with torch.no_grad():
        # NOTE: do not pass `segment` - htdemucs has a trained-segment cap
        # and apply_model falls back to the model's own value when omitted
        sources = apply_model(model, normalized[None], device="cpu",
                              shifts=0, split=True, overlap=0.25,
                              progress=False)[0]
    sources = sources * ref.std() + ref.mean()  # (4, 2, samples)

    stems = dict(zip(model.sources, sources))   # drums, bass, other, vocals
    music_bed = stems["drums"] + stems["bass"] + stems["other"]

    out_dir.mkdir(parents=True, exist_ok=True)
    music_path = out_dir / "music_bed.wav"
    vocals_path = out_dir / "vocals.wav"
    sf.write(str(music_path), music_bed.T.numpy(), model.samplerate)
    sf.write(str(vocals_path), stems["vocals"].T.numpy(), model.samplerate)
    return music_path, vocals_path
```

- [ ] **Step 4: Hook into the analyzer** — in `magicat/modules/audio/analyzer.py`, add the import `from magicat.modules.audio import separation` and replace the `fingerprint_input = wav` line with:

```python
        fingerprint_input = wav
        speech_stem: str | None = None
        if separation.enabled():
            music_bed, vocals = separation.split_music_bed(
                wav, ws.media_dir / "stems")
            fingerprint_input = music_bed
            speech_stem = str(vocals)
```

and just before the return, persist the stem:

```python
        audio["speech_stem"] = speech_stem
```

(Place it after the `audio = manifest.audio.model_dump(...)` line and the music assignment.)

- [ ] **Step 5: Run the full suite**

Run: `.venv/Scripts/python -m pytest -v`
Expected: 67 passed, 2 skipped (demucs test skips without the extra; TransNetV2 as before).

- [ ] **Step 6: Commit**

```bash
git add magicat/modules/audio/separation.py magicat/modules/audio/analyzer.py tests/test_audio_separation.py
git commit -m "feat: optional demucs-infer music/speech separation"
```

---

### Task 8: Music acquisition (resolver chain + policy + trim)

**Files:**
- Create: `magicat/modules/audio/acquire.py`
- Modify: `magicat/core/pipeline.py` (ANALYZERS + loader)
- Test: `tests/test_audio_acquire.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_audio_acquire.py`:

```python
# tests/test_audio_acquire.py
from pathlib import Path

import pytest

from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest, Source
from magicat.modules.audio.acquire import (
    Candidate,
    MusicAcquisition,
    is_licensed_free,
    validate_candidate,
)
from tests.conftest import probe_duration, run_ffmpeg


def music_manifest(**acq_kwargs) -> Manifest:
    return Manifest(job_id="j", source=Source(file="x.mp4", duration=25.0),
                    audio={"music": {
                        "detected": True, "title": "Around the World",
                        "artist": "Daft Punk",
                        "duration_s": 428.0,
                        "song_segment": {"start_in_song": 30.0,
                                         "duration": 20.0},
                        "timeline_offset": 0.0,
                        "acquisition": {"status": "skipped",
                                        "links": {"spotify": "sp"}},
                    }})


def cand(**kw) -> Candidate:
    base = dict(url="https://soundcloud.com/x/y", title="Around the World",
                uploader="daftpunk", duration=428.0,
                license="all-rights-reserved", source="soundcloud")
    base.update(kw)
    return Candidate(**base)


def test_validate_candidate_title_and_duration():
    match_info = {"title": "Around the World", "artist": "Daft Punk",
                  "duration_s": 428.0}
    assert validate_candidate(cand(), match_info) is True
    assert validate_candidate(cand(title="totally different song xyz"),
                              match_info) is False
    assert validate_candidate(cand(duration=120.0), match_info) is False
    # unknown durations are not held against the candidate
    assert validate_candidate(cand(duration=None),
                              {"title": "Around the World",
                               "artist": "Daft Punk",
                               "duration_s": None}) is True


def test_is_licensed_free():
    assert is_licensed_free(cand(license="cc-by-sa")) is True
    assert is_licensed_free(cand(
        license="Creative Commons Attribution license (reuse allowed)")) is True
    assert is_licensed_free(cand(license="all-rights-reserved")) is False
    assert is_licensed_free(cand(license=None)) is False


@pytest.fixture()
def song_mp3(tmp_path) -> Path:
    """60s sine 'song' to trim from."""
    p = tmp_path / "song.mp3"
    run_ffmpeg(["-f", "lavfi", "-i", "sine=frequency=330:duration=60",
                "-c:a", "libmp3lame", str(p)])
    return p


def make_analyzer(monkeypatch, candidate, download_path, policy="always"):
    analyzer = MusicAcquisition()
    monkeypatch.setenv("MAGICAT_ACQUISITION_POLICY", policy)
    monkeypatch.setattr(analyzer, "prober",
                        lambda query: candidate)
    monkeypatch.setattr(analyzer, "downloader",
                        lambda c, out_dir: download_path)
    return analyzer


def test_policy_always_downloads_and_trims(tmp_path, monkeypatch, song_mp3):
    ws = Workspace(tmp_path / "job")
    analyzer = make_analyzer(monkeypatch, cand(), song_mp3)
    patch = analyzer.run(music_manifest(), ws)
    acq = patch["audio"]["music"]["acquisition"]
    assert acq["status"] == "acquired"
    trimmed = Path(acq["file"])
    assert trimmed.is_file()
    assert abs(probe_duration(trimmed) - 20.0) < 0.5   # song_segment.duration
    assert acq["license"] == "all-rights-reserved"
    assert acq["links"]["soundcloud"] == "https://soundcloud.com/x/y"
    assert acq["links"]["spotify"] == "sp"              # pre-existing kept
    assert patch["layers_status"] == {"music_acquisition": "ok"}


def test_policy_link_only_skips_download(tmp_path, monkeypatch, song_mp3):
    ws = Workspace(tmp_path / "job")
    analyzer = make_analyzer(monkeypatch, cand(), song_mp3,
                             policy="link_only")
    called = []
    monkeypatch.setattr(analyzer, "downloader",
                        lambda c, out_dir: called.append(1))
    patch = analyzer.run(music_manifest(), ws)
    acq = patch["audio"]["music"]["acquisition"]
    assert acq["status"] == "skipped"
    assert acq["file"] is None
    assert acq["links"]["soundcloud"] == "https://soundcloud.com/x/y"
    assert not called


def test_policy_licensed_only_blocks_reserved(tmp_path, monkeypatch,
                                              song_mp3):
    ws = Workspace(tmp_path / "job")
    analyzer = make_analyzer(monkeypatch,
                             cand(license="all-rights-reserved"),
                             song_mp3, policy="licensed_only")
    patch = analyzer.run(music_manifest(), ws)
    assert patch["audio"]["music"]["acquisition"]["status"] == "skipped"


def test_policy_licensed_only_allows_cc(tmp_path, monkeypatch, song_mp3):
    ws = Workspace(tmp_path / "job")
    analyzer = make_analyzer(monkeypatch, cand(license="cc-by"),
                             song_mp3, policy="licensed_only")
    patch = analyzer.run(music_manifest(), ws)
    assert patch["audio"]["music"]["acquisition"]["status"] == "acquired"


def test_no_candidates_marks_failed(tmp_path, monkeypatch):
    ws = Workspace(tmp_path / "job")
    analyzer = MusicAcquisition()
    monkeypatch.setenv("MAGICAT_ACQUISITION_POLICY", "always")
    monkeypatch.setattr(analyzer, "prober", lambda query: None)
    patch = analyzer.run(music_manifest(), ws)
    assert patch["audio"]["music"]["acquisition"]["status"] == "failed"
    assert patch["layers_status"] == {"music_acquisition": "failed"}


def test_no_music_detected_skips(tmp_path):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file="x.mp4"))
    patch = MusicAcquisition().run(m, ws)
    assert patch == {"layers_status": {"music_acquisition": "skipped"}}


def test_wrong_duration_candidate_rejected_end_to_end(tmp_path, monkeypatch,
                                                      song_mp3):
    # manifest says the song is 428s; a 120s candidate (cover/preview) must
    # be rejected by the duration gate, leaving no candidate -> failed
    ws = Workspace(tmp_path / "job")
    analyzer = make_analyzer(monkeypatch, cand(duration=120.0), song_mp3)
    patch = analyzer.run(music_manifest(), ws)
    assert patch["audio"]["music"]["acquisition"]["status"] == "failed"


def test_licensed_only_prefers_cc_candidate_later_in_chain(tmp_path,
                                                           monkeypatch,
                                                           song_mp3):
    # SoundCloud hit is all-rights-reserved but the YouTube hit is CC:
    # licensed_only must pick the CC one and download it
    ws = Workspace(tmp_path / "job")
    analyzer = MusicAcquisition()
    monkeypatch.setenv("MAGICAT_ACQUISITION_POLICY", "licensed_only")
    by_query = {
        "scsearch1:Daft Punk Around the World":
            cand(license="all-rights-reserved", source="soundcloud"),
        "ytsearch1:Daft Punk Around the World":
            cand(url="https://youtube.com/watch?v=1",
                 license="Creative Commons Attribution license (reuse allowed)",
                 source="youtube"),
    }
    monkeypatch.setattr(analyzer, "prober", lambda q: by_query.get(q))
    monkeypatch.setattr(analyzer, "downloader", lambda c, out: song_mp3)
    patch = analyzer.run(music_manifest(), ws)
    acq = patch["audio"]["music"]["acquisition"]
    assert acq["status"] == "acquired"
    assert "Creative Commons" in acq["license"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_audio_acquire.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement** — create `magicat/modules/audio/acquire.py`:

```python
# magicat/modules/audio/acquire.py
"""Acquire the identified song per policy (spec section 6.4).

Resolver chain: scsearch1 (SoundCloud) -> ytsearch1 (YouTube), two-phase:
probe with extract_info(download=False) and validate (fuzzy title, duration
tolerance, no preview-only formats), then download per policy:
  always         download whatever validated (CEO launch policy - legal
                 review flagged; one env var flips this)
  licensed_only  download only Creative-Commons-licensed candidates
  link_only      never download; persist links only
"""
from __future__ import annotations

import difflib
import logging
import subprocess
from pathlib import Path

from pydantic import BaseModel

from magicat import config
from magicat.core.registry import register_analyzer
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest

log = logging.getLogger(__name__)

TITLE_SIMILARITY_MIN = 0.6
DURATION_TOLERANCE = 0.2     # +/-20% when both durations are known


class Candidate(BaseModel):
    url: str                  # webpage_url
    title: str
    uploader: str | None = None
    duration: float | None = None
    license: str | None = None
    source: str               # "soundcloud" | "youtube"


def _ydl_opts(out_dir: Path) -> dict:
    return {
        "format": "bestaudio/best",
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),
        "noplaylist": True,
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }


def probe_query(query: str) -> Candidate | None:
    """Phase 1: resolve a search query / URL without downloading."""
    import yt_dlp
    from yt_dlp.utils import DownloadError

    source = "soundcloud" if query.startswith("scsearch") else "youtube"
    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True,
                               "noplaylist": True}) as ydl:
            info = ydl.extract_info(query, download=False)
    except DownloadError as exc:
        log.warning("probe failed for %s: %s", query, exc)
        return None
    if "entries" in info:
        if not info["entries"]:
            return None
        info = info["entries"][0]
    formats = info.get("formats") or []
    if formats and all(
            str(f.get("format_id", "")).endswith("_preview")
            for f in formats):
        return None   # SoundCloud preview-only (not fully streamable)
    return Candidate(
        url=info["webpage_url"],
        title=info.get("title", ""),
        uploader=info.get("uploader") or info.get("channel"),
        duration=info.get("duration"),
        license=info.get("license"),
        source=source,
    )


def download_candidate(candidate: Candidate, out_dir: Path) -> Path:
    """Phase 2: download + extract MP3; returns the final audio path."""
    import yt_dlp

    with yt_dlp.YoutubeDL(_ydl_opts(out_dir)) as ydl:
        info = ydl.extract_info(candidate.url, download=True)
        if "entries" in info:
            info = info["entries"][0]
        return Path(info["requested_downloads"][0]["filepath"])


def validate_candidate(candidate: Candidate, match_info: dict) -> bool:
    """Fuzzy title check + duration tolerance against the identified song."""
    got = f"{candidate.uploader or ''} {candidate.title}".lower()
    ratio = difflib.SequenceMatcher(
        None, match_info["title"].lower(), candidate.title.lower()).ratio()
    contained = match_info["title"].lower() in got
    if ratio < TITLE_SIMILARITY_MIN and not contained:
        return False
    expected = match_info.get("duration_s")
    if expected and candidate.duration:
        if abs(candidate.duration - expected) > DURATION_TOLERANCE * expected:
            return False
    return True


def is_licensed_free(candidate: Candidate) -> bool:
    lic = (candidate.license or "").lower()
    return "creative commons" in lic or lic.startswith("cc-")


def trim_audio(audio: Path, start_s: float, duration_s: float,
               dest: Path) -> Path:
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-ss", f"{start_s:.3f}", "-t", f"{duration_s:.3f}",
         "-i", str(audio), "-c:a", "libmp3lame", "-q:a", "2", str(dest)],
        check=True, capture_output=True,
    )
    return dest


@register_analyzer
class MusicAcquisition:
    name = "music_acquisition"
    layer = "music_acquisition"
    needs_gpu = False
    prober = staticmethod(probe_query)            # injectable for tests
    downloader = staticmethod(download_candidate)  # injectable for tests

    def run(self, manifest: Manifest, ws: Workspace) -> dict:
        music = manifest.audio.music
        if not music.detected:
            return {"layers_status": {"music_acquisition": "skipped"}}

        policy = config.acquisition_policy()
        match_info = {"title": music.title or "", "artist": music.artist or "",
                      "duration_s": music.duration_s}
        query_text = f"{music.artist} {music.title}"
        links = dict(music.acquisition.links)

        chosen: Candidate | None = None
        for query in (f"scsearch1:{query_text}", f"ytsearch1:{query_text}"):
            candidate = self.prober(query)
            if candidate is None:
                continue
            if not validate_candidate(candidate, match_info):
                continue
            links.setdefault(candidate.source, candidate.url)
            if chosen is None or (policy == "licensed_only"
                                  and not is_licensed_free(chosen)
                                  and is_licensed_free(candidate)):
                chosen = candidate  # prefer a CC candidate under licensed_only

        audio = manifest.audio.model_dump(mode="json")
        acq = audio["music"]["acquisition"]
        acq["links"] = links

        if chosen is None:
            acq["status"] = "failed"
            return {"audio": audio,
                    "layers_status": {"music_acquisition": "failed"}}

        acq["license"] = chosen.license
        download_allowed = policy == "always" or (
            policy == "licensed_only" and is_licensed_free(chosen))

        if not download_allowed:
            acq["status"] = "skipped"
            return {"audio": audio,
                    "layers_status": {"music_acquisition": "ok"}}

        try:
            full = self.downloader(chosen, ws.media_dir)
            seg = music.song_segment
            trimmed = trim_audio(Path(full), seg.start_in_song, seg.duration,
                                 ws.media_dir / "music.mp3")
            acq["status"] = "acquired"
            acq["file"] = str(trimmed)
            return {"audio": audio,
                    "layers_status": {"music_acquisition": "ok"}}
        except Exception:
            log.exception("download/trim failed for %s", chosen.url)
            acq["status"] = "failed"
            return {"audio": audio,
                    "layers_status": {"music_acquisition": "failed"}}
```

- [ ] **Step 4: Wire into the pipeline** — in `magicat/core/pipeline.py`:

```python
ANALYZERS = ["cut_detection", "audio_analysis", "music_acquisition"]
```

and add `import magicat.modules.audio.acquire  # noqa: F401` to `load_builtin_modules` (keep alphabetical: acquire before analyzer).

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_audio_acquire.py -v` — expect 10 PASS.
Full suite: 77 passed, 2 skipped. (Pipeline e2e: autouse env fixture -> no providers -> music skipped -> acquisition sees detected=False -> skipped. Consistent.)

- [ ] **Step 6: Commit**

```bash
git add magicat/modules/audio/acquire.py magicat/core/pipeline.py tests/test_audio_acquire.py
git commit -m "feat: policy-driven music acquisition with resolver chain"
```

---

### Task 9: Caption fixture + frame sampling

**Files:**
- Create: `magicat/modules/captions/__init__.py` (empty), `magicat/modules/captions/sampling.py`
- Modify: `tests/conftest.py` (add `caption_video` fixture)
- Test: `tests/test_caption_sampling.py`

- [ ] **Step 1: Add the caption fixture to `tests/conftest.py`**

Append (note the drawtext fontfile escaping — verified working on Windows ffmpeg 8.x: forward slashes, drive colon escaped as `\:`, value single-quoted):

```python
WINDOWS_FONT = Path("C:/Windows/Fonts/arial.ttf")


@pytest.fixture(scope="session")
def caption_video(tmp_path_factory) -> Path:
    """6s 480x854 dark clip with two burned captions at known times/positions:
    'HELLO WORLD' t=1.0-3.0 and 'SECOND LINE' t=3.5-5.2, both bottom-center.
    """
    if not WINDOWS_FONT.is_file():
        pytest.skip("test font not available")
    out = tmp_path_factory.mktemp("captions") / "captions.mp4"
    fontfile = "C\\:/Windows/Fonts/arial.ttf"
    draw1 = (f"drawtext=fontfile='{fontfile}':text='HELLO WORLD'"
             ":fontsize=42:fontcolor=white:x=(w-text_w)/2:y=h-150"
             ":enable='between(t,1,3)'")
    draw2 = (f"drawtext=fontfile='{fontfile}':text='SECOND LINE'"
             ":fontsize=42:fontcolor=white:x=(w-text_w)/2:y=h-150"
             ":enable='between(t,3.5,5.2)'")
    run_ffmpeg([
        "-f", "lavfi", "-i", "color=c=0x202020:s=480x854:r=25:d=6",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
        "-vf", f"{draw1},{draw2}",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac",
        "-shortest", str(out),
    ])
    return out
```

Also add `from pathlib import Path` to conftest imports if not already present.

- [ ] **Step 2: Write the failing tests** — create `tests/test_caption_sampling.py`:

```python
# tests/test_caption_sampling.py
from magicat.modules.captions.sampling import sample_frames


def test_samples_at_five_fps(fixture_video, tmp_path):
    samples = sample_frames(fixture_video, tmp_path / "frames")
    # 6s at 5fps -> ~30 frames (container rounding may add/drop one)
    assert 28 <= len(samples) <= 31
    assert samples[0].t == 0.0
    assert abs(samples[1].t - 0.2) < 1e-9
    for s in samples[:3]:
        assert s.path.is_file() and s.path.stat().st_size > 0


def test_sample_timestamps_monotonic(fixture_video, tmp_path):
    samples = sample_frames(fixture_video, tmp_path / "frames")
    ts = [s.t for s in samples]
    assert ts == sorted(ts)
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_caption_sampling.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 4: Implement** — create `magicat/modules/captions/sampling.py`:

```python
# magicat/modules/captions/sampling.py
"""Sample frames for OCR at a fixed rate (spec section 6.5 step 1)."""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FrameSample:
    t: float          # seconds into the video (frame n -> n / fps)
    path: Path


def sample_frames(video: Path, out_dir: Path,
                  fps: float = 5.0) -> list[FrameSample]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pattern = out_dir / "frame_%05d.jpg"
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-i", str(video), "-vf", f"fps={fps}", "-q:v", "3", str(pattern)],
        check=True, capture_output=True,
    )
    frames = sorted(out_dir.glob("frame_*.jpg"))
    # ffmpeg numbers from 1; frame N samples the source around (N-1)/fps
    return [FrameSample(t=i / fps, path=p) for i, p in enumerate(frames)]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_caption_sampling.py -v`
Expected: 2 PASS. Full suite: 79 passed, 2 skipped.

- [ ] **Step 6: Commit**

```bash
git add magicat/modules/captions tests/test_caption_sampling.py tests/conftest.py
git commit -m "feat: caption frame sampling and synthetic caption fixture"
```

---

### Task 10: OCR engine protocol + RapidOCR implementation

**Files:**
- Create: `magicat/modules/captions/ocr.py`
- Test: `tests/test_caption_ocr.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_caption_ocr.py`:

```python
# tests/test_caption_ocr.py
import numpy as np
import pytest

from magicat.modules.captions.ocr import OcrLine, RapidOcrEngine, quad_to_bbox


def test_quad_to_bbox_normalizes():
    quad = np.array([[100.0, 700.0], [380.0, 700.0],
                     [380.0, 760.0], [100.0, 760.0]])
    bbox = quad_to_bbox(quad, width=480, height=854)
    x, y, w, h = bbox
    assert abs(x - 100 / 480) < 1e-6
    assert abs(y - 700 / 854) < 1e-6
    assert abs(w - 280 / 480) < 1e-6
    assert abs(h - 60 / 854) < 1e-6


def test_ocr_line_model():
    line = OcrLine(text="HI", bbox=(0.1, 0.8, 0.5, 0.05), confidence=0.97)
    assert line.text == "HI"


def test_rapidocr_reads_caption_frame(caption_video, tmp_path):
    pytest.importorskip("rapidocr", reason="rapidocr not installed")
    from magicat.modules.captions.sampling import sample_frames
    samples = sample_frames(caption_video, tmp_path / "frames")
    # t=2.0 -> inside HELLO WORLD window
    frame = next(s for s in samples if abs(s.t - 2.0) < 1e-6)
    engine = RapidOcrEngine()
    lines = engine.read(frame.path)
    assert lines, "OCR found no text on a frame with a caption"
    joined = " ".join(l.text.upper() for l in lines)
    assert "HELLO" in joined
    assert all(0.0 <= v <= 1.0 for l in lines for v in l.bbox)
    assert lines[0].bbox[1] > 0.5  # caption sits in the lower half


def test_rapidocr_empty_frame_returns_no_lines(caption_video, tmp_path):
    pytest.importorskip("rapidocr", reason="rapidocr not installed")
    from magicat.modules.captions.sampling import sample_frames
    samples = sample_frames(caption_video, tmp_path / "frames")
    frame = next(s for s in samples if abs(s.t - 0.2) < 1e-6)  # before t=1
    assert RapidOcrEngine().read(frame.path) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_caption_ocr.py -v`
Expected: FAIL — `ModuleNotFoundError` (first two tests; the rapidocr ones reach importorskip only after the module imports).

- [ ] **Step 3: Implement** — create `magicat/modules/captions/ocr.py`:

```python
# magicat/modules/captions/ocr.py
"""OCR engine protocol + RapidOCR implementation (PP-OCRv4 via ONNX).

RapidOCR specifics (verified against rapidocr 3.8.x): the RapidOCR()
constructor downloads ~15MB of models on first ever use (cached under
site-packages afterwards) - so construction is lazy and shared. Empty
frames return None (not []) in .txts/.boxes/.scores.
"""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image
from pydantic import BaseModel


class OcrLine(BaseModel):
    text: str
    bbox: tuple[float, float, float, float]   # normalized x, y, w, h
    confidence: float


@runtime_checkable
class OcrEngine(Protocol):
    def read(self, image: Path) -> list[OcrLine]: ...


def quad_to_bbox(quad: np.ndarray, width: int,
                 height: int) -> tuple[float, float, float, float]:
    """RapidOCR box: (4,2) pixel corners TL,TR,BR,BL -> normalized xywh."""
    xs, ys = quad[:, 0], quad[:, 1]
    x0, x1 = float(xs.min()), float(xs.max())
    y0, y1 = float(ys.min()), float(ys.max())
    return (x0 / width, y0 / height, (x1 - x0) / width, (y1 - y0) / height)


class RapidOcrEngine:
    def __init__(self) -> None:
        self._engine = None   # constructed on first read (model download)

    def read(self, image: Path) -> list[OcrLine]:
        if self._engine is None:
            from rapidocr import RapidOCR
            self._engine = RapidOCR()
        result = self._engine(str(image))
        if not result.txts:          # None on empty frames
            return []
        width, height = Image.open(image).size
        lines = []
        for quad, text, score in zip(result.boxes, result.txts,
                                     result.scores):
            lines.append(OcrLine(
                text=text,
                bbox=quad_to_bbox(np.asarray(quad), width, height),
                confidence=float(score),
            ))
        return lines
```

- [ ] **Step 4: Pre-fetch the OCR models, then run tests**

First run: `.venv/Scripts/python -c "from rapidocr import RapidOCR; RapidOCR()"`
This performs the one-time ~15 MB model download from modelscope.cn during engine CONSTRUCTION (not first inference) and caches under site-packages. If it fails on a network error, retry once before concluding anything is broken.

Then: `.venv/Scripts/python -m pytest tests/test_caption_ocr.py -v`
Expected: 4 PASS.
Full suite: 83 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat/modules/captions/ocr.py tests/test_caption_ocr.py
git commit -m "feat: OCR engine protocol with RapidOCR implementation"
```

---

### Task 11: Temporal clustering + fill-color estimation (pure logic)

**Files:**
- Create: `magicat/modules/captions/clustering.py`
- Test: `tests/test_caption_clustering.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_caption_clustering.py`:

```python
# tests/test_caption_clustering.py
from magicat.modules.captions.clustering import (
    bbox_iou,
    cluster_detections,
    estimate_fill,
    text_similar,
)
from magicat.modules.captions.ocr import OcrLine


BOTTOM = (0.25, 0.8, 0.5, 0.06)


def line(text: str, bbox=BOTTOM, conf: float = 0.95) -> OcrLine:
    return OcrLine(text=text, bbox=bbox, confidence=conf)


def test_bbox_iou():
    assert bbox_iou((0, 0, 1, 1), (0, 0, 1, 1)) == 1.0
    assert bbox_iou((0, 0, 0.5, 1), (0.5, 0, 0.5, 1)) == 0.0
    assert abs(bbox_iou((0, 0, 1, 1), (0, 0, 1, 0.5)) - 0.5) < 1e-9


def test_text_similar():
    assert text_similar("HELLO WORLD", "HELL0 WORLD") is True   # OCR noise
    assert text_similar("HELLO WORLD", "SECOND LINE") is False


def test_clusters_two_sequential_captions():
    detections = []
    for i in range(5, 15):                       # t=1.0..2.8: HELLO WORLD
        detections.append((i * 0.2, [line("HELLO WORLD")]))
    for i in range(15, 17):                      # gap (no text)
        detections.append((i * 0.2, []))
    for i in range(17, 26):                      # t=3.4..5.0: SECOND LINE
        detections.append((i * 0.2, [line("SECOND LINE")]))

    segments = cluster_detections(detections, frame_interval=0.2)
    assert len(segments) == 2
    first, second = segments
    assert first["text"] == "HELLO WORLD"
    assert abs(first["t_start"] - 1.0) < 0.01
    assert abs(first["t_end"] - 3.0) < 0.01      # last frame t + interval
    assert second["text"] == "SECOND LINE"
    assert abs(second["t_start"] - 3.4) < 0.01


def test_single_frame_noise_dropped():
    detections = [(0.0, []), (0.2, [line("GLITCH")]), (0.4, []),
                  (0.6, []), (0.8, [])]
    assert cluster_detections(detections, frame_interval=0.2) == []


def test_one_frame_ocr_miss_bridged():
    detections = []
    for i in range(10):
        if i == 5:                                # OCR missed one frame
            detections.append((i * 0.2, []))
        else:
            detections.append((i * 0.2, [line("STEADY CAPTION")]))
    segments = cluster_detections(detections, frame_interval=0.2)
    assert len(segments) == 1
    assert abs(segments[0]["t_end"] - 2.0) < 0.01


def test_ocr_text_variants_majority_vote():
    detections = [(i * 0.2, [line("HELLO WORLD")]) for i in range(4)]
    detections.append((0.8, [line("HELL0 WORLD")]))   # one noisy read
    detections.append((1.0, [line("HELLO WORLD")]))
    segments = cluster_detections(detections, frame_interval=0.2)
    assert segments[0]["text"] == "HELLO WORLD"


def test_low_confidence_lines_ignored():
    detections = [(i * 0.2, [line("???", conf=0.3)]) for i in range(6)]
    assert cluster_detections(detections, frame_interval=0.2) == []


def test_estimate_fill_white_text(tmp_path):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (480, 854), (32, 32, 32))
    draw = ImageDraw.Draw(img)
    draw.rectangle([140, 690, 340, 740], fill=(250, 250, 250))
    p = tmp_path / "frame.png"
    img.save(p)
    fill = estimate_fill(p, (140 / 480, 690 / 854, 200 / 480, 50 / 854))
    r, g, b = (int(fill[i:i + 2], 16) for i in (1, 3, 5))
    assert r > 200 and g > 200 and b > 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_caption_clustering.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement** — create `magicat/modules/captions/clustering.py`:

```python
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
    luminance = pixels @ [0.299, 0.587, 0.114]
    bright = pixels[luminance >= np.percentile(luminance, 75)]
    r, g, b = (int(c) for c in np.median(bright, axis=0))
    return f"#{r:02X}{g:02X}{b:02X}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_caption_clustering.py -v`
Expected: 8 PASS. Full suite: 91 passed, 2 skipped.

- [ ] **Step 5: Commit**

```bash
git add magicat/modules/captions/clustering.py tests/test_caption_clustering.py
git commit -m "feat: caption temporal clustering and fill-color estimation"
```

---

### Task 12: CaptionAnalyzer + pipeline wiring + end-to-end + README

**Files:**
- Create: `magicat/modules/captions/analyzer.py`
- Modify: `magicat/core/pipeline.py`, `README.md`
- Test: `tests/test_caption_analyzer.py`

- [ ] **Step 1: Write the failing tests** — create `tests/test_caption_analyzer.py`:

```python
# tests/test_caption_analyzer.py
import pytest

from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import LayerState, Manifest, Source
from magicat.modules.captions.analyzer import CaptionAnalyzer
from magicat.modules.captions.ocr import OcrLine
from magicat.modules.ingest import IngestAnalyzer


class ScriptedEngine:
    """Returns captions on frames 6..15 (1-based ffmpeg numbering), which
    map to t=1.0..2.8 - like a real burn-in. Do NOT change the 6..15 range:
    frame_00006 -> enumerate index 5 -> t=1.0."""

    def read(self, image):
        n = int(image.stem.split("_")[1])      # frame_00001 -> 1
        if 6 <= n <= 15:                        # ffmpeg numbers from 1
            return [OcrLine(text="FAKE CAPTION",
                            bbox=(0.25, 0.8, 0.5, 0.06), confidence=0.95)]
        return []


def test_caption_analyzer_with_scripted_engine(fixture_video, tmp_path,
                                               monkeypatch):
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(fixture_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    analyzer = CaptionAnalyzer()
    monkeypatch.setattr(analyzer, "engine_factory", lambda: ScriptedEngine())
    patch = analyzer.run(m, ws)
    segments = patch["captions"]["segments"]
    assert len(segments) == 1
    seg = segments[0]
    assert seg["text"] == "FAKE CAPTION"
    assert abs(seg["t_start"] - 1.0) < 0.01
    assert seg["style"]["fill"].startswith("#")
    assert seg["style"]["alignment"] == "center"   # bbox is centered
    # size = bbox height * frame height px (fixture is 320x640): 0.06*640
    assert 35 <= seg["style"]["size"] <= 42
    assert patch["layers_status"] == {"captions": "ok"}
    m2 = apply_patch(m, patch)                  # validates against schema
    assert m2.captions.segments[0].text == "FAKE CAPTION"


def test_caption_analyzer_end_to_end_real_ocr(caption_video, tmp_path):
    pytest.importorskip("rapidocr", reason="rapidocr not installed")
    ws = Workspace(tmp_path / "job")
    m = Manifest(job_id="j", source=Source(file=str(caption_video)))
    m = apply_patch(m, IngestAnalyzer().run(m, ws))
    patch = CaptionAnalyzer().run(m, ws)
    segments = patch["captions"]["segments"]
    assert len(segments) == 2
    first, second = segments
    assert "HELLO" in first["text"].upper()
    assert "SECOND" in second["text"].upper()
    assert abs(first["t_start"] - 1.0) <= 0.4
    assert abs(first["t_end"] - 3.0) <= 0.4
    assert abs(second["t_start"] - 3.5) <= 0.4
    assert first["bbox"][1] > 0.5               # bottom half of the frame


def test_pipeline_runs_captions_layer(fixture_video, tmp_path, monkeypatch):
    pytest.importorskip("rapidocr", reason="rapidocr not installed")
    for var in ("AUDD_API_TOKEN", "ACR_HOST", "ACR_ACCESS_KEY",
                "ACR_ACCESS_SECRET", "MAGICAT_MUSIC_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    from magicat.core.pipeline import run_job
    manifest = run_job(str(fixture_video), tmp_path / "job")
    # color-bar fixture has no captions: layer ok, zero segments
    assert manifest.layers_status["captions"] == LayerState.OK
    assert manifest.captions.segments == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python -m pytest tests/test_caption_analyzer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement** — create `magicat/modules/captions/analyzer.py`:

```python
# magicat/modules/captions/analyzer.py
"""Caption analysis: sample frames -> OCR -> cluster -> style (spec 6.5)."""
from __future__ import annotations

from pathlib import Path

from magicat.core.registry import register_analyzer
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest
from magicat.modules.captions.clustering import cluster_detections, estimate_fill
from magicat.modules.captions.ocr import RapidOcrEngine
from magicat.modules.captions.sampling import sample_frames

SAMPLE_FPS = 5.0


@register_analyzer
class CaptionAnalyzer:
    name = "caption_analysis"
    layer = "captions"
    needs_gpu = False
    engine_factory = staticmethod(RapidOcrEngine)   # injectable for tests

    def run(self, manifest: Manifest, ws: Workspace) -> dict:
        engine = self.engine_factory()
        samples = sample_frames(Path(manifest.source.file),
                                ws.media_dir / "ocr_frames", fps=SAMPLE_FPS)
        detections = [(s.t, engine.read(s.path)) for s in samples]
        segments = cluster_detections(detections,
                                      frame_interval=1.0 / SAMPLE_FPS)

        # style (spec 6.5 step 5, the cheaply-derivable parts): fill color
        # from the segment's middle frame, size from bbox height in pixels,
        # alignment from the bbox center. Stroke/shadow are M3.
        frame_height = None
        if manifest.source.resolution:
            frame_height = int(manifest.source.resolution.split("x")[1])
        for seg in segments:
            mid_t = (seg["t_start"] + seg["t_end"]) / 2
            frame = min(samples, key=lambda s: abs(s.t - mid_t))
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

        return {
            "captions": {"segments": segments},
            "layers_status": {"captions": "ok"},
        }
```

- [ ] **Step 4: Wire into the pipeline** — in `magicat/core/pipeline.py`:

```python
ANALYZERS = ["cut_detection", "audio_analysis", "caption_analysis",
             "music_acquisition"]
```

(caption analysis before acquisition keeps all pure-analysis layers ahead of the one that talks to the outside world) and add `import magicat.modules.captions.analyzer  # noqa: F401` to `load_builtin_modules` (alphabetical).

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/Scripts/python -m pytest tests/test_caption_analyzer.py -v` — expect 3 PASS.
Full suite: 94 passed, 2 skipped.

- [ ] **Step 6: Update `README.md`**

Replace the **Status** line with:

```markdown
**Status:** M2 — manifest contract, ingest, cut detection, music
identification + acquisition, caption OCR, preview render, CLI.
```

and append after the Quickstart section:

```markdown
## Music & captions (M2)

Music identification needs a provider key (set one of):

    $env:AUDD_API_TOKEN = "..."                       # audd.io (300 free requests)
    $env:ACR_HOST = "identify-eu-west-1.acrcloud.com" # acrcloud.com console
    $env:ACR_ACCESS_KEY = "..."
    $env:ACR_ACCESS_SECRET = "..."

Without a key the music layer is skipped; captions always run.
Acquisition policy: `$env:MAGICAT_ACQUISITION_POLICY = "always" | "licensed_only" | "link_only"` (default `always`).
Optional speech/music separation for noisy voiceovers: `pip install -e .[separation]`.
```

- [ ] **Step 7: Full suite + real-CLI smoke test**

Run: `.venv/Scripts/python -m pytest -v` — expect 94 passed, 2 skipped.

Smoke (generate a clip first — no local video is guaranteed to exist):

```
ffmpeg -y -hide_banner -loglevel error -f lavfi -i color=c=red:s=320x640:r=30:d=4 -f lavfi -i sine=frequency=440:duration=4 -c:v libx264 -pix_fmt yuv420p -c:a aac -shortest smoke.mp4
.venv/Scripts/magicat run smoke.mp4 --workdir jobs/m2-smoke
```

Verify the summary lists `music` (skipped — no keys), `captions` (ok), and `music_acquisition` (skipped) layers, and `jobs/m2-smoke/manifest.json` contains a `captions` section. Delete `smoke.mp4` afterwards (jobs/ is gitignored; do not commit either).

- [ ] **Step 8: Commit**

```bash
git add magicat/modules/captions/analyzer.py magicat/core/pipeline.py README.md tests/test_caption_analyzer.py
git commit -m "feat: caption analyzer wired into pipeline; M2 README"
```

---

## Out of Scope (M2) — every deviation here is deliberate and recorded

- Font classification (M3 — spec §6.5 step 4; `style.font_family` stays empty)
- Stroke/shadow extraction (spec §6.5 step 5 remainder) — M3, alongside font work (fill/size/alignment ARE in M2, Task 12)
- Burned-in vs in-scene text discrimination beyond persistence/confidence filters (spec §6.5 step 3 full heuristics) — M3
- audio.com resolver (spec §6.4) — yt-dlp has NO audio.com extractor (verified 2026-06); the `audiocom` links key stays documented as future; revisit in M3
- Retry/backoff + circuit breaker on provider HTTP calls (spec §11) — M4 hardening; M2 degrades a failed window to no-match and falls back across providers
- Separation as a default: spec §6.3 makes Demucs step 1 mandatory; M2 ships it as the `[separation]` extra (dev machines stay light). **The M4 prod image must install `[separation]` so §6.3 holds by default — without it, fingerprinting runs on the full mix and degrades exactly on voiceover-heavy videos.**
- Music placed into the preview render (M3 — needs the single-pass concat filter per the M1 review note on AAC drift)
- FCP7 XML export, report (M3); SaaS shell (M4); CapCut + reverse search (M5)
- Real-API integration tests against AudD/ACRCloud (manual, gated on keys — run once when keys arrive)

## Self-Review + Panel-Review Notes (already applied)

- **Spec coverage:** §6.3 steps 1–3 → Tasks 2/3/4/5/6/7 (provider fallback chain implemented in Task 6 per §6.3 step 2; separation optional with prod note above); §6.4 → Task 8 (policy-swappable, links always persisted, duration-validated candidates, CC-preferring under licensed_only); §6.5 → Tasks 9/10/11/12 (fill/size/alignment styles; deferrals recorded above); §5 degradation → Task 1 + per-analyzer layers.
- **Module contract:** `config.py` is provider-agnostic and never imports from `modules/`; provider construction lives in the audio package (`providers_from_env`). No module imports a sibling module package.
- **Hermetic tests:** autouse `_isolated_magicat_env` fixture (Task 1) guarantees the suite never reads ambient API keys — no paid calls, no real downloads during pytest, regardless of the machine's environment.
- **Sequencing:** acquisition runs after audio_analysis in ANALYZERS order; it reads `manifest.audio.music` — the sequential runner guarantees the patch landed. Re-check when M4 parallelizes analyzers (acquisition must stay downstream of the join).
- **Type consistency:** `SongMatch`/`AudioWindow` defined Tasks 2/3, consumed Tasks 5/6; `Music.duration_s` added Task 1, written by `align()` (Task 5), consumed by acquisition validation (Task 8); `OcrLine` defined Task 10, consumed Tasks 11/12; analyzer `layer` attr introduced Task 1, used by every new analyzer.
- **Counts traced per task:** 36 → 41 → 46 → 51 → 58 → 65 (T6) → 67/2sk (T7) → 77 (T8) → 79 (T9) → 83 (T10) → 91 (T11) → 94/2sk (T12). Totals assume rapidocr installed with its one-time model download done (Task 10 pre-fetch step); demucs/transnet tests skip without extras.







