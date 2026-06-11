# Magicat — Short-Form Video Deconstruction Framework

**Design spec — 2026-06-09**
**Status:** Approved (Approach A: modular monolith with split worker pools)

## 1. Purpose

Magicat is a web SaaS that takes a short-form video (TikTok / Instagram Reels / YouTube Shorts URL, or direct upload) and deconstructs it into the layers used to make it — scene cuts, source footage references, background music, and captions — then delivers an editable, reconstructed project for professional NLEs plus a rendered preview.

**Core value proposition (v1):** editable project recreation — an editor opens a finished short as a layered Premiere/Resolve project with cuts, music, and captions already placed. Reverse video search (source-footage discovery) is a v2 value-add layer.

**User-facing deliverables per job:**

- Summary report: source-footage links (v2), detected song (title/artist + provider links), detected caption font(s)
- Adobe Premiere project (zip: FCP7 XML + media + SRT + style sheet)
- DaVinci Resolve project (same FCP7 XML zip; Resolve imports it natively)
- CapCut project (v1.5, feature-flagged)
- Preview video (browser playback + MP4 download)

## 2. Key Decisions (with rationale)

| Decision | Choice | Rationale |
|---|---|---|
| Form factor | Web SaaS | Users paste a URL; processing server-side; easiest to monetize |
| Core value | Editable project recreation | Buildable with known tech; reverse search is highest-risk and becomes additive |
| Export order | Premiere first via FCP7 XML | One stable, documented format covers both Premiere and Resolve; CapCut's format is reverse-engineered and fragile → v1.5, flagged |
| Music acquisition | `always` download at launch | CEO decision; **flagged for legal review pre-launch**. Acquisition module is policy-swappable via one config value (`always \| licensed_only \| link_only`) |
| Build vs. buy | Off-the-shelf models/APIs first | TransNetV2, Demucs, PaddleOCR, AudD/ACRCloud are proven; the only custom-trained model is the caption-font classifier (no off-the-shelf equivalent for the short-form font universe) |
| Architecture | Modular monolith, split GPU/CPU worker pools | One codebase/deploy, modules swappable behind the Manifest contract; GPU and CPU capacity scale independently; modules can be lifted to serverless GPU (Modal/RunPod functions) later without touching other modules |

## 3. High-Level Architecture

```
┌─────────────┐     ┌──────────────┐     ┌───────────────────────────────┐
│  Web App     │────▶│  API Service │────▶│  Job Orchestrator (Celery)    │
│  (Next.js)   │ SSE │  (FastAPI)   │     │  DAG per job, Redis broker    │
└─────────────┘     └──────┬───────┘     └──────┬────────────────┬───────┘
                           │                    │                │
                    ┌──────▼──────┐      ┌──────▼─────┐   ┌──────▼─────┐
                    │  Postgres    │      │ GPU workers│   │ CPU/IO     │
                    │ jobs, users, │      │ cuts, font,│   │ workers    │
                    │ manifests    │      │ OCR, demucs│   │ ingest,    │
                    └─────────────┘      └────────────┘   │ APIs,      │
                    ┌─────────────┐                       │ export,    │
                    │  S3 / R2     │◀──────all media──────│ render     │
                    └─────────────┘                       └────────────┘
```

One Python codebase (`magicat/`), one Docker image, three process types: **API**, **GPU worker**, **CPU worker**. Every analysis layer and exporter is a plugin module in a module registry. Modules never call each other — they only read and write the **Reconstruction Manifest**.

## 4. The Reconstruction Manifest — core contract

A versioned JSON document (Pydantic schema) that is the single source of truth for a job. Every module receives the manifest + a media workspace and returns a **patch** to its own section; the orchestrator merges patches. Modules are therefore order-independent within their stage and individually replaceable.

```jsonc
{
  "manifest_version": "1.0",
  "job_id": "…",
  "source": { "url": "", "platform": "", "file": "", "fps": 0, "resolution": "", "duration": 0 },

  "shots": [ { "id": "", "start": 0, "end": 0, "keyframes": ["s3://…"], "confidence": 0 } ],

  "source_matches": [            // reverse video search — v2, empty in v1
    { "shot_id": "", "candidates": [ { "url": "", "title": "", "thumbnail": "", "score": 0 } ] }
  ],

  "audio": {
    "speech_stem": "s3://…",     // separated voice/sfx from original audio
    "music": {
      "detected": true,
      "title": "", "artist": "",
      "provider_ids": { "audd": "", "acrcloud": "", "isrc": "" },
      "song_segment": { "start_in_song": 0, "duration": 0 },   // which part of the song is used
      "timeline_offset": 0.0,                                  // where it sits in the video
      "acquisition": {
        "status": "acquired | failed | skipped",
        "file": "s3://…", "license": "",
        "links": { "soundcloud": "", "audiocom": "", "spotify": "" }
      }
    }
  },

  "captions": {
    "segments": [ {
      "text": "", "t_start": 0, "t_end": 0,
      "bbox": [0, 0, 0, 0],      // normalized 0–1: x, y, w, h
      "style": {
        "font_family": "",
        "font_candidates": [ { "name": "", "confidence": 0 } ],
        "size": 0, "fill": "", "stroke": "", "shadow": "", "alignment": ""
      }
    } ]
  },

  "layers_status": { "shots": "ok", "music": "ok", "captions": "ok", "source_matches": "skipped" },
  "exports": [ { "format": "", "artifact": "s3://…" } ],
  "report": { /* user-facing summary, §9 */ }
}
```

**Module interfaces** — the whole plugin system is two protocols:

```python
class Analyzer(Protocol):
    name: str
    needs_gpu: bool
    def run(self, manifest: Manifest, ws: Workspace) -> ManifestPatch: ...

class Exporter(Protocol):
    format: str
    def export(self, manifest: Manifest, ws: Workspace) -> Artifact: ...
```

## 5. Pipeline DAG

```
ingest
  ├──▶ cut_detection        (GPU)   ─┐
  ├──▶ audio_analysis       (GPU→IO)─┤  run in PARALLEL
  ├──▶ caption_analysis     (GPU)   ─┤
  └──▶ reverse_search [v2, flagged] ─┘
            │ join (merge patches)
            ▼
        assemble  (manifest → OpenTimelineIO timeline)
            ▼
        acquire_music  (policy-driven download + trim)
            ├──▶ export_premiere_resolve  (FCP7 XML + media + SRT, zipped)
            ├──▶ export_capcut [v1.5, flagged]
            └──▶ render_preview  (ffmpeg MP4)
            ▼
        report → notify user
```

A failed analyzer does **not** kill the job: it marks its layer `failed` in `layers_status` and downstream stages work with what exists (no music detected → timeline ships without a music track; the report says so). Only `ingest` failure is fatal.

## 6. Analysis Modules

### 6.1 Ingest (CPU)
`yt-dlp` fetches from TikTok / Reels / Shorts URLs; direct file upload also supported. Normalize to H.264 MP4, extract WAV, `ffprobe` fills `source` metadata.

### 6.2 Cut detection (GPU)
**TransNetV2** (state-of-the-art open shot-boundary model) with PySceneDetect's content detector as a cross-check; disagreements resolved by confidence threshold. Extracts 3 keyframes per shot (start/middle/end) for the UI and for reverse search.

### 6.3 Audio analysis (GPU → IO)
1. **Demucs (htdemucs)** separates music from speech — voiceover on top of music breaks fingerprinting.
2. Fingerprint the music stem in overlapping 12 s windows: **AudD** primary (returns in-song offsets), **ACRCloud** fallback.
3. Align window offsets to compute `song_segment` (which part of the track is used) and `timeline_offset` (where it starts in the video).

### 6.4 Music acquisition (IO, policy-swappable)
Config `acquisition_policy: always | licensed_only | link_only`. Launch policy: **`always`** — resolver chain tries SoundCloud (`scdl`), audio.com, then yt-dlp fallback; trims the file to `song_segment` with ffmpeg. Whatever the policy, the manifest always carries the provider **links**, so the report and a correctly-timed placeholder slot work even when download fails or policy forbids it. **Open item: legal review of the `always` policy before launch; switching policy is a one-line config change.**

### 6.5 Caption analysis (GPU)
1. Sample frames at 5 fps; **PaddleOCR** detects + recognizes text.
2. Cluster detections across time (text similarity + bbox IoU) into caption segments with start/end times.
3. Separate burned-in captions from in-scene text (heuristics: positional stability, recurrence, frame region).
4. **Font classifier** — the one custom-trained model: a lightweight image classifier over caption crops, trained on **synthetic renders** of the ~80-font universe used in short-form (CapCut/TikTok built-ins: Montserrat, Proxima Nova, TheBoldFont, Komika, Bangers, …) with augmentation (stroke, shadow, compression artifacts). Outputs top-3 candidates with confidence; the report shows all three.
5. Style extraction: fill/stroke color, shadow, size, alignment.

### 6.6 Reverse video search (v2, feature-flagged)
Keyframes → provider chain → candidate URLs per shot → ranked + domain-deduped. *(As-built note, M5 2026-06-11: Bing Visual Search was retired by Microsoft in Aug 2025 — the shipped providers are SerpAPI Google Lens (needs public keyframe hosting) and Google Cloud Vision WEB_DETECTION (local bytes); CLIP re-ranking is deferred to a post-v1 optional extra — see the M5 plan.)* The manifest slot (`source_matches`) and UI panel exist from day one so this drops in without schema changes.

## 7. Assembly & Exporters

**Assembly** builds an **OpenTimelineIO** timeline from the manifest: video track (shots as subclips of the source video), music track (acquired file placed at `timeline_offset`), caption track. OTIO provides the FCP7 XML adapter.

| Target | Mechanism | Ships |
|---|---|---|
| Premiere | FCP7 XML via OTIO + media folder + SRT sidecar + style sheet — zipped | v1 |
| Resolve | Same FCP7 XML zip (native import); Resolve-specific `.drp` later if demand | v1 |
| CapCut | Generated draft folder (`draft_content.json`) — reverse-engineered format; feature-flagged, version-pinned, with a format test suite so breakage is detected by CI, not users | v1.5 |
| Preview MP4 | ffmpeg: cut shots + music mixed at offset + captions burned via ASS/libass in the matched font | v1 |

**Caption fidelity:** FCP7 XML titles are limited, so captions travel three ways — basic titles in the timeline, an SRT sidecar, and exact style specs (font, colors, position) in the report. Editors get timing/position in the NLE and one-click styling info.

## 8. Web SaaS Shell

- **Frontend:** Next.js + Tailwind. Flow: paste URL → job page with live stage-by-stage progress (SSE) → results page.
- **Results page:** preview player; summary card (per-shot source links when available, song title/artist + provider links, font name + confidence); download buttons (Premiere zip / Resolve zip / CapCut zip / MP4).
- **API:** FastAPI — auth (Clerk), `POST /jobs`, `GET /jobs/{id}` + SSE progress stream, presigned S3 download URLs.
- **Billing (post-launch):** Stripe, credit-per-job; the `users` schema includes a `credits` column from day one so billing is additive.

## 9. Report (user-facing summary)

Generated last, stored in the manifest and rendered in the UI and as `report.html` inside every export zip:

- Per-shot thumbnails + timing (+ source-footage links in v2)
- Song: title, artist, links (SoundCloud / audio.com / Spotify), which segment of the song is used
- Captions: font name (top-3 with confidence), style swatch, full transcript
- Layer status: what was recovered and what failed/skipped

## 10. Infrastructure & Ops

- **Dev:** single `docker compose up` — API, both worker types, Postgres, Redis, MinIO. GPU optional locally (GPU modules degrade to CPU, slower).
- **Prod:** API + web on a PaaS (Fly/Render); GPU workers on RunPod/EC2 g5 with queue-depth autoscaling; Postgres (Neon/RDS); Redis; Cloudflare R2 for media (zero egress fees on big deliverable zips).
- **Media retention:** source videos and acquired audio auto-deleted after 7 days (cost + copyright hygiene); manifests kept indefinitely.
- **Observability:** Sentry, structured logs, per-stage timings stored on the job row.

## 11. Error Handling

- **Graceful degradation per layer** (§5): each layer independently `ok | failed | skipped`; the report states exactly what was and wasn't recovered.
- **External APIs** (AudD, ACRCloud, SerpAPI): retry with exponential backoff, provider fallback chain, circuit breaker.
- **Idempotency:** tasks keyed by `(job_id, stage)`; artifacts content-addressed in S3 — a retried job reuses finished stages.
- **Poison-pill protection:** per-task time and memory limits so one pathological video cannot wedge a worker.

## 12. Testing

- **Golden fixtures:** curated short videos with hand-labeled ground truth (cut timestamps, song + offset, caption text/font). Tolerances: cuts ±2 frames, song offset ±0.5 s, font present in top-3.
- **Manifest schema tests:** every module's patch validates against the Pydantic schema; manifest version migrations tested.
- **Exporter validation:** generated FCP7 XML round-trips through OTIO parse + XSD validation; CapCut drafts checked against pinned format fixtures. Manual QA checklist: import into real Premiere/Resolve before each release.
- **E2E smoke:** docker-compose pipeline run on a 15 s fixture asserting a complete manifest + all artifacts.

## 13. Build Order

| Milestone | Scope | Proves |
|---|---|---|
| **M1** | Skeleton: manifest schema, module registry, ingest, cut detection, preview render — CLI-driven | The contract works end-to-end |
| **M2** | Audio: Demucs + fingerprinting + acquisition; caption OCR + timing | The two hard analyzers |
| **M3** | Font classifier (synthetic training set) + FCP7 XML export + report | The full v1 deliverable |
| **M4** | SaaS shell: web app, queue, progress, auth, downloads | **Launchable v1** |
| **M5** | CapCut exporter + reverse video search (flagged) | The complete vision |

Each milestone gets its own implementation plan; M1 is planned first.

## 14. Open Items

- **Legal review of `always` music-download policy** before launch (decision recorded in §2; mitigation: policy-swappable config).
- CapCut draft format version pinning strategy — decided during M5 planning.
- Reverse-search provider costs (SerpAPI/Bing pricing at scale) — evaluated during M5 planning.
