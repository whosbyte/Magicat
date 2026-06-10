# Magicat

Deconstructs short-form videos (TikTok / Reels / Shorts) into editable
layers — scene cuts, music, captions — and rebuilds them as NLE projects.

**Status:** M4 — launchable v1: web service with live progress, plus the
full deconstruction pipeline (cuts, music, captions + fonts, NLE export).

Docs: [design spec](docs/superpowers/specs/2026-06-09-magicat-framework-design.md)
· [M1 plan](docs/superpowers/plans/2026-06-09-m1-skeleton.md)

## Quickstart

Prereqs: Python 3.11+, ffmpeg + ffprobe on PATH.

    python -m venv .venv
    .venv/Scripts/python -m pip install -e .[dev]
    .venv/Scripts/python -m pytest

Run on a local file or a video URL:

    .venv/Scripts/magicat run path/to/video.mp4 --workdir jobs/demo
    .venv/Scripts/magicat run https://www.tiktok.com/@user/video/123

Outputs land in the workdir: `manifest.json`, `keyframes/`,
`exports/preview.mp4`.

## Music & captions (M2)

Music identification needs a provider key (set one of):

    $env:AUDD_API_TOKEN = "..."                       # audd.io (300 free requests)
    $env:ACR_HOST = "identify-eu-west-1.acrcloud.com" # acrcloud.com console
    $env:ACR_ACCESS_KEY = "..."
    $env:ACR_ACCESS_SECRET = "..."

Without a key the music layer is skipped; captions always run.
Acquisition policy: `$env:MAGICAT_ACQUISITION_POLICY = "always" | "licensed_only" | "link_only"` (default `always`).
Note: link_only still performs network probes (to collect links); it only skips the download itself.
Optional speech/music separation for noisy voiceovers: `pip install -e .[separation]`.

## Project export (M3)

Every job now produces `exports/premiere_resolve.zip`: an FCP7-XML project
(imports into Adobe Premiere AND DaVinci Resolve), `captions.srt`,
`report.html`, import instructions, and the referenced media. Captions
travel as SRT (xmeml titles are not portable); font + style specs are in
the report. Bundled OFL fonts under `assets/fonts/` extend the font
matcher; add your own via `$env:MAGICAT_FONT_DIRS`.

## Architecture

Modular monolith. Every analysis layer and exporter is a plugin that
reads/writes the **Reconstruction Manifest** (`magicat/manifest/schema.py`)
and never calls sibling modules. See the design spec for the full picture.

## Web service (M4)

    .venv/Scripts/magicat serve --port 8123

Open http://127.0.0.1:8123/ - paste a short-form URL or upload a clip,
watch per-layer progress live, then preview and download the deliverables.
JSON API: POST /api/jobs, GET /api/jobs/{id}, GET /api/jobs/{id}/events
(SSE), GET /api/jobs/{id}/artifacts/{name}. Optional auth: set
$env:MAGICAT_API_KEY and send X-API-Key. Local-first stack (SQLite +
thread pool + static UI) behind the same seams the cloud deployment
(Celery/Postgres/S3/Next.js) slots into - see the M4 plan's deviation
table.
