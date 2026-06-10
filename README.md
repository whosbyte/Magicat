# Magicat

Deconstructs short-form videos (TikTok / Reels / Shorts) into editable
layers — scene cuts, music, captions — and rebuilds them as NLE projects.

**Status:** M2 — manifest contract, ingest, cut detection, music
identification + acquisition, caption OCR, preview render, CLI.

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
Optional speech/music separation for noisy voiceovers: `pip install -e .[separation]`.

## Architecture

Modular monolith. Every analysis layer and exporter is a plugin that
reads/writes the **Reconstruction Manifest** (`magicat/manifest/schema.py`)
and never calls sibling modules. See the design spec for the full picture.
