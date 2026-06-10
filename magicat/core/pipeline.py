# magicat/core/pipeline.py
"""M1 sequential runner. Celery replaces this in M4; module contracts stay.

Failure policy (spec section 5): ingest failure is fatal; any other layer
failure marks that layer FAILED and the job continues.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Callable

from magicat.core import registry
from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import Manifest, Source
from magicat.modules.report import build_report

log = logging.getLogger(__name__)

ANALYZERS = ["cut_detection", "audio_analysis", "caption_analysis",
             "reverse_search", "music_acquisition"]
EXPORTERS = ["preview_mp4", "report_html", "premiere_resolve_zip"]

ProgressFn = Callable[[str, str], None]


def _noop_progress(stage: str, state: str) -> None:
    return None


def load_builtin_modules() -> None:
    """Import modules for their @register side effects."""
    import magicat.modules.audio.acquire  # noqa: F401
    import magicat.modules.audio.analyzer  # noqa: F401
    import magicat.modules.captions.analyzer  # noqa: F401
    import magicat.modules.cuts_pyscenedetect  # noqa: F401
    import magicat.modules.cuts_transnetv2  # noqa: F401
    import magicat.modules.export.package  # noqa: F401
    import magicat.modules.ingest  # noqa: F401
    import magicat.modules.render_preview  # noqa: F401
    import magicat.modules.report  # noqa: F401
    import magicat.modules.sources.analyzer  # noqa: F401


def run_job(input_arg: str, workdir: Path, job_id: str | None = None,
            on_progress: ProgressFn | None = None) -> Manifest:
    raw_progress = on_progress or _noop_progress

    def progress(stage: str, state: str) -> None:
        try:
            raw_progress(stage, state)
        except Exception:                  # telemetry must never kill the job
            log.exception("progress callback failed at %s/%s", stage, state)

    load_builtin_modules()
    # resolve so every path persisted into the manifest is absolute - the
    # manifest outlives the process and may be loaded from a different cwd
    ws = Workspace(Path(workdir).resolve())

    if input_arg.startswith(("http://", "https://")):
        source = Source(url=input_arg)
    else:
        source = Source(file=str(Path(input_arg).resolve()))
    manifest = Manifest(job_id=job_id or uuid.uuid4().hex, source=source)

    # ingest is fatal on failure
    progress("ingest", "start")
    manifest = apply_patch(manifest, registry.get_analyzer("ingest")
                           .run(manifest, ws))
    progress("ingest", "ok")

    for name in ANALYZERS:
        analyzer = registry.get_analyzer(name)
        progress(name, "start")
        try:
            manifest = apply_patch(manifest, analyzer.run(manifest, ws))
            state = manifest.layers_status.get(analyzer.layer)
            progress(name, state.value if state else "ok")
        except Exception:
            log.exception("analyzer %s failed", name)
            manifest = apply_patch(
                manifest, {"layers_status": {analyzer.layer: "failed"}})
            progress(name, "failed")

    manifest = apply_patch(manifest, {"report": build_report(manifest)})

    for fmt in EXPORTERS:
        exporter = registry.get_exporter(fmt)
        progress(fmt, "start")
        try:
            artifact = exporter.export(manifest, ws)
            manifest = apply_patch(manifest, {
                "exports": [{"format": fmt, "artifact": str(artifact)}],
                "layers_status": {fmt: "ok"},
            })
            progress(fmt, "ok")
        except Exception:
            log.exception("exporter %s failed", fmt)
            manifest = apply_patch(
                manifest, {"layers_status": {fmt: "failed"}})
            progress(fmt, "failed")

    manifest = apply_patch(manifest, {"report": build_report(manifest)})

    ws.save_manifest(manifest)
    progress("job", "done")
    return manifest
