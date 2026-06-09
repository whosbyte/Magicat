# magicat/core/pipeline.py
"""M1 sequential runner. Celery replaces this in M4; module contracts stay.

Failure policy (spec section 5): ingest failure is fatal; any other layer
failure marks that layer FAILED and the job continues.
"""
from __future__ import annotations

import logging
import uuid
from pathlib import Path

from magicat.core import registry
from magicat.core.workspace import Workspace
from magicat.manifest.patch import apply_patch
from magicat.manifest.schema import Manifest, Source

log = logging.getLogger(__name__)

# layer name written to layers_status when an analyzer crashes before
# returning its patch
LAYER_OF_ANALYZER = {"cut_detection": "shots"}

ANALYZERS = ["cut_detection"]          # M2+: audio_analysis, caption_analysis
EXPORTERS = ["preview_mp4"]            # M3+: premiere_resolve_zip


def load_builtin_modules() -> None:
    """Import modules for their @register side effects."""
    import magicat.modules.cuts_pyscenedetect  # noqa: F401
    import magicat.modules.cuts_transnetv2  # noqa: F401
    import magicat.modules.ingest  # noqa: F401
    import magicat.modules.render_preview  # noqa: F401


def run_job(input_arg: str, workdir: Path) -> Manifest:
    load_builtin_modules()
    ws = Workspace(workdir)

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
            layer = LAYER_OF_ANALYZER.get(name, name)
            manifest = apply_patch(
                manifest, {"layers_status": {layer: "failed"}})

    for fmt in EXPORTERS:
        exporter = registry.get_exporter(fmt)
        try:
            artifact = exporter.export(manifest, ws)
            manifest = apply_patch(manifest, {"exports": [
                *[e.model_dump() for e in manifest.exports],
                {"format": fmt, "artifact": str(artifact)},
            ]})
        except Exception:
            log.exception("exporter %s failed", fmt)

    ws.save_manifest(manifest)
    return manifest
