# magicat/manifest/patch.py
"""Apply a module's patch to the manifest.

A patch is a plain dict keyed by top-level manifest sections. Sections are
replaced wholesale, except `layers_status`, which is merged, and `exports`,
which appends. The result is fully re-validated.
"""
from __future__ import annotations

from typing import Any

from magicat.manifest.schema import Manifest

ManifestPatch = dict[str, Any]


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
