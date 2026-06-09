# magicat/manifest/patch.py
"""Apply a module's patch to the manifest.

A patch is a plain dict keyed by top-level manifest sections. Sections are
replaced wholesale, except `layers_status`, which is merged so concurrent
modules never clobber each other's status. The result is fully re-validated.
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
        else:
            data[key] = value
    return Manifest.model_validate(data)
