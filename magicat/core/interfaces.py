# magicat/core/interfaces.py
"""The two protocols every plugin implements (spec section 4)."""
from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from magicat.core.workspace import Workspace
from magicat.manifest.patch import ManifestPatch
from magicat.manifest.schema import Manifest

__all__ = ["Analyzer", "Exporter", "ManifestPatch"]


@runtime_checkable
class Analyzer(Protocol):
    name: str
    layer: str  # layers_status key this analyzer owns (marked failed on crash)
    needs_gpu: bool

    def run(self, manifest: Manifest, ws: Workspace) -> ManifestPatch: ...


@runtime_checkable
class Exporter(Protocol):
    format: str

    def export(self, manifest: Manifest, ws: Workspace) -> Path: ...
