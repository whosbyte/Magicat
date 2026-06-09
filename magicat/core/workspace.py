# magicat/core/workspace.py
"""Per-job working directory: media, keyframes, exports, manifest.json."""
from __future__ import annotations

from pathlib import Path

from magicat.manifest.schema import Manifest


class Workspace:
    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        for d in (self.media_dir, self.keyframes_dir, self.exports_dir):
            d.mkdir(parents=True, exist_ok=True)

    @property
    def media_dir(self) -> Path:
        return self.root / "media"

    @property
    def keyframes_dir(self) -> Path:
        return self.root / "keyframes"

    @property
    def exports_dir(self) -> Path:
        return self.root / "exports"

    @property
    def manifest_path(self) -> Path:
        return self.root / "manifest.json"

    def save_manifest(self, manifest: Manifest) -> None:
        self.manifest_path.write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )

    def load_manifest(self) -> Manifest:
        return Manifest.model_validate_json(
            self.manifest_path.read_text(encoding="utf-8")
        )
