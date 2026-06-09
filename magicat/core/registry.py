# magicat/core/registry.py
"""Plugin registry. Modules self-register at import time via decorators."""
from __future__ import annotations

from magicat.core.interfaces import Analyzer, Exporter

_ANALYZERS: dict[str, Analyzer] = {}
_EXPORTERS: dict[str, Exporter] = {}


def register_analyzer(cls: type) -> type:
    instance = cls()
    if instance.name in _ANALYZERS:
        raise ValueError(f"analyzer {instance.name!r} already registered")
    _ANALYZERS[instance.name] = instance
    return cls


def register_exporter(cls: type) -> type:
    instance = cls()
    if instance.format in _EXPORTERS:
        raise ValueError(f"exporter {instance.format!r} already registered")
    _EXPORTERS[instance.format] = instance
    return cls


def get_analyzer(name: str) -> Analyzer:
    return _ANALYZERS[name]


def get_exporter(fmt: str) -> Exporter:
    return _EXPORTERS[fmt]


def list_analyzers() -> list[str]:
    return sorted(_ANALYZERS)


def list_exporters() -> list[str]:
    return sorted(_EXPORTERS)


def clear() -> None:
    """Test helper — wipe all registrations."""
    _ANALYZERS.clear()
    _EXPORTERS.clear()
