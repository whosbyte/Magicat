# tests/test_registry.py
import pytest

from magicat.core import registry
from magicat.core.workspace import Workspace
from magicat.manifest.schema import Manifest


@pytest.fixture(autouse=True)
def clean_registry():
    # snapshot + restore: other test files rely on the built-in modules
    # registered at import time, so this file must not leak a wiped registry
    saved_analyzers = dict(registry._ANALYZERS)
    saved_exporters = dict(registry._EXPORTERS)
    registry.clear()
    yield
    registry.clear()
    registry._ANALYZERS.update(saved_analyzers)
    registry._EXPORTERS.update(saved_exporters)


def test_register_and_get_analyzer():
    @registry.register_analyzer
    class Dummy:
        name = "dummy"
        needs_gpu = False

        def run(self, manifest: Manifest, ws: Workspace) -> dict:
            return {"layers_status": {"dummy": "ok"}}

    a = registry.get_analyzer("dummy")
    assert a.name == "dummy"
    assert "dummy" in registry.list_analyzers()


def test_register_and_get_exporter():
    @registry.register_exporter
    class Dummy:
        format = "noop"

        def export(self, manifest: Manifest, ws: Workspace):
            return ws.exports_dir / "noop.txt"

    e = registry.get_exporter("noop")
    assert e.format == "noop"


def test_unknown_module_raises():
    with pytest.raises(KeyError):
        registry.get_analyzer("nope")


def test_duplicate_name_raises():
    @registry.register_analyzer
    class A:
        name = "dup"
        needs_gpu = False

        def run(self, manifest, ws):
            return {}

    with pytest.raises(ValueError):
        @registry.register_analyzer
        class B:
            name = "dup"
            needs_gpu = False

            def run(self, manifest, ws):
                return {}


def test_analyzer_layer_attribute():
    @registry.register_analyzer
    class WithLayer:
        name = "layered"
        layer = "mylayer"
        needs_gpu = False

        def run(self, manifest, ws):
            return {}

    assert registry.get_analyzer("layered").layer == "mylayer"
