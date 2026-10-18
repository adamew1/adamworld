from __future__ import annotations

from pathlib import Path

from adamworld.world import World
from adamworld.world.schema import FORMAT_VERSION, InputSpec


def test_world_create_load_meta(tmp_path: Path) -> None:
    w = World.create(
        tmp_path / "w",
        input=InputSpec(kind="prompt", prompt="hello"),
        seed=42,
        name="demo",
    )
    assert w.world_json.exists()
    meta = w.meta()
    assert meta.format_version == FORMAT_VERSION
    assert meta.seed == 42
    assert meta.input.prompt == "hello"
    assert meta.history_head == "0000"

    reloaded = World.load(tmp_path / "w")
    assert reloaded.meta().id == meta.id


def test_history_versioning(tmp_path: Path) -> None:
    w = World.create(tmp_path / "w", input=InputSpec(kind="prompt", prompt="x"))
    h1 = w.add_edit({"type": "noop", "n": 1})
    h2 = w.add_edit({"type": "noop", "n": 2})
    assert h1 == "0001"
    assert h2 == "0002"
    assert w.meta().history_head == "0002"
    assert len(w.history()) == 2
