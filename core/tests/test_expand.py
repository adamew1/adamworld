from __future__ import annotations

from pathlib import Path

from adamworld.pipeline import Pipeline, _direction_vector
from adamworld.profiling import Profiler
from adamworld.splat import GaussianCloud
from adamworld.types import GenerationRequest


def test_direction_vectors() -> None:
    assert tuple(_direction_vector("north")) == (0.0, 0.0, -1.0)
    assert tuple(_direction_vector("east")) == (1.0, 0.0, 0.0)
    v = _direction_vector("1,0,0")
    assert abs(float(v[0]) - 1.0) < 1e-6


def test_expand_grows_world(tmp_path: Path) -> None:
    out = tmp_path / "w"
    pipe = Pipeline(profiler=Profiler())
    world = pipe.generate(GenerationRequest(prompt="a glade", resolution=64, backend="mock"), out)
    before = len(GaussianCloud.load_ply(world.splats_ply))
    assert before > 0
    assert len(world.meta().tiles) == 0

    pipe.expand(world, direction="north", prompt="a cave mouth", distance=10.0)
    after = len(GaussianCloud.load_ply(world.splats_ply))
    meta = world.meta()

    assert after > before  # the world grew
    assert len(meta.tiles) == 1
    assert meta.tiles[0].source == "expand"
    assert meta.tiles[0].splat_range == (before, after)
    # A versioned history entry was recorded.
    assert meta.history_head == "0001"
    assert world.splats_spz.exists()
    # Bounds extend northwards (−z) past the original region.
    assert meta.bounds.aabb_min[2] < -5.0
