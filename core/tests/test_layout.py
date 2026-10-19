from __future__ import annotations

from pathlib import Path

import numpy as np

from adamworld.layout_raster import rasterize_layout
from adamworld.pipeline import Pipeline
from adamworld.profiling import Profiler
from adamworld.splat import GaussianCloud
from adamworld.types import GenerationRequest
from adamworld.world.schema import Layout, LayoutBlock

# A unit box scaled x4 (half-size 2), centered at the origin (column-major scale).
BIG_BOX = LayoutBlock(
    id="b0", shape="box", label="room",
    transform=[4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 1],
)  # fmt: skip


def test_rasterize_box_from_inside() -> None:
    depth, label = rasterize_layout(Layout(blocks=[BIG_BOX]), 32, 64)
    # Camera at origin is inside the box -> every ray hits a wall, labelled block 0.
    assert np.all(label == 0)
    assert depth.min() > 1.9  # nearest wall at distance 2
    assert depth.max() < 3.6  # farthest (corner-ward) within 2*sqrt(3)


def test_generate_from_layout_mock(tmp_path: Path) -> None:
    layout = Layout(blocks=[BIG_BOX], style_prompt="a stone chamber")
    world = Pipeline(profiler=Profiler()).generate_from_layout(
        layout, tmp_path / "w", backend="mock", resolution=96
    )
    assert world.splats_ply.exists() and world.splats_spz.exists()
    meta = world.meta()
    assert meta.input.kind == "layout"
    assert meta.layout is not None and len(meta.layout.blocks) == 1
    assert len(GaussianCloud.load_ply(world.splats_ply)) > 0


def test_edit_region_delete(tmp_path: Path) -> None:
    pipe = Pipeline(profiler=Profiler())
    world = pipe.generate(
        GenerationRequest(prompt="a hall", resolution=64, backend="mock"), tmp_path / "w"
    )
    before = len(GaussianCloud.load_ply(world.splats_ply))
    pipe.edit_region(world, (-40, -40, -40), (40, 40, 40), delete=True)
    after = len(GaussianCloud.load_ply(world.splats_ply))
    assert after < before
    assert world.meta().history_head == "0001"
    assert world.history()[0]["op"]["type"] == "delete_region"
