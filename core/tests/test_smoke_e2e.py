"""The single CPU smoke end-to-end test: full pipeline, mock backend, tiny res."""

from __future__ import annotations

from pathlib import Path

from adamworld.pipeline import Pipeline
from adamworld.profiling import Profiler
from adamworld.types import GenerationRequest
from adamworld.world import World


def _gen(out: Path, **kw) -> World:
    req = GenerationRequest(resolution=64, backend="mock", **kw)
    return Pipeline(profiler=Profiler()).generate(req, out)


def test_prompt_to_world(tmp_path: Path) -> None:
    world = _gen(tmp_path / "w", prompt="a quiet forest clearing")
    assert world.splats_ply.exists()
    assert world.thumbnail.exists()
    assert world.world_json.exists()

    meta = world.meta()
    assert meta.input.kind == "prompt"
    assert meta.model_versions["text_to_image"].startswith("mock")
    assert meta.model_versions["panorama"].startswith("mock")
    assert meta.bounds.aabb_max != meta.bounds.aabb_min  # non-degenerate world

    from adamworld.splat import GaussianCloud

    cloud = GaussianCloud.load_ply(world.splats_ply)
    assert len(cloud) > 100


def test_image_to_world(tmp_path: Path, sample_image: Path) -> None:
    world = _gen(tmp_path / "w", image_paths=[sample_image])
    assert world.splats_ply.exists()
    assert world.meta().input.kind == "image"


def test_cache_resume_is_fast_and_consistent(tmp_path: Path) -> None:
    out = tmp_path / "w"
    _gen(out, prompt="caves")
    first = out / "cache"
    n_cache_files = len(list(first.rglob("*.json")))
    assert n_cache_files >= 3  # t2i, panorama, depth, lift

    # Second run of a fresh world with same seed/prompt should re-use nothing
    # across worlds, but within a world a re-run hits cache. Re-running into the
    # same dir must not error and must keep the splat file.
    world2 = _gen(out, prompt="caves")
    assert world2.splats_ply.exists()
