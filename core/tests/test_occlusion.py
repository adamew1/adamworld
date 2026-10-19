from __future__ import annotations

from pathlib import Path

import numpy as np

from adamworld import geometry
from adamworld.pipeline import Pipeline
from adamworld.profiling import Profiler
from adamworld.splat import GaussianCloud
from adamworld.types import GenerationRequest


def test_perspective_unproject_distance() -> None:
    size = 16
    depth = np.full((size, size), 2.0, np.float32)
    colors = np.zeros((size, size, 3), np.uint8)
    cam = np.array([1.0, 0.0, 0.0], np.float32)
    pts, cols = geometry.perspective_unproject(depth, colors, 90.0, cam, (0.0, 0.0))
    assert pts.shape == (size * size, 3)
    # Every unprojected point sits at radial distance 2 from the camera.
    np.testing.assert_allclose(np.linalg.norm(pts - cam, axis=1), 2.0, atol=1e-4)


def test_fit_scale_shift_recovers_affine() -> None:
    rng = np.random.default_rng(0)
    src = rng.random((32, 32)).astype(np.float32)
    ref = 3.0 * src + 1.0
    mask = np.ones_like(src, bool)
    aligned, a, b = geometry.fit_scale_shift(src, ref, mask)
    assert abs(a - 3.0) < 1e-4 and abs(b - 1.0) < 1e-4
    np.testing.assert_allclose(aligned, ref, atol=1e-4)


def test_occlusion_mock_is_noop(tmp_path: Path) -> None:
    # The mock pipeline must not invoke the GPU occlusion loop.
    world = Pipeline(profiler=Profiler()).generate(
        GenerationRequest(prompt="a cave", resolution=64, backend="mock"), tmp_path / "w"
    )
    assert "occlusion" not in world.meta().model_versions
    assert len(GaussianCloud.load_ply(world.splats_ply)) > 0
