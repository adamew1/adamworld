from __future__ import annotations

from pathlib import Path

import numpy as np

from adamworld.splat.ply_io import read_gaussian_ply, write_gaussian_ply


def test_ply_roundtrip(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    n = 500
    means = rng.standard_normal((n, 3)).astype(np.float32)
    colors = rng.random((n, 3)).astype(np.float32)
    opacity = rng.random(n).astype(np.float32) * 0.98 + 0.01
    scales = (rng.random((n, 3)).astype(np.float32) * 0.1) + 0.01
    quats = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))

    path = tmp_path / "s.ply"
    count = write_gaussian_ply(path, means, colors, opacity, scales, quats)
    assert count == n
    assert path.exists() and path.stat().st_size > n * 17 * 4

    d = read_gaussian_ply(path)
    assert d["means"].shape == (n, 3)
    np.testing.assert_allclose(d["means"], means, atol=1e-5)
    # color/opacity/scale go through (de)quantizing encodings -> loose tolerance.
    np.testing.assert_allclose(d["colors"], colors, atol=1e-3)
    np.testing.assert_allclose(d["opacity"], opacity, atol=1e-3)
    np.testing.assert_allclose(d["scales"], scales, atol=1e-3)
