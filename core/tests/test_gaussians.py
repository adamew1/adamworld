from __future__ import annotations

import numpy as np

from adamworld.splat.gaussians import GaussianCloud


def test_from_rgbd_equirect_shapes_and_count() -> None:
    h, w = 64, 128
    rgb = np.zeros((h, w, 3), np.uint8)
    depth = np.full((h, w), 2.0, np.float32)
    cloud = GaussianCloud.from_rgbd_equirect(rgb, depth, max_points=2000)
    assert len(cloud) > 0
    assert cloud.means.shape[1] == 3
    assert cloud.colors.shape == cloud.means.shape
    assert cloud.quats.shape[1] == 4
    # Constant radius-2 sphere: every point ~2 units from origin.
    r = np.linalg.norm(cloud.means, axis=1)
    np.testing.assert_allclose(r, 2.0, atol=1e-3)


def test_concat_and_aabb() -> None:
    h, w = 16, 32
    depth = np.ones((h, w), np.float32)
    a = GaussianCloud.from_rgbd_equirect(np.zeros((h, w, 3), np.uint8), depth)
    b = GaussianCloud.from_rgbd_equirect(np.zeros((h, w, 3), np.uint8), depth * 3)
    merged = GaussianCloud.concat([a, b])
    assert len(merged) == len(a) + len(b)
    lo, hi = merged.aabb()
    assert all(h_ >= l_ for l_, h_ in zip(lo, hi, strict=False))


def test_invalid_depth_dropped() -> None:
    h, w = 16, 32
    depth = np.zeros((h, w), np.float32)  # all non-positive -> dropped
    cloud = GaussianCloud.from_rgbd_equirect(np.zeros((h, w, 3), np.uint8), depth)
    assert len(cloud) == 0
