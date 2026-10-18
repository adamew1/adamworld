from __future__ import annotations

import numpy as np

from adamworld import geometry


def test_uv_dir_roundtrip() -> None:
    u = np.linspace(0.1, 0.9, 9)
    v = np.linspace(0.1, 0.9, 9)
    uu, vv = np.meshgrid(u, v)
    dirs = geometry.equirect_uv_to_dirs(uu, vv)
    u2, v2 = geometry.dirs_to_equirect_uv(dirs)
    np.testing.assert_allclose(u2, uu, atol=1e-5)
    np.testing.assert_allclose(v2, vv, atol=1e-5)
    # Directions are unit length.
    np.testing.assert_allclose(np.linalg.norm(dirs, axis=-1), 1.0, atol=1e-5)


def test_sample_equirect_constant_and_wrap() -> None:
    img = np.full((16, 32, 3), 123, np.uint8)
    u = np.array([0.0, 0.5, 0.999, 1.0 - 1e-9])
    v = np.array([0.1, 0.5, 0.9, 0.5])
    out = geometry.sample_equirect(img, u, v)
    np.testing.assert_allclose(out, 123.0, atol=1e-4)


def test_equirect_to_perspective_shape_and_front() -> None:
    img = np.zeros((32, 64, 3), np.uint8)
    img[16, 32] = [255, 0, 0]  # mark near the front/center (+z)
    patch = geometry.equirect_to_perspective(img, 0.0, 0.0, 90.0, 24)
    assert patch.shape == (24, 24, 3)
    # Center of the front perspective view looks at the marked equirect center.
    assert patch[12, 12, 0] > patch[0, 0, 0]


def test_place_image_front_mask() -> None:
    seed = np.zeros((40, 40, 3), np.uint8)
    seed[:] = [10, 200, 30]
    canvas, mask = geometry.place_image_front(seed, 64, 128, fov_deg=90.0)
    assert canvas.shape == (64, 128, 3)
    # Front (+z) center is known and carries the seed color.
    assert mask[32, 64] == 1.0
    np.testing.assert_array_equal(canvas[32, 64], [10, 200, 30])
    # The back (-z, u=0) is unknown -> must be outpainted.
    assert mask[32, 0] == 0.0


def test_ring_fusion_covers_horizon() -> None:
    h, w = 48, 96
    accum = np.zeros((h, w), np.float32)
    weight = np.zeros((h, w), np.float32)
    patch = np.ones((32, 32), np.float32)
    for lon0, lat0 in geometry.ring_centers(8):
        geometry.perspective_to_equirect_accumulate(patch, lon0, lat0, 90.0, h, w, accum, weight)
    # The horizon band is fully covered by the 8-view ring.
    band = weight[int(h * 0.45) : int(h * 0.55)]
    assert np.all(band > 0)


def test_horizontal_wrap_pad() -> None:
    img = np.arange(3 * 6).reshape(3, 6, 1).astype(np.uint8)
    padded = geometry.horizontal_wrap_pad(img, 2)
    assert padded.shape == (3, 10, 1)
    np.testing.assert_array_equal(padded[:, :2, 0], img[:, -2:, 0])
    np.testing.assert_array_equal(padded[:, -2:, 0], img[:, :2, 0])
