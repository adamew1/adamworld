from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

from adamworld.splat.spz_io import FRACTIONAL_BITS, NGSP_MAGIC, VERSION, write_spz


def test_spz_header_and_layout(tmp_path: Path) -> None:
    rng = np.random.default_rng(3)
    n = 128
    means = rng.standard_normal((n, 3)).astype(np.float32)
    colors = rng.random((n, 3)).astype(np.float32)
    opacity = rng.random(n).astype(np.float32)
    scales = (rng.random((n, 3)).astype(np.float32) * 0.1) + 0.01
    quats = np.tile(np.array([1, 0, 0, 0], np.float32), (n, 1))

    path = tmp_path / "s.spz"
    count = write_spz(path, means, colors, opacity, scales, quats)
    assert count == n

    raw = gzip.decompress(path.read_bytes())
    magic, ver, num, sh_deg, frac, flags, reserved = struct.unpack("<IIIBBBB", raw[:16])
    assert magic == NGSP_MAGIC
    assert ver == VERSION
    assert num == n
    assert frac == FRACTIONAL_BITS
    assert sh_deg == 0 and flags == 0 and reserved == 0
    # positions(9N) + scales(3N) + rotations(3N) + alphas(N) + colors(3N)
    assert len(raw) == 16 + n * 9 + n * 3 + n * 3 + n + n * 3


def test_spz_positions_fixed_point(tmp_path: Path) -> None:
    # A known position encodes to the expected 24-bit fixed-point value.
    means = np.array([[1.0, 0.0, -1.0]], np.float32)
    write_spz(
        tmp_path / "p.spz",
        means,
        np.zeros((1, 3), np.float32),
        np.ones(1, np.float32),
        np.ones((1, 3), np.float32),
        np.array([[1, 0, 0, 0]], np.float32),
    )
    raw = gzip.decompress((tmp_path / "p.spz").read_bytes())
    pos = raw[16 : 16 + 9]
    x = pos[0] | (pos[1] << 8) | (pos[2] << 16)
    assert x == (1 << FRACTIONAL_BITS)  # 1.0 * 4096
