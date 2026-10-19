"""Write Niantic ``.spz`` files (MIT format) — ~10x smaller than PLY.

SPZ v2 layout (little-endian, then gzip-compressed): a 16-byte header followed by
per-attribute arrays in the order positions, alphas, colors, scales, rotations,
(spherical harmonics). Quantization matches the Niantic reference so Spark and
other SPZ viewers decode it correctly:

  positions : 24-bit fixed-point, value = round(coord * 2**fractionalBits)
  scales    : uint8 = round((ln(scale) + 10) * 16)
  rotations : 3x uint8 = round(q.xyz * 127.5 + 127.5), with w >= 0
  alphas    : uint8 = round(opacity * 255)
  colors    : uint8 = round(f_dc * colorScale * 255 + 127.5),  f_dc = (rgb-0.5)/SH_C0

Pure numpy — no torch.
"""

from __future__ import annotations

import gzip
import struct
from pathlib import Path

import numpy as np

from adamworld.splat.ply_io import SH_C0

NGSP_MAGIC = 0x5053474E  # 'NGSP'
VERSION = 2
FRACTIONAL_BITS = 12
COLOR_SCALE = 0.15  # Niantic SPZ color scale constant (encoder/decoder must agree)


def _u8(x: np.ndarray) -> np.ndarray:
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def write_spz(
    path: Path,
    means: np.ndarray,  # (N, 3)
    colors: np.ndarray,  # (N, 3) linear RGB in [0, 1]
    opacity: np.ndarray,  # (N,) in [0, 1]
    scales: np.ndarray,  # (N, 3) positive stddev
    quats: np.ndarray,  # (N, 4) (w, x, y, z)
) -> int:
    """Write a gzip-compressed SPZ v2 file. Returns the splat count."""
    means = np.ascontiguousarray(means, np.float32)
    colors = np.ascontiguousarray(colors, np.float32)
    opacity = np.ascontiguousarray(opacity, np.float32).reshape(-1)
    scales = np.ascontiguousarray(scales, np.float32)
    quats = np.ascontiguousarray(quats, np.float32)
    n = means.shape[0]

    # positions: 24-bit signed fixed-point, little-endian 3 bytes/coord.
    fixed = np.round(means * (1 << FRACTIONAL_BITS)).astype(np.int32).reshape(-1)
    pos = np.empty((fixed.size, 3), np.uint8)
    pos[:, 0] = fixed & 0xFF
    pos[:, 1] = (fixed >> 8) & 0xFF
    pos[:, 2] = (fixed >> 16) & 0xFF
    pos_bytes = pos.reshape(-1).tobytes()

    # scales: uint8 of log-scale.
    log_scale = np.log(np.clip(scales, 1e-9, None))
    scale_bytes = _u8((log_scale + 10.0) * 16.0).reshape(-1).tobytes()

    # rotations: first three quaternion components, w >= 0.
    q = quats / (np.linalg.norm(quats, axis=1, keepdims=True) + 1e-9)
    q = np.where(q[:, :1] < 0, -q, q)  # enforce w >= 0
    rot = _u8(q[:, 1:4] * 127.5 + 127.5).reshape(-1).tobytes()

    # alphas: opacity already in [0,1].
    alpha_bytes = _u8(opacity * 255.0).tobytes()

    # colors: via SH DC + color scale.
    f_dc = (colors - 0.5) / SH_C0
    color_bytes = _u8(f_dc * (COLOR_SCALE * 255.0) + 0.5 * 255.0).reshape(-1).tobytes()

    header = struct.pack(
        "<IIIBBBB", NGSP_MAGIC, VERSION, n, 0, FRACTIONAL_BITS, 0, 0
    )  # shDegree=0, flags=0, reserved=0
    # Field order per the Niantic SPZ reference: positions, alphas, colors, scales, rotations, sh.
    payload = header + pos_bytes + alpha_bytes + color_bytes + scale_bytes + rot

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(gzip.compress(payload))
    return n
