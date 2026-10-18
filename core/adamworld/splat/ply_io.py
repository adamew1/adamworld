"""Read/write 3D Gaussian Splatting ``.ply`` files (the INRIA/de-facto layout).

Stored per splat: position (x,y,z), zero normals, SH degree-0 color as f_dc_*,
logit-opacity, log-scale, and a scalar-first quaternion (rot_0=w). Values use the
same encodings that gsplat / Spark / other 3DGS viewers expect, so output renders
directly. SH degree-0 only (no ``f_rest_*``); higher bands can be added later.

Pure numpy — no torch — so this works in CPU-only environments and tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814

_FIELDS = [
    "x", "y", "z",
    "nx", "ny", "nz",
    "f_dc_0", "f_dc_1", "f_dc_2",
    "opacity",
    "scale_0", "scale_1", "scale_2",
    "rot_0", "rot_1", "rot_2", "rot_3",
]  # fmt: skip


def _inverse_sigmoid(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1.0 - 1e-6)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def write_gaussian_ply(
    path: Path,
    means: np.ndarray,  # (N, 3) world-space
    colors: np.ndarray,  # (N, 3) linear RGB in [0, 1]
    opacity: np.ndarray,  # (N,) in [0, 1]
    scales: np.ndarray,  # (N, 3) positive world-space stddev
    quats: np.ndarray,  # (N, 4) (w, x, y, z), assumed normalized
) -> int:
    """Write a binary-little-endian 3DGS PLY. Returns the splat count."""
    path = Path(path)
    means = np.ascontiguousarray(means, dtype=np.float32)
    colors = np.ascontiguousarray(colors, dtype=np.float32)
    opacity = np.ascontiguousarray(opacity, dtype=np.float32).reshape(-1)
    scales = np.ascontiguousarray(scales, dtype=np.float32)
    quats = np.ascontiguousarray(quats, dtype=np.float32)
    n = means.shape[0]
    if not (colors.shape[0] == opacity.shape[0] == scales.shape[0] == quats.shape[0] == n):
        raise ValueError("All gaussian attribute arrays must share the same length N.")

    f_dc = (colors - 0.5) / SH_C0
    opacity_logit = _inverse_sigmoid(opacity)
    scale_log = np.log(np.clip(scales, 1e-9, None))

    dtype = np.dtype([(f, "<f4") for f in _FIELDS])
    arr = np.zeros(n, dtype=dtype)
    arr["x"], arr["y"], arr["z"] = means[:, 0], means[:, 1], means[:, 2]
    # normals left at zero
    arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"] = f_dc[:, 0], f_dc[:, 1], f_dc[:, 2]
    arr["opacity"] = opacity_logit
    arr["scale_0"], arr["scale_1"], arr["scale_2"] = (
        scale_log[:, 0],
        scale_log[:, 1],
        scale_log[:, 2],
    )
    arr["rot_0"], arr["rot_1"] = quats[:, 0], quats[:, 1]
    arr["rot_2"], arr["rot_3"] = quats[:, 2], quats[:, 3]

    header = "ply\nformat binary_little_endian 1.0\n"
    header += f"element vertex {n}\n"
    header += "".join(f"property float {f}\n" for f in _FIELDS)
    header += "end_header\n"

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(arr.tobytes())
    return n


def read_gaussian_ply(path: Path) -> dict[str, np.ndarray]:
    """Read a 3DGS PLY written by :func:`write_gaussian_ply`.

    Returns decoded arrays: ``means`` (N,3), ``colors`` (N,3) in [0,1],
    ``opacity`` (N,), ``scales`` (N,3), ``quats`` (N,4). Ignores ``f_rest_*``.
    """
    path = Path(path)
    with open(path, "rb") as f:
        # --- header ---
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fmt = f.readline().strip()
        if fmt != b"format binary_little_endian 1.0":
            raise ValueError(f"Unsupported PLY format: {fmt!r} (expected binary LE 1.0)")
        n = 0
        props: list[str] = []
        while True:
            line = f.readline().strip()
            if line.startswith(b"element vertex"):
                n = int(line.split()[-1])
            elif line.startswith(b"property"):
                props.append(line.split()[-1].decode("ascii"))
            elif line == b"end_header":
                break
        dtype = np.dtype([(p, "<f4") for p in props])
        arr = np.frombuffer(f.read(n * dtype.itemsize), dtype=dtype, count=n)

    means = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(np.float32)
    f_dc = np.stack([arr["f_dc_0"], arr["f_dc_1"], arr["f_dc_2"]], axis=1).astype(np.float32)
    colors = np.clip(f_dc * SH_C0 + 0.5, 0.0, 1.0)
    opacity = _sigmoid(arr["opacity"].astype(np.float32))
    scales = np.exp(
        np.stack([arr["scale_0"], arr["scale_1"], arr["scale_2"]], axis=1).astype(np.float32)
    )
    quats = np.stack([arr["rot_0"], arr["rot_1"], arr["rot_2"], arr["rot_3"]], axis=1).astype(
        np.float32
    )
    return {
        "means": means,
        "colors": colors,
        "opacity": opacity,
        "scales": scales,
        "quats": quats,
    }
