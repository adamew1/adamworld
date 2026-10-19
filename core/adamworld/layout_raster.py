"""Rasterize a block-out layout to an equirectangular depth + label map (stage 8).

The browser editor places unit primitives (box / plane / cylinder) each with a
column-major 4x4 transform (three.js convention). We cast a ray per equirect pixel
from the origin and keep the nearest primitive hit, producing a depth map and a
per-pixel label id. That depth drives generation: the ``mock`` backend lifts it
directly into a 3D block-out preview; the ``default`` backend feeds it to
ControlNet-depth (SDXL) so generation follows the blocked-out geometry.

Pure numpy. Affine ray transforms preserve the ray parameter ``t``, so a world-
space unit direction gives depth == t at the hit.
"""

from __future__ import annotations

import numpy as np

from adamworld import geometry
from adamworld.world.schema import Layout

FAR = 1e9


def _matrix(transform: list[float]) -> np.ndarray:
    # three.js elements are column-major; reshape+transpose -> row-major 4x4.
    return np.array(transform, np.float64).reshape(4, 4).T


def _hit_box(o: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Nearest t>0 for a unit cube [-0.5,0.5]^3 in local space (slab method)."""
    t = np.full(o.shape[0], FAR)
    tmin = np.full(o.shape[0], -FAR)
    tmax = np.full(o.shape[0], FAR)
    for ax in range(3):
        oa, da = o[:, ax], d[:, ax]
        parallel = np.abs(da) < 1e-9
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (-0.5 - oa) / da
            t2 = (0.5 - oa) / da
        lo = np.minimum(t1, t2)
        hi = np.maximum(t1, t2)
        tmin = np.where(parallel, tmin, np.maximum(tmin, lo))
        tmax = np.where(parallel, tmax, np.minimum(tmax, hi))
        # rays parallel and outside the slab never hit
        outside = parallel & ((oa < -0.5) | (oa > 0.5))
        tmax = np.where(outside, -FAR, tmax)
    hit = (tmax >= tmin) & (tmax > 0)
    tnear = np.where(tmin > 0, tmin, tmax)
    return np.where(hit & (tnear > 0), tnear, t)


def _hit_plane(o: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Unit plane at local z=0, |x|,|y| <= 0.5."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t = -o[:, 2] / d[:, 2]
    x = o[:, 0] + t * d[:, 0]
    y = o[:, 1] + t * d[:, 1]
    ok = (t > 0) & (np.abs(x) <= 0.5) & (np.abs(y) <= 0.5) & (np.abs(d[:, 2]) > 1e-9)
    return np.where(ok, t, FAR)


def _hit_cylinder(o: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Unit cylinder: radius 0.5 about local y axis, |y| <= 0.5 (no caps)."""
    ox, oz, dx, dz = o[:, 0], o[:, 2], d[:, 0], d[:, 2]
    a = dx * dx + dz * dz
    b = 2 * (ox * dx + oz * dz)
    c = ox * ox + oz * oz - 0.25
    disc = b * b - 4 * a * c
    out = np.full(o.shape[0], FAR)
    valid = (disc >= 0) & (np.abs(a) > 1e-12)
    sq = np.sqrt(np.where(valid, disc, 0.0))
    for t in ((-b - sq) / (2 * a), (-b + sq) / (2 * a)):
        y = o[:, 1] + t * d[:, 1]
        ok = valid & (t > 0) & (np.abs(y) <= 0.5)
        out = np.where(ok & (t < out), t, out)
    return out


_HITTERS = {"box": _hit_box, "plane": _hit_plane, "cylinder": _hit_cylinder}


def rasterize_layout(layout: Layout, height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (depth (H,W) float32, label_id (H,W) int32; -1 = background)."""
    vv, uu = np.mgrid[0:height, 0:width]
    u = (uu + 0.5) / width
    v = (vv + 0.5) / height
    dirs = geometry.equirect_uv_to_dirs(u, v).reshape(-1, 3).astype(np.float64)
    n = dirs.shape[0]
    best_t = np.full(n, FAR)
    best_id = np.full(n, -1, np.int32)

    for bi, block in enumerate(layout.blocks):
        hitter = _HITTERS.get(block.shape)
        if hitter is None:
            continue
        minv = np.linalg.inv(_matrix(block.transform))
        o_local = (minv @ np.array([0.0, 0.0, 0.0, 1.0]))[:3]
        d_local = dirs @ minv[:3, :3].T
        o_arr = np.broadcast_to(o_local, (n, 3))
        t = hitter(o_arr, d_local)
        closer = t < best_t
        best_t = np.where(closer, t, best_t)
        best_id = np.where(closer, bi, best_id)

    depth = np.where(best_t >= FAR, 0.0, best_t).astype(np.float32).reshape(height, width)
    return depth, best_id.reshape(height, width)


def layout_label_colors(layout: Layout, label_id: np.ndarray) -> np.ndarray:
    """Flat per-block colors keyed by label text (for the mock block-out preview)."""
    import hashlib

    h, w = label_id.shape
    rgb = np.zeros((h, w, 3), np.uint8)
    rgb[...] = (120, 130, 140)  # background
    for bi, block in enumerate(layout.blocks):
        seed = int(hashlib.sha256((block.label or block.id).encode()).hexdigest()[:6], 16)
        color = np.array([(seed >> 16) & 255, (seed >> 8) & 255, seed & 255], np.uint8)
        rgb[label_id == bi] = color
    return rgb
