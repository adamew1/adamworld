"""Coarse environment mesh from an equirectangular depth map (stage 7, CPU).

Builds a sphere-grid mesh displaced by depth with per-vertex colors from the
panorama, and exports it to GLB. This is a lightweight collision/export mesh (not
a beauty mesh); it reuses the depth we already compute, so it needs no GPU. A
higher-quality TSDF/SuGaR path can plug in later behind a flag.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adamworld import geometry
from adamworld.imaging import load_rgb
from adamworld.splat.glb_io import write_glb
from adamworld.types import MeshRef
from adamworld.world import World


def env_mesh_from_equirect(
    depth: np.ndarray,  # (H, W) radial depth
    rgb: np.ndarray,  # (H, W, 3)
    *,
    max_rows: int = 96,
    far_cap: float = 25.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (verts (V,3), faces (F,3) uint32, colors (V,3)) for the scene shell."""
    h, w = depth.shape[:2]
    gh = min(max_rows, h)
    gw = gh * 2
    # Resample depth + color to the mesh grid.
    ri = np.linspace(0, h - 1, gh).astype(np.int64)
    ci = np.linspace(0, w - 1, gw).astype(np.int64)
    d = depth[np.ix_(ri, ci)].astype(np.float32)
    col = (rgb.astype(np.float32) / 255.0)[np.ix_(ri, ci)]

    u = (np.arange(gw) + 0.5) / gw
    v = (np.arange(gh) + 0.5) / gh
    uu, vv = np.meshgrid(u, v)
    dirs = geometry.equirect_uv_to_dirs(uu, vv)
    verts = (dirs * d[..., None]).reshape(-1, 3).astype(np.float32)
    colors = col.reshape(-1, 3).astype(np.float32)

    def idx(i: int, j: int) -> int:
        return i * gw + (j % gw)

    faces = []
    for i in range(gh - 1):
        for j in range(gw):
            a, b, c, e = idx(i, j), idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)
            # Drop near-sky quads (keeps a bounded shell instead of a huge dome).
            if (
                min(d[i, j % gw], d[i, (j + 1) % gw], d[i + 1, j % gw], d[i + 1, (j + 1) % gw])
                >= far_cap
            ):
                continue
            faces.append((a, c, b))
            faces.append((b, c, e))
    faces_arr = np.asarray(faces, dtype=np.uint32) if faces else np.zeros((0, 3), np.uint32)
    return verts, faces_arr, colors


def _find_latest(folder: Path, pattern: str) -> Path | None:
    files = sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None


def mesh_world(world: World, max_rows: int = 96) -> MeshRef:
    """Build ``world/mesh.glb`` from the world's cached depth + panorama."""
    depth_npy = _find_latest(world.cache_dir / "depth", "*.npy")
    pano_png = _find_latest(world.cache_dir / "panorama", "*.png")
    if depth_npy is None or pano_png is None:
        raise FileNotFoundError(
            "Cannot build mesh: this world has no cached depth/panorama. "
            "Regenerate it (the stage cache must be present)."
        )
    depth = np.load(depth_npy)
    rgb = load_rgb(pano_png)
    verts, faces, colors = env_mesh_from_equirect(depth, rgb, max_rows=max_rows)
    v, f = write_glb(world.mesh_glb, verts, faces, colors)
    return MeshRef(path=world.mesh_glb, format="glb", vertices=v, faces=f)
