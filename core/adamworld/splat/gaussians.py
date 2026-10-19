"""``GaussianCloud``: an in-memory set of 3D Gaussians (numpy) plus the core
RGB-D → splat lift used by stage 4.

The lift unprojects an equirectangular panorama + depth map into world-space
points (one Gaussian per sampled pixel). Pure numpy so it runs on CPU and in
tests; gsplat optimization (GPU) refines these initial Gaussians in stage 4's
``default`` backend.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from adamworld.splat.ply_io import read_gaussian_ply, write_gaussian_ply
from adamworld.splat.spz_io import write_spz
from adamworld.types import Vec3


@dataclass
class GaussianCloud:
    means: np.ndarray  # (N, 3) float32
    colors: np.ndarray  # (N, 3) float32 in [0, 1]
    opacity: np.ndarray  # (N,) float32 in [0, 1]
    scales: np.ndarray  # (N, 3) float32 positive stddev
    quats: np.ndarray  # (N, 4) float32 (w, x, y, z)

    def __len__(self) -> int:
        return int(self.means.shape[0])

    @classmethod
    def empty(cls) -> GaussianCloud:
        return cls(
            means=np.zeros((0, 3), np.float32),
            colors=np.zeros((0, 3), np.float32),
            opacity=np.zeros((0,), np.float32),
            scales=np.zeros((0, 3), np.float32),
            quats=np.zeros((0, 4), np.float32),
        )

    def aabb(self) -> tuple[Vec3, Vec3]:
        if len(self) == 0:
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        lo = self.means.min(axis=0)
        hi = self.means.max(axis=0)
        return (float(lo[0]), float(lo[1]), float(lo[2])), (
            float(hi[0]),
            float(hi[1]),
            float(hi[2]),
        )

    @classmethod
    def concat(cls, clouds: list[GaussianCloud]) -> GaussianCloud:
        clouds = [c for c in clouds if len(c) > 0]
        if not clouds:
            return cls.empty()
        return cls(
            means=np.concatenate([c.means for c in clouds], axis=0),
            colors=np.concatenate([c.colors for c in clouds], axis=0),
            opacity=np.concatenate([c.opacity for c in clouds], axis=0),
            scales=np.concatenate([c.scales for c in clouds], axis=0),
            quats=np.concatenate([c.quats for c in clouds], axis=0),
        )

    def save_ply(self, path: Path) -> int:
        return write_gaussian_ply(
            path, self.means, self.colors, self.opacity, self.scales, self.quats
        )

    def save_spz(self, path: Path) -> int:
        return write_spz(path, self.means, self.colors, self.opacity, self.scales, self.quats)

    @classmethod
    def load_ply(cls, path: Path) -> GaussianCloud:
        d = read_gaussian_ply(path)
        return cls(d["means"], d["colors"], d["opacity"], d["scales"], d["quats"])

    @classmethod
    def from_rgbd_equirect(
        cls,
        rgb: np.ndarray,  # (H, W, 3) uint8 or float in [0, 1]
        depth: np.ndarray,  # (H, W) float, relative or metric
        *,
        depth_scale: float = 1.0,
        max_points: int = 500_000,
        min_scale: float = 1e-3,
        opacity: float = 0.99,
    ) -> GaussianCloud:
        """Unproject an equirectangular RGB-D pair into a Gaussian cloud.

        Convention: y is up, longitude spans [-pi, pi) across the width, latitude
        spans [+pi/2, -pi/2] down the height. The camera sits at the origin.
        """
        rgb = np.asarray(rgb)
        depth = np.asarray(depth, dtype=np.float32)
        h, w = depth.shape[:2]
        if rgb.shape[:2] != (h, w):
            raise ValueError(f"rgb {rgb.shape[:2]} and depth {(h, w)} must match")
        if rgb.dtype == np.uint8:
            rgb = rgb.astype(np.float32) / 255.0
        else:
            rgb = rgb.astype(np.float32)

        stride = max(1, int(round(np.sqrt(h * w / max(1, max_points)))))
        rows = np.arange(0, h, stride)
        cols = np.arange(0, w, stride)
        jj, ii = np.meshgrid(cols, rows)  # (R, C)

        u = (jj.astype(np.float32) + 0.5) / w
        v = (ii.astype(np.float32) + 0.5) / h
        lon = u * 2.0 * np.pi - np.pi
        lat = (0.5 * np.pi) - v * np.pi

        cos_lat = np.cos(lat)
        dir_x = cos_lat * np.sin(lon)
        dir_y = np.sin(lat)
        dir_z = cos_lat * np.cos(lon)
        dirs = np.stack([dir_x, dir_y, dir_z], axis=-1).reshape(-1, 3)

        d = (depth[ii, jj] * depth_scale).reshape(-1, 1)
        means = (dirs * d).astype(np.float32)

        colors = rgb[ii, jj].reshape(-1, 3).astype(np.float32)

        # Per-point size ~ angular pixel footprint * depth (isotropic).
        ang = (2.0 * np.pi / w) * float(stride)
        s = np.clip(d.reshape(-1) * ang, min_scale, None).astype(np.float32)
        scales = np.repeat(s[:, None], 3, axis=1)

        n = means.shape[0]
        opac = np.full((n,), float(opacity), dtype=np.float32)
        quats = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1))

        # Drop invalid / non-positive depth.
        valid = np.isfinite(d.reshape(-1)) & (d.reshape(-1) > 0)
        return cls(
            means=means[valid],
            colors=colors[valid],
            opacity=opac[valid],
            scales=scales[valid],
            quats=quats[valid],
        )
