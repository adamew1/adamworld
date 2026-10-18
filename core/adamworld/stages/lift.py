"""Stage 4 — Lift RGB-D to 3D Gaussian splats.

The unprojection (equirect RGB + depth → Gaussian cloud) is pure numpy and runs
on CPU for every backend. On the ``default`` backend with a GPU, gsplat refines
these initial Gaussians against the input view(s); that optimization is wired up
in M1 and is skipped automatically when torch/gsplat or a GPU is unavailable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adamworld.imaging import load_rgb
from adamworld.splat.gaussians import GaussianCloud
from adamworld.stages.base import Stage, StageContext
from adamworld.types import DepthRef, PanoramaRef, SplatRef

MAX_POINTS = 800_000


class LiftStage(Stage):
    name = "lift"
    version = "1"

    def run(
        self,
        ctx: StageContext,
        *,
        panorama: PanoramaRef,
        depth: DepthRef,
        out_prefix: Path,
    ) -> SplatRef:
        rgb = load_rgb(panorama.path)
        depth_map = np.load(depth.path)
        cloud = GaussianCloud.from_rgbd_equirect(rgb, depth_map, max_points=MAX_POINTS)

        if ctx.backend != "mock":
            cloud = self._maybe_refine(cloud, rgb, depth_map, ctx)

        path = out_prefix.with_suffix(".ply")
        count = cloud.save_ply(path)
        lo, hi = cloud.aabb()
        return SplatRef(path=path, count=count, format="ply", aabb_min=lo, aabb_max=hi)

    def model_version(self, ctx: StageContext) -> str:
        return "numpy-unproject@1" if ctx.backend == "mock" else "gsplat-refit@1"

    # --- optional GPU refinement ------------------------------------------
    def _maybe_refine(
        self,
        cloud: GaussianCloud,
        rgb: np.ndarray,
        depth_map: np.ndarray,
        ctx: StageContext,
    ) -> GaussianCloud:
        if ctx.settings.device != "cuda":
            # No GPU -> keep the unprojected cloud (still a valid, viewable world).
            return cloud
        # M1: short gsplat optimization against the rendered equirect view(s).
        return cloud
