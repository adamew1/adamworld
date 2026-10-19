"""Stage 5 — Fill occlusions (inpaint + lift + refit loop).

Clean-room: this implements the *method* described in the LucidDreamer paper
(Chung et al., arXiv:2311.13384) — render novel views, detect holes, inpaint,
estimate + align depth, lift, merge, iterate — written from scratch. No code from
the LucidDreamer repository (CC-BY-NC-SA-4.0) was read or used, and none of its
dependencies (their Inria-derived rasterizer, ZoeDepth) are used: rendering is via
gsplat (Apache-2.0), depth via Depth Anything V2, geometry via our ``geometry``
module. GPU-only; ``mock`` is a no-op passthrough.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adamworld.splat.gaussians import GaussianCloud
from adamworld.stages.base import Stage, StageContext
from adamworld.stages.depth import FAR, NEAR
from adamworld.types import PanoramaRef, SplatRef

_FOV = 90.0
_SIZE = 512
# Novel cameras: small steps off the origin, looking out along the horizon.
_OFFSETS = [(0.5, 0.0, 0.0), (-0.5, 0.0, 0.0), (0.0, 0.0, 0.5), (0.0, 0.0, -0.5)]
_LOOKS = [(0.0, 0.0), (np.pi / 2, 0.0), (np.pi, 0.0), (-np.pi / 2, 0.0)]


class OcclusionStage(Stage):
    name = "occlusion"
    version = "1"

    def run(
        self,
        ctx: StageContext,
        *,
        splats: SplatRef,
        panorama: PanoramaRef,
        out_prefix: Path,
        iterations: int = 2,
    ) -> SplatRef:
        if ctx.backend == "mock":
            return splats  # no-op; real hole-filling needs a GPU

        from PIL import Image

        from adamworld import geometry
        from adamworld.splat.render import render_view

        cloud = GaussianCloud.load_ply(splats.path)
        inpaint = ctx.get_registry().sdxl_inpaint()
        depth_pipe = ctx.get_registry().depth_anything(
            "large" if ctx.settings.allow_noncommercial else "small"
        )
        ang = np.radians(_FOV) / _SIZE

        for _ in range(iterations):
            for off in _OFFSETS:
                cam = np.array(off, np.float32)
                for look in _LOOKS:
                    rgb, alpha, depth_r = render_view(cloud, cam, look, _FOV, _SIZE)
                    hole = alpha < 0.5
                    if hole.mean() < 0.02:
                        continue
                    filled = np.asarray(
                        inpaint(
                            prompt="seamless scene continuation, same style",
                            image=Image.fromarray(rgb),
                            mask_image=Image.fromarray((hole * 255).astype(np.uint8)),
                            width=_SIZE,
                            height=_SIZE,
                            num_inference_steps=20,
                        )
                        .images[0]
                        .convert("RGB"),
                        dtype=np.uint8,
                    )
                    disp = np.asarray(depth_pipe(Image.fromarray(filled))["depth"], np.float32)
                    disp = (disp - disp.min()) / (disp.max() - disp.min() + 1e-6)
                    d_new = NEAR + (1.0 - disp) * (FAR - NEAR)
                    known = (~hole) & (depth_r > 0)
                    d_aligned, _, _ = geometry.fit_scale_shift(d_new, depth_r, known)

                    pts, cols = geometry.perspective_unproject(d_aligned, filled, _FOV, cam, look)
                    keep = hole.reshape(-1)
                    cloud = GaussianCloud.concat(
                        [cloud, _cloud_from_points(pts[keep], cols[keep], ang)]
                    )
            ctx.get_registry().unload_all()

        path = out_prefix.with_suffix(".ply")
        count = cloud.save_ply(path)
        lo, hi = cloud.aabb()
        return SplatRef(path=path, count=count, format="ply", aabb_min=lo, aabb_max=hi)

    def model_version(self, ctx: StageContext) -> str:
        return "noop" if ctx.backend == "mock" else "inpaint-lift-refit@1"


def _cloud_from_points(pts: np.ndarray, cols: np.ndarray, ang: float) -> GaussianCloud:
    n = pts.shape[0]
    if n == 0:
        return GaussianCloud.empty()
    dist = np.linalg.norm(pts, axis=1, keepdims=True)
    scale = np.clip(dist * ang, 1e-3, None).repeat(3, axis=1).astype(np.float32)
    return GaussianCloud(
        means=pts.astype(np.float32),
        colors=(cols.astype(np.float32) / 255.0 if cols.max() > 1.0 else cols.astype(np.float32)),
        opacity=np.full((n,), 0.99, np.float32),
        scales=scale,
        quats=np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (n, 1)),
    )
