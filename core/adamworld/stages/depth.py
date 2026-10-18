"""Stage 3 — Depth.

Default backend: Depth Anything V2 (Apache-2.0) over tangent-plane crops of the
panorama, fused back to an equirect depth map (M1). ``mock`` backend returns a
geometrically meaningful placeholder: a flat ground plane below eye height and a
far backdrop above the horizon, so the lift produces recognizable 3D structure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adamworld.stages.base import Stage, StageContext
from adamworld.types import DepthRef, PanoramaRef

EYE_HEIGHT = 1.6
FAR = 30.0


class DepthStage(Stage):
    name = "depth"
    version = "1"

    def run(self, ctx: StageContext, *, panorama: PanoramaRef, out_prefix: Path) -> DepthRef:
        h, w = panorama.height, panorama.width
        if ctx.backend == "mock":
            depth = self._mock(w, h)
            metric = False
        else:
            depth = self._depth_anything(panorama, ctx)
            metric = False
        path = out_prefix.with_suffix(".npy")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, depth.astype(np.float32))
        return DepthRef(
            path=path,
            width=w,
            height=h,
            min_depth=float(depth.min()),
            max_depth=float(depth.max()),
            metric=metric,
        )

    def model_version(self, ctx: StageContext) -> str:
        return "mock-depth@1" if ctx.backend == "mock" else "depth-anything-v2-large"

    # --- backends ----------------------------------------------------------
    def _mock(self, w: int, h: int) -> np.ndarray:
        rows = np.arange(h)
        v = (rows + 0.5) / h
        lat = (0.5 * np.pi) - v * np.pi
        dir_y = np.sin(lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            floor = np.where(dir_y < -0.02, EYE_HEIGHT / np.maximum(-dir_y, 1e-3), FAR)
        depth_rows = np.clip(floor, 0.1, FAR)
        return np.repeat(depth_rows[:, None], w, axis=1).astype(np.float32)

    def _depth_anything(self, panorama: PanoramaRef, ctx: StageContext) -> np.ndarray:
        # M1: tangent-plane projection -> Depth Anything V2 per crop -> fuse.
        ctx.get_registry().depth_anything("large")
        raise NotImplementedError("Real depth (Depth Anything V2) lands in M1.")
