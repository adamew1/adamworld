"""Stage 2 — Image → 360° equirectangular panorama.

Default backend: SDXL equirectangular outpainting (OpenRAIL++, flagged default)
with horizontal circular padding + seam blending, wired up in M1. ``mock`` backend
synthesizes a plausible sky/ground equirect (placing the input image on the front
view) so the pipeline runs on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adamworld.imaging import load_rgb, resize, save_rgb, seeded_rng
from adamworld.stages.base import Stage, StageContext
from adamworld.types import ImageRef, PanoramaRef


class PanoramaStage(Stage):
    name = "panorama"
    version = "1"

    def run(
        self,
        ctx: StageContext,
        *,
        image: ImageRef | None,
        prompt: str,
        out_prefix: Path,
    ) -> PanoramaRef:
        height = max(16, min(ctx.resolution, 2048) // 2)
        width = height * 2
        if ctx.backend == "mock":
            arr = self._mock(image, prompt, width, height, ctx)
        else:
            arr = self._sdxl_outpaint(image, prompt, width, height, ctx)
        path = out_prefix.with_suffix(".png")
        save_rgb(path, arr)
        return PanoramaRef(path=path, width=width, height=height)

    def model_version(self, ctx: StageContext) -> str:
        if ctx.backend == "mock":
            return "mock-pano@1"
        if ctx.backend == "flux-pano":
            return "dit360-flux.1-dev(NC)"
        return "sdxl-equirect-outpaint(OpenRAIL)"

    # --- backends ----------------------------------------------------------
    def _mock(
        self,
        image: ImageRef | None,
        prompt: str,
        w: int,
        h: int,
        ctx: StageContext,
    ) -> np.ndarray:
        rng = seeded_rng(ctx.seed, "pano", prompt)
        # Sky (top) -> horizon -> ground (bottom) vertical gradient.
        sky = np.array([135, 180, 235], np.float32) + rng.integers(-15, 15, 3)
        horizon = np.array([225, 225, 220], np.float32)
        ground = np.array([70, 110, 60], np.float32) + rng.integers(-15, 15, 3)
        arr = np.zeros((h, w, 3), np.float32)
        for row in range(h):
            v = (row + 0.5) / h
            if v < 0.5:
                t = v / 0.5
                arr[row, :] = sky * (1 - t) + horizon * t
            else:
                t = (v - 0.5) / 0.5
                arr[row, :] = horizon * (1 - t) + ground * t
        # Place the input image as the front-facing view (centered band).
        if image is not None:
            src = load_rgb(image.path)
            bw, bh = max(1, w // 4), max(1, h // 2)
            band = resize(src, bw, bh).astype(np.float32)
            x0 = w // 2 - bw // 2
            y0 = int(h * 0.30)
            arr[y0 : y0 + bh, x0 : x0 + bw] = band
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _sdxl_outpaint(
        self,
        image: ImageRef | None,
        prompt: str,
        w: int,
        h: int,
        ctx: StageContext,
    ) -> np.ndarray:
        # M1: project seed image onto an equirect canvas, then SDXL-inpaint the
        # rest with horizontal circular padding + circular VAE blending for a
        # seamless 360 wrap. (flux-pano backend uses DiT360 / FLUX.1-dev, NC.)
        ctx.get_registry().sdxl_inpaint()
        raise NotImplementedError("Real panorama (SDXL outpaint) lands in M1.")
