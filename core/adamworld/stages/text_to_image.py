"""Stage 1 — Text → seed image.

Default backend: FLUX.1-schnell (Apache-2.0), wired up in M3. ``mock`` backend
produces a deterministic placeholder from the prompt+seed so the pipeline runs on
CPU with no model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adamworld.imaging import save_rgb, seeded_rng
from adamworld.stages.base import Stage, StageContext
from adamworld.types import ImageRef


class TextToImageStage(Stage):
    name = "text_to_image"
    version = "1"

    def run(self, ctx: StageContext, *, prompt: str, out_prefix: Path) -> ImageRef:
        if ctx.backend == "mock":
            arr = self._mock(prompt, ctx)
        else:
            arr = self._flux(prompt, ctx)
        path = out_prefix.with_suffix(".png")
        save_rgb(path, arr)
        return ImageRef(path=path, width=int(arr.shape[1]), height=int(arr.shape[0]))

    def model_version(self, ctx: StageContext) -> str:
        return "mock-t2i@1" if ctx.backend == "mock" else "flux.1-schnell"

    # --- backends ----------------------------------------------------------
    def _mock(self, prompt: str, ctx: StageContext) -> np.ndarray:
        r = max(16, min(ctx.resolution, 1024))
        rng = seeded_rng(ctx.seed, "t2i", prompt)
        top = rng.integers(40, 220, size=3)
        bottom = rng.integers(40, 220, size=3)
        t = np.linspace(0.0, 1.0, r)[:, None, None]
        img = (top[None, None, :] * (1 - t) + bottom[None, None, :] * t).repeat(r, axis=1)
        # A few soft blobs so the placeholder has structure to lift.
        yy, xx = np.mgrid[0:r, 0:r]
        for _ in range(4):
            cx, cy = rng.integers(0, r, size=2)
            rad = rng.integers(r // 8 + 1, r // 3 + 1)
            color = rng.integers(0, 256, size=3)
            m = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * rad**2)))[..., None]
            img = img * (1 - m) + color[None, None, :] * m
        return np.clip(img, 0, 255).astype(np.uint8)

    def _flux(self, prompt: str, ctx: StageContext) -> np.ndarray:
        # Implemented in M3 using the FLUX.1-schnell pipeline (fp8 + offload on T4).
        ctx.get_registry().flux_schnell()
        raise NotImplementedError("Real text->image (FLUX.1-schnell) lands in M3.")
