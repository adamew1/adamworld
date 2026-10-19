"""Stage 2 — Image → 360° equirectangular panorama.

Default backend: SDXL equirectangular outpainting (OpenRAIL++, flagged default)
with horizontal circular padding + seam blending, wired up in M1. ``mock`` backend
synthesizes a plausible sky/ground equirect (placing the input image on the front
view) so the pipeline runs on CPU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adamworld.imaging import load_rgb, save_rgb, seeded_rng
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
        # Project the input image onto the front view using the real equirect
        # geometry (same code path the SDXL backend conditions on).
        if image is not None:
            from adamworld import geometry  # noqa: PLC0415

            seed = load_rgb(image.path)
            canvas, known = geometry.place_image_front(seed, h, w, fov_deg=90.0)
            m = known[..., None]
            arr = arr * (1 - m) + canvas.astype(np.float32) * m
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _sdxl_outpaint(
        self,
        image: ImageRef | None,
        prompt: str,
        w: int,
        h: int,
        ctx: StageContext,
    ) -> np.ndarray:
        """Project the seed image onto an equirect canvas, then SDXL-inpaint the
        rest with seamless-360 fixes (see ``pano_seams``): circular-padded convs so
        the left/right edges wrap, optional latent rolling, and cube-face pole
        inpainting. (The ``flux-pano`` backend swaps in DiT360 / FLUX.1-dev, NC.)
        """
        from PIL import Image  # noqa: PLC0415

        from adamworld import geometry, pano_seams  # noqa: PLC0415

        if image is not None:
            seed = load_rgb(image.path)
            canvas, known = geometry.place_image_front(seed, h, w, fov_deg=90.0)
        else:
            canvas = self._mock(None, prompt, w, h, ctx)  # gradient seed
            known = np.zeros((h, w), np.float32)

        # diffusers inpaint convention: white (255) = fill, black = keep.
        inpaint_mask = ((known < 0.5) * 255).astype(np.uint8)
        full_prompt = (
            prompt or "a seamless 360 degree equirectangular panorama"
        ) + ", 360 panorama"
        pipe = ctx.get_registry().sdxl_inpaint()

        # Circular padding on every conv makes features wrap across the seam.
        saved = pano_seams.patch_circular_conv(pipe.unet, pipe.vae)
        # Latent rolling only when there is no fixed region (rolling desyncs the
        # inpaint mask / known-region blend, so skip it when a seed is pinned).
        roller = pano_seams.LatentRoller() if known.sum() == 0 else None
        extra = (
            {"callback_on_step_end": roller, "callback_on_step_end_tensor_inputs": ["latents"]}
            if roller is not None
            else {}
        )
        try:
            result = pipe(
                prompt=full_prompt,
                image=Image.fromarray(canvas),
                mask_image=Image.fromarray(inpaint_mask),
                width=w,
                height=h,
                num_inference_steps=30,
                guidance_scale=7.0,
                **extra,
            )
            out = np.asarray(result.images[0].convert("RGB"), dtype=np.uint8)
            if roller is not None:
                out = roller.unroll_image(out)
        finally:
            pano_seams.unpatch_circular_conv(saved)

        # Fix the poles (top/bottom cube faces).
        return pano_seams.inpaint_poles(out, pipe, prompt)
