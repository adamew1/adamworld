"""Seamless-360 helpers for the SDXL panorama stage (GPU).

Three techniques, applied together, to make the equirectangular output wrap
cleanly left↔right and stop the poles from collapsing:

1. ``circular_conv`` — patch every Conv2d in the UNet + VAE to pad **circularly
   on width** (and zero on height), so features wrap around the 360° seam.
2. ``latent_roll_callback`` — roll the latent horizontally by a random amount at
   each denoise step so no single column is privileged; the cumulative shift is
   tracked and undone at the end to keep content in place.
3. ``inpaint_poles`` — render the up/down cube faces, inpaint the stretched/empty
   pole regions, and composite them back into the equirect top/bottom bands.

Everything here touches torch/diffusers and runs only on the GPU backend; it's
imported lazily by ``stages.panorama``. The geometry it reuses (``geometry``) is
CPU-tested.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# --- 1. circular (width) padding on all convolutions -----------------------


def patch_circular_conv(*modules: Any) -> list[tuple[Any, Any]]:  # noqa: ANN401
    """Make every Conv2d pad circularly in W and zero in H. Returns a restore list."""
    import types

    import torch
    import torch.nn.functional as F

    def _asym_forward(self, inp, weight, bias):  # type: ignore[no-untyped-def]
        # self._reversed_padding_repeated_twice = (wl, wr, ht, hb)
        wl, wr, ht, hb = self._reversed_padding_repeated_twice
        x = F.pad(inp, (wl, wr, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, ht, hb), mode="constant")
        return F.conv2d(x, weight, bias, self.stride, (0, 0), self.dilation, self.groups)

    saved: list[tuple[Any, Any]] = []
    for module in modules:
        for m in module.modules():
            if isinstance(m, torch.nn.Conv2d) and m.padding_mode != "circular_w":
                saved.append((m, m._conv_forward))
                m.padding_mode = "circular_w"  # marker so we don't double-patch
                m._conv_forward = types.MethodType(_asym_forward, m)
    return saved


def unpatch_circular_conv(saved: list[tuple[Any, Any]]) -> None:
    for m, orig in saved:
        m._conv_forward = orig
        m.padding_mode = "zeros"


# --- 2. latent rolling during denoising ------------------------------------


class LatentRoller:
    """diffusers ``callback_on_step_end`` that rolls latents in W each step and
    tracks the cumulative shift so it can be undone after denoising."""

    def __init__(self) -> None:
        self.total = 0

    def __call__(self, pipe: Any, step: int, timestep: Any, kw: dict) -> dict:  # noqa: ANN401
        import torch

        lat = kw["latents"]
        w = lat.shape[-1]
        shift = int(torch.randint(0, w, (1,)).item())
        self.total = (self.total + shift) % w
        kw["latents"] = torch.roll(lat, shifts=shift, dims=-1)
        return kw

    def unroll_image(self, img: np.ndarray) -> np.ndarray:
        """Undo the cumulative shift on the decoded image (shift is in latent
        columns; VAE downscales by 8, so image shift = total * 8)."""
        if self.total == 0:
            return img
        px = (self.total * 8) % img.shape[1]
        return np.roll(img, shift=-px, axis=1)


# --- 3. pole (top/bottom) inpainting via cube faces ------------------------


def inpaint_poles(
    equirect: np.ndarray,  # (H, W, 3) uint8
    inpaint_pipe: Any,  # noqa: ANN401 - SDXL inpaint pipeline
    prompt: str,
    *,
    face: int = 512,
    steps: int = 20,
) -> np.ndarray:
    """Inpaint the up/down views and composite them back into the equirect."""
    from PIL import Image

    from adamworld import geometry

    h, w = equirect.shape[:2]
    accum = equirect.astype(np.float32).copy()
    weight = np.ones((h, w), np.float32)

    for lat0 in (np.pi / 2, -np.pi / 2):  # straight up, straight down
        view = geometry.equirect_to_perspective(equirect, 0.0, float(lat0), 90.0, face)
        # Missing pixels are near-black (unfilled) -> inpaint them.
        hole = (view.max(axis=2) < 12).astype(np.uint8) * 255
        if hole.mean() < 2:  # already full
            continue
        filled = np.asarray(
            inpaint_pipe(
                prompt=prompt or "seamless sky and ground, 360 panorama",
                image=Image.fromarray(view.astype(np.uint8)),
                mask_image=Image.fromarray(hole),
                width=face,
                height=face,
                num_inference_steps=steps,
            )
            .images[0]
            .convert("RGB"),
            dtype=np.float32,
        )
        geometry.perspective_to_equirect_accumulate(
            filled, 0.0, float(lat0), 90.0, h, w, accum, weight
        )

    return np.clip(accum / np.maximum(weight[..., None], 1e-6), 0, 255).astype(np.uint8)
