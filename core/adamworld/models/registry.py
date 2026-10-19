"""Lazy model registry.

Weights live outside the repo in ``settings.weights_dir`` and are downloaded on
first use. Models are loaded one at a time and unloaded between stages so peak
VRAM stays within ~16 GB (``unload_all`` frees the GPU between sequential stages).

The GPU backends (SDXL, Depth Anything, FLUX) are wired up in milestones M1/M3;
until then their loaders raise an actionable ``NotImplementedError``. The CPU
``mock`` backends used by tests never touch this registry.
"""

from __future__ import annotations

from typing import Any

from adamworld.config import Settings

# Default open checkpoints (all recorded in docs/LICENSES.md).
FLUX_SCHNELL_REPO = "black-forest-labs/FLUX.1-schnell"  # Apache-2.0
SDXL_BASE_REPO = "stabilityai/stable-diffusion-xl-base-1.0"  # OpenRAIL++ (flagged default)
SDXL_INPAINT_REPO = "diffusers/stable-diffusion-xl-1.0-inpainting-0.1"  # OpenRAIL++
CONTROLNET_DEPTH_SDXL = "diffusers/controlnet-depth-sdxl-1.0"  # openrail++
# Depth Anything V2: ONLY Small is Apache-2.0; Base/Large/Giant are CC-BY-NC-4.0.
DEPTH_ANYTHING_V2 = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
    "giant": "depth-anything/Depth-Anything-V2-Giant-hf",
}
DEPTH_ANYTHING_APACHE = {"small"}  # the rest require allow_noncommercial


def _require_gpu_deps(what: str) -> None:
    try:
        import torch  # noqa: F401, PLC0415
    except Exception as exc:  # pragma: no cover - env dependent
        raise NotImplementedError(
            f"{what} needs the GPU extra. Install with:  pip install -e '.[gpu]'  "
            f"(and run on a CUDA GPU), or use '--backend mock' for a CPU placeholder."
        ) from exc


class ModelRegistry:
    """Caches loaded models for the duration of a run."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._loaded: dict[str, Any] = {}

    # --- utilities ---------------------------------------------------------
    def weights_path(self) -> str:
        return str(self.settings.weights_dir)

    def unload_all(self) -> None:
        """Drop references and free GPU memory between sequential stages."""
        self._loaded.clear()
        try:
            import gc

            import torch  # noqa: PLC0415

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def _torch_dtype(self) -> Any:  # noqa: ANN401
        import torch  # noqa: PLC0415

        return torch.float16 if self.settings.dtype == "float16" else torch.float32

    # --- loaders ----------------------------------------------------------
    def sdxl_text(self) -> Any:  # noqa: ANN401
        """SDXL base text-to-image (OpenRAIL++-M). The default stage-1 model on a T4."""
        _require_gpu_deps("SDXL text-to-image (stage 1 default)")
        if "sdxl_text" in self._loaded:
            return self._loaded["sdxl_text"]
        from diffusers import StableDiffusionXLPipeline  # noqa: PLC0415

        pipe = StableDiffusionXLPipeline.from_pretrained(
            SDXL_BASE_REPO,
            torch_dtype=self._torch_dtype(),
            variant="fp16",
            use_safetensors=True,
            cache_dir=self.weights_path(),
        )
        if self.settings.offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.enable_vae_tiling()
        pipe.set_progress_bar_config(disable=True)
        self._loaded["sdxl_text"] = pipe
        return pipe

    def flux_schnell(self) -> Any:  # noqa: ANN401 - diffusers pipeline
        """FLUX.1-schnell (Apache-2.0), bf16 + sequential CPU offload to fit a T4."""
        _require_gpu_deps("FLUX.1-schnell (stage 1 default)")
        if "flux_schnell" in self._loaded:
            return self._loaded["flux_schnell"]
        import torch  # noqa: PLC0415
        from diffusers import FluxPipeline  # noqa: PLC0415

        pipe = FluxPipeline.from_pretrained(
            FLUX_SCHNELL_REPO,
            torch_dtype=torch.bfloat16,  # FLUX is trained/released in bf16
            cache_dir=self.weights_path(),
        )
        # 12B params: sequential CPU offload is required on 16 GB; otherwise .to(cuda).
        if self.settings.offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.vae.enable_tiling()
        pipe.set_progress_bar_config(disable=True)
        self._loaded["flux_schnell"] = pipe
        return pipe

    def sdxl_inpaint(self) -> Any:  # noqa: ANN401
        """SDXL inpainting pipeline (OpenRAIL++). Runs on a T4 with offload/tiling."""
        _require_gpu_deps("SDXL inpaint (stages 2/5/8 default)")
        if "sdxl_inpaint" in self._loaded:
            return self._loaded["sdxl_inpaint"]
        from diffusers import AutoPipelineForInpainting  # noqa: PLC0415

        pipe = AutoPipelineForInpainting.from_pretrained(
            SDXL_INPAINT_REPO,
            torch_dtype=self._torch_dtype(),
            variant="fp16",
            cache_dir=self.weights_path(),
        )
        if self.settings.offload:
            pipe.enable_sequential_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.enable_vae_tiling()
        pipe.set_progress_bar_config(disable=True)
        self._loaded["sdxl_inpaint"] = pipe
        return pipe

    def depth_anything(self, size: str = "small") -> Any:  # noqa: ANN401
        """Depth Anything V2 depth-estimation pipeline.

        Default ``small`` is Apache-2.0. ``base``/``large``/``giant`` are
        CC-BY-NC-4.0 and require ``settings.allow_noncommercial=True``.
        """
        if size not in DEPTH_ANYTHING_V2:
            raise ValueError(f"unknown Depth Anything size '{size}'")
        if size not in DEPTH_ANYTHING_APACHE and not self.settings.allow_noncommercial:
            raise PermissionError(
                f"Depth Anything V2 '{size}' is CC-BY-NC-4.0 (non-commercial). "
                f"Default is 'small' (Apache-2.0). To use it, pass --allow-noncommercial "
                f"(or Settings(allow_noncommercial=True)). See docs/LICENSES.md."
            )
        _require_gpu_deps("Depth Anything V2 (stage 3 default)")
        key = f"depth_anything_{size}"
        if key in self._loaded:
            return self._loaded[key]
        import torch  # noqa: PLC0415
        from transformers import pipeline  # noqa: PLC0415

        pipe = pipeline(
            task="depth-estimation",
            model=DEPTH_ANYTHING_V2[size],
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=self._torch_dtype(),
        )
        self._loaded[key] = pipe
        return pipe
