"""Pipeline orchestrator.

Runs the stage DAG for a generation request, caching each stage by content hash
(resume-on-crash), profiling time + VRAM, translating CUDA OOM into an actionable
error, and writing the final ``world/`` artifacts. Models load/unload sequentially
so peak VRAM stays within ~16 GB.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from adamworld.cache import stage_cache_key
from adamworld.config import Settings, default_settings
from adamworld.imaging import load_rgb, resize, save_rgb, save_thumbnail
from adamworld.models.registry import ModelRegistry
from adamworld.profiling import OutOfMemoryError, Profiler, oom_message
from adamworld.splat import GaussianCloud
from adamworld.stages.base import Stage, StageContext
from adamworld.stages.depth import DepthStage
from adamworld.stages.lift import LiftStage
from adamworld.stages.panorama import PanoramaStage
from adamworld.stages.text_to_image import TextToImageStage
from adamworld.types import DepthRef, GenerationRequest, ImageRef, PanoramaRef, SplatRef
from adamworld.world import InputSpec, World

P = TypeVar("P", bound=BaseModel)

ProgressCb = Callable[[str, float], None]


def _is_cuda_oom(exc: BaseException) -> bool:
    if "out of memory" in str(exc).lower():
        return True
    try:
        import torch  # noqa: PLC0415

        return isinstance(exc, torch.cuda.OutOfMemoryError)
    except Exception:
        return False


@dataclass
class Pipeline:
    settings: Settings = field(default_factory=default_settings)
    profiler: Profiler = field(default_factory=Profiler)
    force: bool = False
    progress: ProgressCb | None = None

    def _emit(self, stage: str, frac: float) -> None:
        if self.progress is not None:
            self.progress(stage, frac)

    def _run_stage(
        self,
        stage: Stage,
        ctx: StageContext,
        *,
        inputs: BaseModel | None,
        out_cls: type[P],
        extra: dict[str, object],
        runner: Callable[[Path], P],
    ) -> P:
        params = {**stage.cache_params(ctx), **extra}
        key = stage_cache_key(stage.name, stage.version, inputs, params)
        stage_dir = ctx.world.cache_dir / stage.name
        cache_file = stage_dir / f"{key}.json"
        if cache_file.exists() and not ctx.force:
            return out_cls.model_validate_json(cache_file.read_text(encoding="utf-8"))

        out_prefix = stage_dir / key
        out_prefix.parent.mkdir(parents=True, exist_ok=True)
        with self.profiler.track(stage.name):
            try:
                payload = runner(out_prefix)
            except Exception as exc:  # noqa: BLE001
                if _is_cuda_oom(exc):
                    raise OutOfMemoryError(
                        oom_message(stage.name, ctx.resolution, ctx.backend)
                    ) from exc
                raise
        payload.key = key  # type: ignore[attr-defined]
        cache_file.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        # Free any GPU model this stage loaded before the next one loads.
        if ctx.registry is not None:
            ctx.registry.unload_all()
        return payload

    def generate(self, request: GenerationRequest, out_dir: Path) -> World:
        kind = request.kind()
        assets = [Path(p).name for p in request.image_paths]
        if request.panorama_path is not None:
            assets = [Path(request.panorama_path).name]
        world = World.create(
            out_dir,
            input=InputSpec(kind=kind, prompt=request.prompt, assets=assets),
            seed=request.seed,
            params={
                "resolution": request.resolution,
                "backend": request.backend,
                "offload": self.settings.offload,
            },
        )
        ctx = StageContext(
            world=world,
            settings=self.settings,
            profiler=self.profiler,
            seed=request.seed,
            resolution=request.resolution,
            backend=request.backend,
            registry=ModelRegistry(self.settings),
            force=self.force,
        )
        model_versions: dict[str, str] = {}

        t2i, pano_stage = TextToImageStage(), PanoramaStage()
        depth_stage, lift_stage = DepthStage(), LiftStage()

        # --- stages 1-2: reach an equirectangular panorama -----------------
        self._emit("panorama", 0.05)
        if request.panorama_path is not None:
            panorama = self._ingest_panorama(request.panorama_path, ctx)
        else:
            image = self._resolve_seed_image(request, ctx, t2i, model_versions)
            panorama = self._run_stage(
                pano_stage,
                ctx,
                inputs=image,
                out_cls=PanoramaRef,
                extra={"prompt": request.prompt or ""},
                runner=lambda prefix: pano_stage.run(
                    ctx, image=image, prompt=request.prompt or "", out_prefix=prefix
                ),
            )
            model_versions["panorama"] = pano_stage.model_version(ctx)

        # --- stage 3: depth ------------------------------------------------
        self._emit("depth", 0.45)
        depth = self._run_stage(
            depth_stage,
            ctx,
            inputs=panorama,
            out_cls=DepthRef,
            extra={},
            runner=lambda prefix: depth_stage.run(ctx, panorama=panorama, out_prefix=prefix),
        )
        model_versions["depth"] = depth_stage.model_version(ctx)

        # --- stage 4: lift to splats ---------------------------------------
        self._emit("lift", 0.70)
        splats = self._run_stage(
            lift_stage,
            ctx,
            inputs=_merge_refs(panorama, depth),
            out_cls=SplatRef,
            extra={},
            runner=lambda prefix: lift_stage.run(
                ctx, panorama=panorama, depth=depth, out_prefix=prefix
            ),
        )
        model_versions["lift"] = lift_stage.model_version(ctx)

        # --- finalize world ------------------------------------------------
        self._emit("finalize", 0.92)
        shutil.copyfile(splats.path, world.splats_ply)
        GaussianCloud.load_ply(world.splats_ply).save_spz(world.splats_spz)  # primary, Spark-native
        pano_rgb = load_rgb(panorama.path)
        save_thumbnail(world.thumbnail, pano_rgb)

        meta = world.meta()
        meta.model_versions = model_versions
        meta.bounds.aabb_min = splats.aabb_min
        meta.bounds.aabb_max = splats.aabb_max
        world.save_meta(meta)
        self._emit("done", 1.0)
        return world

    # --- helpers -----------------------------------------------------------
    def _resolve_seed_image(
        self,
        request: GenerationRequest,
        ctx: StageContext,
        t2i: TextToImageStage,
        model_versions: dict[str, str],
    ) -> ImageRef | None:
        if request.image_paths:
            src = request.image_paths[0]
            arr = load_rgb(src)
            dst = ctx.world.cache_dir / "input" / Path(src).name
            save_rgb(dst, arr)
            return ImageRef(
                path=dst, width=int(arr.shape[1]), height=int(arr.shape[0]), key="input"
            )
        if request.prompt is not None:
            image = self._run_stage(
                t2i,
                ctx,
                inputs=None,
                out_cls=ImageRef,
                extra={"prompt": request.prompt},
                runner=lambda prefix: t2i.run(ctx, prompt=request.prompt or "", out_prefix=prefix),
            )
            model_versions["text_to_image"] = t2i.model_version(ctx)
            return image
        return None

    def _ingest_panorama(self, path: Path, ctx: StageContext) -> PanoramaRef:
        arr = load_rgb(path)
        h, w = arr.shape[:2]
        if w != 2 * h:  # coerce to 2:1 equirect
            h = min(h, ctx.resolution // 2) or h
            w = 2 * h
            arr = resize(arr, w, h)
        dst = ctx.world.cache_dir / "panorama" / "user.png"
        save_rgb(dst, arr)
        return PanoramaRef(path=dst, width=w, height=h, key="user")


def _merge_refs(panorama: PanoramaRef, depth: DepthRef) -> BaseModel:
    class _LiftInputs(BaseModel):
        panorama: PanoramaRef
        depth: DepthRef

    return _LiftInputs(panorama=panorama, depth=depth)
