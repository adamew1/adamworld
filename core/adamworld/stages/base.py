"""Stage protocol and shared context.

Each pipeline stage is a small class with a stable ``name`` and ``version`` and a
``run`` method that writes its artifact(s) under ``out_prefix`` and returns a typed
payload. ``version`` participates in the cache key, so bumping it invalidates
stale cached outputs. Caching/profiling/OOM handling live in ``pipeline`` and
``cache`` so individual stages stay simple.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from adamworld.config import Settings, default_settings
from adamworld.profiling import Profiler

if TYPE_CHECKING:
    from adamworld.models.registry import ModelRegistry
    from adamworld.world import World


@dataclass
class StageContext:
    """Everything a stage needs besides its typed inputs."""

    world: World
    settings: Settings = field(default_factory=default_settings)
    profiler: Profiler = field(default_factory=Profiler)
    seed: int = 0
    resolution: int = 1024
    backend: str = "default"
    registry: ModelRegistry | None = None
    force: bool = False  # ignore cache and recompute

    def get_registry(self) -> ModelRegistry:
        if self.registry is None:
            from adamworld.models.registry import ModelRegistry

            self.registry = ModelRegistry(self.settings)
        return self.registry


class Stage:
    """Base class for pipeline stages."""

    name: str = "stage"
    version: str = "0"

    def cache_params(self, ctx: StageContext) -> dict[str, object]:
        """Parameters that affect output and must invalidate the cache."""
        return {
            "backend": ctx.backend,
            "resolution": ctx.resolution,
            "seed": ctx.seed,
            "device_family": "gpu" if ctx.settings.device == "cuda" else "cpu",
        }

    def model_version(self, ctx: StageContext) -> str:
        """Human-readable model id recorded in world.json (overridden per stage)."""
        return f"{self.name}:{ctx.backend}@{self.version}"
