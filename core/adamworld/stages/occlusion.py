"""Stage 5 — Fill occlusions (inpaint + lift + refit loop).

Clean-room reimplementation of the LucidDreamer "Dreaming + Alignment" loop on
gsplat: render a novel view -> detect holes -> SDXL-inpaint -> estimate depth ->
align to existing geometry -> lift new pixels -> add Gaussians -> refit. Wired up
in M3. ``mock`` backend is a no-op passthrough.
"""

from __future__ import annotations

from pathlib import Path

from adamworld.stages.base import Stage, StageContext
from adamworld.types import PanoramaRef, SplatRef


class OcclusionStage(Stage):
    name = "occlusion"
    version = "0"

    def run(
        self,
        ctx: StageContext,
        *,
        splats: SplatRef,
        panorama: PanoramaRef,
        out_prefix: Path,
        iterations: int = 3,
    ) -> SplatRef:
        if ctx.backend == "mock":
            return splats  # no-op; real hole-filling lands in M3
        raise NotImplementedError("Occlusion filling lands in M3.")
