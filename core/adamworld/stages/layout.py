"""Stage 8 — Layout conditioning.

Coarse 3D blocks (from the browser block-out editor) are rendered to a depth (and
optional label) map and used to condition the stage-2 generator via ControlNet-
depth for SDXL, so generation follows the blocked-out geometry. Also drives region
regenerate/delete. Wired up in M5.
"""

from __future__ import annotations

from pathlib import Path

from adamworld.stages.base import Stage, StageContext
from adamworld.types import PanoramaRef
from adamworld.world.schema import Layout


class LayoutStage(Stage):
    name = "layout"
    version = "0"

    def run(self, ctx: StageContext, *, layout: Layout, out_prefix: Path) -> PanoramaRef:
        raise NotImplementedError("Layout conditioning lands in M5.")
