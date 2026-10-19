"""Stage 6 — World expansion (walk to an edge, generate the adjoining region).

Clean-room, built on stages 2-5. The approach is informed by the WonderWorld
*paper's* guided-depth idea; no code from the WonderWorld repository (which
publishes no license, i.e. all rights reserved) was read or used. Place a camera
at the frontier, generate the unseen region, boundary-align depth to the existing
splats, lift, and merge as a new tile in world.json. (The runnable mechanics live
in ``Pipeline.expand``.)
"""

from __future__ import annotations

from pathlib import Path

from adamworld.stages.base import Stage, StageContext
from adamworld.types import SplatRef


class ExpandStage(Stage):
    name = "expand"
    version = "0"

    def run(
        self,
        ctx: StageContext,
        *,
        splats: SplatRef,
        direction: str,
        prompt: str,
        out_prefix: Path,
    ) -> SplatRef:
        raise NotImplementedError("World expansion lands in M4.")
