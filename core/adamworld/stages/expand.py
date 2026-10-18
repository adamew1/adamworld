"""Stage 6 — World expansion (walk to an edge, generate the adjoining region).

Clean-room, built on stages 2-5 and informed by WonderWorld's guided-depth idea:
place a camera at the frontier, outpaint the unseen region, boundary-align depth
to the existing splats, lift, and merge as a new tile in world.json. Wired up in
M4.
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
