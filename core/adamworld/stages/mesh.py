"""Stage 7 — Mesh extraction (optional).

Default: render depth from the fitted splats at several views, TSDF-fuse
(Open3D, MIT), marching cubes, decimate, export GLB (trimesh, MIT). A coarse
collision/export mesh, not a beauty mesh. Wired up in M6.
"""

from __future__ import annotations

from pathlib import Path

from adamworld.stages.base import Stage, StageContext
from adamworld.types import MeshRef, SplatRef


class MeshStage(Stage):
    name = "mesh"
    version = "0"

    def run(self, ctx: StageContext, *, splats: SplatRef, out_prefix: Path) -> MeshRef:
        raise NotImplementedError("Mesh extraction lands in M6.")
