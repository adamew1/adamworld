"""Typed payloads passed between pipeline stages.

Stages exchange lightweight *references* (a path plus small metadata), not large
arrays, so payloads are cheap to serialize, cache, and hash. Each payload carries
the ``key`` (content hash) that produced it, which chains into downstream cache
keys for content-addressed resume.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

Vec3 = tuple[float, float, float]


class StagePayload(BaseModel):
    """Base for all inter-stage payloads. ``key`` is set by the pipeline."""

    key: str = ""


class ImageRef(StagePayload):
    path: Path
    width: int
    height: int


class PanoramaRef(StagePayload):
    """Equirectangular panorama (width == 2 * height by convention)."""

    path: Path
    width: int
    height: int


class DepthRef(StagePayload):
    """Depth map stored as a float32 ``.npy`` of shape (height, width)."""

    path: Path
    width: int
    height: int
    min_depth: float
    max_depth: float
    metric: bool = False


class SplatRef(StagePayload):
    """A 3D Gaussian splat cloud on disk (PLY by default)."""

    path: Path
    count: int
    format: str = "ply"
    aabb_min: Vec3 = (0.0, 0.0, 0.0)
    aabb_max: Vec3 = (0.0, 0.0, 0.0)


class MeshRef(StagePayload):
    path: Path
    format: str = "glb"
    vertices: int = 0
    faces: int = 0


class GenerationRequest(BaseModel):
    """Everything needed to generate (and later reproduce) a world."""

    prompt: str | None = None
    image_paths: list[Path] = []
    panorama_path: Path | None = None
    seed: int = 0
    resolution: int = 1024
    backend: str = "default"

    def kind(self) -> str:
        if self.panorama_path is not None:
            return "panorama"
        if self.image_paths:
            return "image"
        if self.prompt is not None:
            return "prompt"
        raise ValueError("GenerationRequest needs a prompt, image, or panorama.")
