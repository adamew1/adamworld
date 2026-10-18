"""Pydantic schema for ``world.json``.

FORMAT CONTRACT: ``format_version`` is 1. The build plan freezes this schema at
milestone M4; changes after that require maintainer sign-off and a version bump
plus a migration. Worlds are reproducible: ``seed`` + ``model_versions`` +
``params`` are sufficient to regenerate the world.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]  # (x, y, z, w)

FORMAT_VERSION = 1


class InputSpec(BaseModel):
    kind: str  # "prompt" | "image" | "panorama" | "layout"
    prompt: str | None = None
    assets: list[str] = Field(default_factory=list)  # relative asset filenames


class CameraSpawn(BaseModel):
    position: Vec3 = (0.0, 0.0, 0.0)
    rotation_quat: Quat = (0.0, 0.0, 0.0, 1.0)
    fov_deg: float = 70.0


class Bounds(BaseModel):
    aabb_min: Vec3 = (0.0, 0.0, 0.0)
    aabb_max: Vec3 = (0.0, 0.0, 0.0)


class LayoutBlock(BaseModel):
    id: str
    shape: str  # "box" | "plane" | "cylinder"
    transform: list[float] = Field(default_factory=lambda: _identity4x4())
    label: str = ""
    style: str = ""


class Layout(BaseModel):
    blocks: list[LayoutBlock] = Field(default_factory=list)
    style_prompt: str = ""


class Tile(BaseModel):
    id: str
    origin: Vec3 = (0.0, 0.0, 0.0)
    source: str = "initial"  # "initial" | "expand"
    splat_range: tuple[int, int] = (0, 0)


class WorldMeta(BaseModel):
    # "model_versions" would otherwise collide with pydantic's protected "model_" namespace.
    model_config = ConfigDict(protected_namespaces=())

    format_version: int = FORMAT_VERSION
    id: str
    name: str = ""
    created_at: str = ""
    seed: int = 0
    input: InputSpec
    model_versions: dict[str, str] = Field(default_factory=dict)
    params: dict[str, Any] = Field(default_factory=dict)
    camera_spawn: CameraSpawn = Field(default_factory=CameraSpawn)
    bounds: Bounds = Field(default_factory=Bounds)
    layout: Layout | None = None
    tiles: list[Tile] = Field(default_factory=list)
    history_head: str = "0000"


def _identity4x4() -> list[float]:
    return [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
