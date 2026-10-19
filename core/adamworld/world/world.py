"""The ``World`` object: on-disk layout, metadata IO, and versioned history.

Directory layout (see docs/PLAN.md §3)::

    world/
      splats.spz            # primary (added in M6; PLY is always written)
      splats.ply            # lossless fallback / interchange
      mesh.glb              # optional (stage 7)
      world.json            # metadata (WorldMeta)
      thumbnail.jpg
      history/              # NNNN.json edit ops
      cache/<stage>/<key>.* # content-addressed intermediates (resume)
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from adamworld.world.schema import FORMAT_VERSION, InputSpec, WorldMeta


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class World:
    """Wraps a world directory on disk."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.history_dir.mkdir(parents=True, exist_ok=True)

    # --- paths -------------------------------------------------------------
    @property
    def world_json(self) -> Path:
        return self.root / "world.json"

    @property
    def splats_ply(self) -> Path:
        return self.root / "splats.ply"

    @property
    def splats_spz(self) -> Path:
        return self.root / "splats.spz"

    @property
    def mesh_glb(self) -> Path:
        return self.root / "mesh.glb"

    @property
    def thumbnail(self) -> Path:
        return self.root / "thumbnail.jpg"

    @property
    def cache_dir(self) -> Path:
        return self.root / "cache"

    @property
    def history_dir(self) -> Path:
        return self.root / "history"

    # --- lifecycle ---------------------------------------------------------
    @classmethod
    def create(
        cls,
        root: Path,
        *,
        input: InputSpec,
        seed: int = 0,
        name: str = "",
        params: dict | None = None,
    ) -> World:
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        world = cls(root)
        meta = WorldMeta(
            id=uuid.uuid4().hex,
            name=name or root.name,
            created_at=_utc_now_iso(),
            seed=seed,
            input=input,
            params=params or {},
        )
        world.save_meta(meta)
        return world

    @classmethod
    def load(cls, root: Path) -> World:
        root = Path(root)
        if not (root / "world.json").exists():
            raise FileNotFoundError(f"No world.json in {root}")
        data = json.loads((root / "world.json").read_text(encoding="utf-8"))
        fv = int(data.get("format_version", 1))
        if fv > FORMAT_VERSION:
            raise ValueError(
                f"world.json format_version {fv} is newer than this build supports "
                f"({FORMAT_VERSION}). Upgrade adamworld."
            )
        return cls(root)

    # --- metadata ----------------------------------------------------------
    def save_meta(self, meta: WorldMeta) -> None:
        self.world_json.write_text(meta.model_dump_json(indent=2), encoding="utf-8")

    def meta(self) -> WorldMeta:
        data = json.loads(self.world_json.read_text(encoding="utf-8"))
        return WorldMeta.model_validate(data)

    # --- history -----------------------------------------------------------
    def add_edit(self, op: dict) -> str:
        """Append a versioned edit op; returns the new history id (e.g. "0003")."""
        meta = self.meta()
        next_n = int(meta.history_head) + 1
        hid = f"{next_n:04d}"
        record = {"id": hid, "at": _utc_now_iso(), "op": op}
        (self.history_dir / f"{hid}.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        meta.history_head = hid
        self.save_meta(meta)
        return hid

    def history(self) -> list[dict]:
        out = []
        for p in sorted(self.history_dir.glob("[0-9]*.json")):
            out.append(json.loads(p.read_text(encoding="utf-8")))
        return out
