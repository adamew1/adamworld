"""FastAPI app exposing world generation, status, retrieval, and export.

Endpoints (see docs/PLAN.md §4):
  POST /worlds                      -> enqueue generation, returns job + world id
  GET  /jobs/{job_id}               -> job status/progress
  GET  /worlds/{id}                 -> world.json + asset URLs
  GET  /worlds/{id}/asset/{name}    -> static asset (splats/thumbnail/...)
  GET  /worlds/{id}/export?format=  -> download ply|spz|glb
  POST /worlds/{id}/expand          -> stage 6 (M4)
  POST /worlds/{id}/edit            -> stage 8 (M5)
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from adamworld.api.jobs import InProcessJobQueue, JobQueue, ProgressCb
from adamworld.config import Settings, default_settings
from adamworld.pipeline import Pipeline
from adamworld.profiling import Profiler
from adamworld.types import GenerationRequest

_ASSET_ALLOWLIST = {"splats.ply", "splats.spz", "mesh.glb", "thumbnail.jpg", "world.json"}
_EXPORT_MAP = {"ply": "splats.ply", "spz": "splats.spz", "glb": "mesh.glb"}


class GenerateBody(BaseModel):
    prompt: str | None = None
    image_path: str | None = None
    panorama_path: str | None = None
    seed: int = 0
    resolution: int = 1024
    backend: str = "default"


def create_app(
    worlds_root: Path | None = None,
    settings: Settings | None = None,
    queue: JobQueue | None = None,
) -> FastAPI:
    worlds_root = Path(worlds_root or Path("worlds")).resolve()
    worlds_root.mkdir(parents=True, exist_ok=True)
    settings = settings or default_settings()
    queue = queue or InProcessJobQueue()

    app = FastAPI(title="AdamWorld", version="0.1.0.dev0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def world_dir(world_id: str) -> Path:
        d = (worlds_root / world_id).resolve()
        if worlds_root not in d.parents and d != worlds_root:
            raise HTTPException(400, "invalid world id")
        return d

    @app.post("/worlds")
    def create_world(body: GenerateBody) -> dict:
        world_id = _new_world_id()
        out_dir = worlds_root / world_id
        req = GenerationRequest(
            prompt=body.prompt,
            image_paths=[Path(body.image_path)] if body.image_path else [],
            panorama_path=Path(body.panorama_path) if body.panorama_path else None,
            seed=body.seed,
            resolution=body.resolution,
            backend=body.backend,
        )

        def _job(progress: ProgressCb) -> str:
            pipeline = Pipeline(settings=settings, profiler=Profiler(), progress=progress)
            pipeline.generate(req, out_dir)
            return world_id

        job_id = queue.submit(_job)
        return {"job_id": job_id, "world_id": world_id}

    @app.get("/jobs/{job_id}")
    def get_job(job_id: str) -> dict:
        job = queue.get(job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job.as_dict()

    @app.get("/worlds/{world_id}")
    def get_world(world_id: str) -> dict:
        d = world_dir(world_id)
        wj = d / "world.json"
        if not wj.exists():
            raise HTTPException(404, "world not found or not ready")
        meta = json.loads(wj.read_text(encoding="utf-8"))
        assets = {
            name: f"/worlds/{world_id}/asset/{name}"
            for name in _ASSET_ALLOWLIST
            if (d / name).exists()
        }
        return {"meta": meta, "assets": assets}

    @app.get("/worlds/{world_id}/asset/{name}")
    def get_asset(world_id: str, name: str) -> FileResponse:
        if name not in _ASSET_ALLOWLIST:
            raise HTTPException(404, "unknown asset")
        p = world_dir(world_id) / name
        if not p.exists():
            raise HTTPException(404, "asset not found")
        return FileResponse(p)

    @app.get("/worlds/{world_id}/export")
    def export_world(world_id: str, format: str = "ply") -> FileResponse:
        fname = _EXPORT_MAP.get(format)
        if fname is None:
            raise HTTPException(400, f"unsupported format '{format}' (use ply|spz|glb)")
        p = world_dir(world_id) / fname
        if not p.exists():
            raise HTTPException(404, f"{format} export not available for this world yet")
        return FileResponse(p, filename=f"{world_id}.{format}")

    @app.post("/worlds/{world_id}/expand")
    def expand_world(world_id: str) -> dict:
        raise HTTPException(501, "expansion lands in M4")

    @app.post("/worlds/{world_id}/edit")
    def edit_world(world_id: str) -> dict:
        raise HTTPException(501, "editing lands in M5")

    return app


def _new_world_id() -> str:
    import uuid

    return uuid.uuid4().hex[:12]
