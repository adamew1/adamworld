> Historical design document. Implementation status and limitations are maintained in [README](../README.md) and [VERIFICATION](VERIFICATION.md); planned capabilities below are not release claims.

# ARCHITECTURE.md

How the pieces fit together. See `PLAN.md` for the full module tree and the
Mermaid diagram; this doc explains the runtime contracts.

## Layers

```
CLI (typer)  ─┐
              ├─> Pipeline ──> Stage[1..8] ──> models.registry (lazy, GPU)
API (FastAPI)─┘        │            │
   + JobQueue          │            └─ each stage: run(ctx, inputs, out_prefix) -> Payload
                       │
                       └─ caching (content hash) · profiling (time/VRAM) · World (disk)

Web (Vite+TS+three+Spark)  ──fetch──>  API  ──reads──>  world/ on disk
```

## Core contracts

- **Stage** (`stages/base.py`): a class with a stable `name` + `version`. Its
  `run` writes artifacts under a caller-provided `out_prefix` and returns a typed
  pydantic payload (`types.py`). Stages never manage their own caching or VRAM —
  the `Pipeline` does, so each stage stays small and independently testable.
- **Backends**: every stage switches on `ctx.backend`. `mock` is a CPU,
  model-free placeholder (used by all tests and the no-GPU demo). `default` uses
  the Apache/OpenRAIL models on a GPU. `flux-pano` / `large` are optional heavier
  or non-commercial backends. A stage must always provide `mock`.
- **Content-addressed cache** (`cache.py` + `Pipeline._run_stage`): a stage's key
  hashes `name+version`, the upstream payload (which itself carries its own key →
  chaining), and output-affecting params. Results are written to
  `world/cache/<stage>/<key>.json` (+ sibling artifact). Re-running skips
  completed stages — this is the crash/cutoff resume mechanism.
- **Sequential model loading** (`models/registry.py`): models load on first use
  and `unload_all()` runs between stages, so peak VRAM is one model at a time.
- **OOM → fix** (`profiling.oom_message`): CUDA OOM is caught in the pipeline and
  re-raised with concrete remedies (lower `--resolution`, `--offload`, `--backend`).
- **World** (`world/`): the on-disk unit. `world.json` (pydantic `WorldMeta`,
  format frozen at M4) + `splats.ply`/`.spz` + optional `mesh.glb` + `thumbnail` +
  `history/` + `cache/`. Reproducible from `seed` + `model_versions` + `params`.

## Splats

`splat/gaussians.py` holds a numpy `GaussianCloud` and the equirectangular RGB-D
lift (stage 4's core, CPU). `splat/ply_io.py` reads/writes the de-facto 3DGS PLY
encoding (SH-DC color, logit opacity, log scale, scalar-first quaternion) so
output renders directly in gsplat/Spark. The GPU `default` backend adds a gsplat
optimization pass on top of the unprojected cloud (M1).

## Why these boundaries

Keeping stages pure and file-based means each one runs alone from the CLI, caches
to disk, and is swappable (mock ↔ real ↔ NC backend) without touching the
orchestrator, the API, or the viewer. The viewer only ever reads finished world
files over HTTP, so it needs no GPU and no Python.
