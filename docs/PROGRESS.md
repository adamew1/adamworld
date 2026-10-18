# PROGRESS.md

## Status: M0 ✅ · M1 ✅ — building M2 (web viewer)

### Decisions (resolved)
- **Name:** AdamWorld (`adamworld` package/CLI, `@adamworld/web`).
- **Panorama default:** SDXL equirect outpaint (OpenRAIL++, flagged); FLUX (NC) behind `--backend flux-pano`; schnell pano-LoRA on roadmap.
- **Cadence:** unattended M0→M7, short summary per milestone.
- **Execution note:** this PC can't run the GPU models (4 GB GPU; no cp313 torch wheels). GPU backends are authored for the Kaggle/Colab T4 and validated there (M7). Everything else is built + verified locally; tests are GPU-free (mock).

### Done
- **Research + plan:** `RESEARCH.md`, `PLAN.md`, `ARCHITECTURE.md`, `LICENSES.md`.
- **M0 Scaffold** (commit): monorepo, pinned deps, ruff/mypy/eslint, CPU-only CI, pre-commit, justfile; stage protocol, typed payloads, content-hash cache, profiler, OOM→fix, World format, 3DGS PLY IO, equirect RGB-D lift (real, numpy), FastAPI + job queue, Typer CLI; web shell. Gates green; mock CLI → real 131k-splat world.
- **M1 Image→world (stages 2-4):**
  - `geometry.py` — spherical/equirect core: dir↔uv, bilinear equirect sampling (wrap), perspective/tangent rendering, cosine-feathered fusion, `place_image_front` seed projection, circular padding. **CPU, fully unit-tested (6 tests).**
  - Stage 2 panorama `default` = SDXL equirect outpaint (place seed on equirect canvas + inpaint mask + horizontal circular padding). Mock now uses the real `place_image_front` projection.
  - Stage 3 depth `default` = Depth Anything V2 over an 8-view horizon ring + up/down tangent crops, fused to an equirect depth map.
  - Stage 4 lift = real numpy unproject (M0); gsplat GPU refine hook present (optional, M-later).
  - Model registry loaders for SDXL-inpaint (diffusers) and Depth Anything V2 (transformers), lazy + offload/tiling.
  - Gates green: ruff ✓ mypy ✓ pytest **19** ✓ (incl. geometry + smoke e2e + API). CLI image→world verified.

### Next — M2 (web viewer)
- Load a generated world's `.ply`/`.spz` in the browser via **Spark** `SplatMesh`; WASD/mouse + touch nav; WebXR; loading progress. Verify by driving a headless browser (Playwright) to load a mock-generated world and screenshot. Confirm Spark's current npm version/API first.

### Known issues / notes
- Global `%TEMP%\pytest-of-*` is permission-locked on this PC; `conftest.py` redirects temp root to project-local `.pytest_tmp`.
- Pinned `click==8.1.7` (typer 0.12.5 vs click ≥8.2 secondary-flag bug).
- Depth fusion yields *relative* depth; metric scale is approximate and stored per-world (refine later).
- `.spz` writer + `mesh.glb` land in M6; PLY is current splat output.
