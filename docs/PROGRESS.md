# PROGRESS.md

## Status: M0 (Scaffold) complete ✅ — building M1

### Decisions (resolved)
- **Project name:** AdamWorld (`adamworld` package / CLI, `@adamworld/web`).
- **Panorama default:** SDXL equirectangular outpainting (OpenRAIL++, commercial-usable)
  as the flagged default; FLUX (NC) behind `--backend flux-pano`; FLUX.1-schnell
  pano-LoRA on the roadmap as the eventual pure-Apache default.
- **Cadence:** unattended through M0→M7, 5-line summary per milestone.

### Done
- **Section 1 research + plan:** `docs/RESEARCH.md`, `PLAN.md`, `LICENSES.md`, `ARCHITECTURE.md`.
- **M0 Scaffold:**
  - Monorepo: `core/` (Python), `web/` (TS), `docs/`, `.github/workflows/ci.yml`, `justfile`, pre-commit.
  - Core: pinned deps + `[gpu]`/`[dev]` extras; stage protocol, typed payloads, content-hash cache,
    profiler (time + peak VRAM), OOM→fix, `World` format, 3DGS PLY IO, equirect RGB-D lift (real, numpy),
    FastAPI + in-process job queue, Typer CLI.
  - All 8 stages present; mock backends implemented for stages 1-4; real/NC backends stubbed to their milestones.
  - Web: Vite+TS+three.js placeholder viewer (Spark viewer lands M2); eslint + build green.
  - **Quality gates green:** `ruff` ✓, `mypy` ✓, `pytest` 13 passed (incl. CPU smoke e2e + API), web `lint`+`build` ✓.
  - CLI verified end-to-end: `generate --backend mock` → real `world/` (131k splats, valid world.json, thumbnail, resumable cache).
  - `docs/BENCHMARKS.md` generated via `scripts/benchmark.py`.

### Next — M1 (Image → world, real stages 2-4)
- Stage 2: SDXL equirect outpainting (place seed image on equirect canvas, circular-padding inpaint).
- Stage 3: Depth Anything V2 over tangent-plane crops, fused to equirect depth.
- Stage 4: gsplat optimization pass on the unprojected cloud (GPU `default`).
- GPU code is authored to run on Kaggle/Colab T4 (M7); CPU keeps mock + real numpy lift. Tests stay GPU-free.

### Known issues / notes
- This machine's global `%TEMP%\pytest-of-*` is permission-locked; `conftest.py` redirects the temp root to a
  project-local `.pytest_tmp` so `pytest -q` works here and anywhere.
- Pinned `click==8.1.7` (typer 0.12.5 is incompatible with click ≥8.2 secondary-flag handling).
- `.spz` writer + `mesh.glb` arrive in M6; PLY is the current splat output.
- Several `LICENSES.md` rows marked *(verify at adoption)* — re-check before first weight download (M1/M7).
