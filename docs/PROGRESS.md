# PROGRESS.md

## Status: M0 ✅ · M1 ✅ · M2 ✅ · M3a ✅ · M4 ✅ · M5 ✅ · M6 ✅ · M7 notebooks ✅ — remaining: M3b (occlusion), run on Kaggle

> Milestone order adjusted: did M6 (export) early because it's fully CPU-verifiable and completes the
> primary `.spz` format + Blender/Unity export. M3a (FLUX text→image) is done; M3b (occlusion fill),
> M4 (expansion), M5 (layout editor) remain. The real GPU stages await a Kaggle run (needs your token).

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

- **M2 Web viewer:** Spark **v2.3.1** (`@sparkjsdev/spark`, verified) + three r0.180. `web/src/main.ts` loads a world's `.spz`/`.ply` via `SplatMesh` + `SparkRenderer`; `controls.ts` = WASD/arrows + mouse/touch drag-look + Space/C + Shift-sprint; WebXR via `VRButton`; loading progress bar; `?world=<id>` (API) or `?src=<url>` (direct). **Verified in a headless browser (Playwright):** loaded a generated world's PLY, no load errors, splats render (see `adamworld-m2-viewer.png`). Viewer needs no GPU.

- **M3a Text input:** stage 1 default = FLUX.1-schnell (Apache), 4-step, bf16 + sequential offload (T4).
- **M7 notebooks:** `kaggle_generate.ipynb`, `colab_generate.ipynb`, and `run_on_kaggle.py` (headless: package `core/` → push GPU kernel → poll → pull `world.zip`).
- **M6 Export:** `splats.spz` (hand-written Niantic SPZ v2, **verified rendering in Spark**, ~15× smaller than PLY) + `splats.ply` + `mesh.glb` (equirect-depth environment mesh via a self-contained GLB writer). CLI `export --format ply|spz|glb` (glb builds from cache) + `mesh`. Guides: `docs/export/blender.md`, `unity.md`. Gates green (23 tests).

- **M4 Persistence + expansion:** `world.json` format **frozen** (format_version=1, load-time version guard); versioned `history/` (tested); stage 6 `expand` — generate an adjoining region (stages 1-4), translate to the frontier, merge into the splat set, record a `tile` + history edit. CLI `expand` + API `POST /worlds/{id}/expand`. Verified: world grows 4.6k→9.2k→13.8k splats across two expansions, 2 tiles, 2 edits.

- **M5 Layout editor:** `layout_raster.py` casts a ray per equirect pixel against unit box/plane/cylinder primitives (from the editor's column-major transforms) → equirect depth + label map (stage 8). `generate_from_layout` (mock) lifts that into a 3D block-out; `edit_region` deletes/regenerates splats in an AABB (delete is CPU). Browser `editor.html`/`editor.ts`: three.js + OrbitControls + TransformControls, add/move/rotate/scale box·plane·cylinder, labels, style prompt, "Generate world". CLI `generate --layout`, `edit --region --delete`; API `/edit` + layout in `POST /worlds`. **Verified end-to-end in a headless browser:** placed 2 blocks → Generate → backend built a layout world (kind=layout, spz/ply) → viewer rendered it. Gates green (29 tests).

### Next
- **Run on Kaggle** (needs your `~/.kaggle/kaggle.json`): `python notebooks/run_on_kaggle.py --mode image` → validate real SDXL pano + Depth Anything + lift on a T4, then `--mode text` for FLUX.
- **M3b Occlusion fill** (novel-view inpaint + refit; GPU, validated on Kaggle).
- Then tag **v0.1.0** (README GIFs + model/license table).

### Known issues / notes
- Global `%TEMP%\pytest-of-*` is permission-locked on this PC; `conftest.py` redirects temp root to project-local `.pytest_tmp`.
- Pinned `click==8.1.7` (typer 0.12.5 vs click ≥8.2 secondary-flag bug).
- Depth fusion yields *relative* depth; metric scale is approximate and stored per-world (refine later).
- `.spz` writer + `mesh.glb` land in M6; PLY is current splat output.
