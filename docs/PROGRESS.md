# PROGRESS.md

## Status: v0.1.0 — M0–M7 ✅ and **default pipeline validated on a real free Kaggle T4**.

Validated (image→world, default backend, Tesla T4): SDXL pano + Depth Anything + lift →
294,912 splats in 132 s, peak VRAM 885 MB (fits a T4 with headroom); loaded + navigated in
Spark; SPZ ~9× smaller than PLY. See `docs/BENCHMARKS.md`. Known quality limit: plain SDXL
isn't equirect-aware (seams/soft poles) — higher fidelity via `--backend flux-pano` (NC) or
the planned FLUX.1-schnell pano LoRA. Still optional: `--mode text` (FLUX) + occlusion (stage 5)
GPU runs; README GIFs.

> Milestone order adjusted: did M6 (export) early because it's fully CPU-verifiable and completes the
> primary `.spz` format + Blender/Unity export. M3a (FLUX text→image) is done; M3b (occlusion fill),
> M4 (expansion), M5 (layout editor) remain. The real GPU stages await a Kaggle run (needs your token).

## Pre-release hardening (H1–H8, in order)
- **H1 License audit ✅** — Verified on model cards (2026-10-19): DAv2 **Small = apache-2.0**, **Base/Large/Giant = cc-by-nc-4.0** (earlier draft was wrong). Default is now **Small**; NC sizes gated behind `--allow-noncommercial` (`Settings.allow_noncommercial`), enforced before load (tested). SDXL + inpaint = OpenRAIL++-M; ControlNet-depth-sdxl = openrail++; FLUX.1-schnell = apache-2.0 — all recorded in `docs/LICENSES.md`. README now states "code Apache-2.0, models under their own licenses". 36 tests green.
- **H2 Clean-room audit ✅** — LucidDreamer (luciddreamer-cvlab) is **CC-BY-NC-SA-4.0** (NC + ShareAlike), not MIT as an earlier draft said; WonderWorld has **no license**. Audited `stages/occlusion.py` + `Pipeline.expand`: written from the *papers*, single-file, own `geometry`/`GaussianCloud`/loop/constants; use **gsplat** + **Depth Anything** (not their Inria-derived rasterizer or ZoeDepth); no shared identifiers or file structure → no copied code, nothing to rewrite. Tightened docstrings to name the papers + "no repo code used"; fixed the wrong MIT license in RESEARCH.md and LICENSES.md.
- **H3 Secrets scan ✅** — gitleaks v8.30.1 over **full history (14 commits): no leaks**. Targeted pickaxe searches confirm the Kaggle token (`KGAT_…`), its fragment, `access_token`, `kaggle.json`, and `.env` were **never committed** (token lives only in `~/.kaggle/`, outside the repo). `.gitignore` now excludes `.env*`, `kaggle.json`, `access_token`, `*.pem/*.key/id_rsa*`; added a `gitleaks` CI job (full-history scan on every push/PR).
- **H4 ✅ (T4-validated)** — profiler reports **reserved** VRAM with per-stage reset. `--mode full` on a T4: panorama 634 MB, depth 202, lift 152, occlusion 106 (reserved, offload on); low because offload streams SDXL from CPU — honest. `docs/BENCHMARKS.md` updated.
- **H5 ⚠ validated, DOES NOT fix it** — circular-conv + latent-roll + pole-inpaint ran on a T4; the pano is still low quality (black regions, dome) and the **pole inpaint made it worse (circular artifacts) → reverted**. Root cause is architectural: SDXL isn't equirect-aware. Circular-conv kept (harmless); SDXL pano marked **experimental**; real fix = flux-pano / pano LoRA.
- **H6 partial** — **SDXL is the text→image default**, FLUX opt-in ✅. **Occlusion loop T4-validated ✅** (ran, gsplat built). ControlNet-depth loader added; *layout ControlNet path + text/ControlNet T4 validation still pending.*
- **H7 ✅ (runs on T4)** — `splat/fit.py` gsplat optimization ran (300 iters ≈ 330 s); count **294,912 → 294,911** (fit optimizes params; pruning minimal at default threshold — tunable). Quality gain hard to judge on the poor input pano (coupled to H5).
- **H8 scaffold ✅, deploy pending** — `web/gallery.html` + `src/gallery.ts`: a card grid reading `gallery/worlds.json`, each card a walk-through link into the Spark viewer. `core/scripts/make_gallery.py` generates N worlds + the manifest (6 placeholder mock worlds in; `--backend default` for real ones). Vite `base:"./"` so it works under a Pages subpath; `scripts/deploy_gallery.ps1` pushes `web/dist` to `gh-pages`. **Verified rendering in a headless browser.** Pending: the user's repo URL to push + enable Pages; GIFs (need good worlds, which are gated on the pano fix).

> **Honest takeaway from the T4 validation:** the plumbing (profiler, occlusion, fit, end-to-end default run) works on free hardware; the **panorama quality is the real gap**, and it's architectural (SDXL ≠ pano-aware). The credible quality path is a FLUX-based pano model (NC) or a trained FLUX.1-schnell pano LoRA — both already on the roadmap.

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
