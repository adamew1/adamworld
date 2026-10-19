# RESEARCH.md — model & library survey

> Working project name: **AdamWorld** (provisional — see PLAN.md §0).
> An open alternative to tools like Marble. Clean-room; composes existing open models.
> Research date: **2026-10-18**. All licenses verified against primary sources on this date; marked `(verify at adoption)` where the source was secondary. Re-check before any weights are downloaded.

Target runtime for every **default** path: one NVIDIA **T4 / P100, ~16 GB VRAM**, single GPU, 9–12 h sessions. Anything needing 24–80 GB goes behind `--backend large` and is documented as rented-GPU only.

Legend for license column:
- **Permissive** = Apache-2.0 / MIT / BSD — compatible with our Apache-2.0 project, no further questions.
- **OpenRAIL** = CreativeML OpenRAIL(++) — commercial use & redistribution permitted but with *behavioural use restrictions*. **Not pure Apache.** Allowed only as a flagged default after maintainer sign-off (§6 of the build prompt).
- **NC** = non-commercial (e.g. FLUX.1-dev license, CC-BY-NC). **Never a default.** Optional, pluggable backend only, clearly flagged.

---

## Summary of chosen defaults

| Stage | Default (license) | VRAM @ fp16/fp8 | Fallback | NC "quality" backend behind a flag |
|------|-------------------|-----------------|----------|-------------------------------------|
| 1 Text→image | **FLUX.1-schnell** (Apache-2.0) | ~10–13 GB fp8 + offload | SDXL-base (OpenRAIL++) / SD1.5 | — |
| 2 Image→360° pano | **SDXL equirectangular outpainting** (OpenRAIL++) ⚠ needs sign-off | ~9–12 GB fp16 | SD2-inpaint; our FLUX.1-schnell pano-LoRA (future) | **DiT360** / **Gimbal360** (FLUX.1-dev → NC) |
| 3 Depth | **Depth Anything V2 (L)** (Apache-2.0) via tangent crops | ~4 GB fp16 | Depth Anything V2 (B/S) | **UniDepth v2** (CC-BY-NC), **Depth Pro** (Apple-ASCL), **DA²/DAP** 360 (verify) |
| 4 Lift→splats | **gsplat** (Apache-2.0) + our RGBD-unproject init | ~3–8 GB (scene dependent) | pure point-cloud (no optim) | — |
| 5 Occlusion fill | **SDXL-inpaint** (OpenRAIL++) + gsplat refit (LucidDreamer algo, clean-room) | shares stage 2/4 | SD2-inpaint | FLUX.1-Fill-dev (NC) |
| 6 Expansion | our code on stages 2–5 (WonderWorld algo, clean-room) | shares above | — | — |
| 7 Mesh (opt) | **TSDF + marching cubes** (Open3D MIT, trimesh MIT) | CPU / <2 GB | Poisson recon | **SuGaR**/**2DGS** (Inria rasterizer → NC) |
| 8 Layout cond. | **ControlNet-depth (SDXL)** (verify OpenRAIL/Apache) | shares stage 2 | T2I-Adapter depth | — |
| Viewer | **Spark** (MIT) + three.js r180+ | CPU/WebGL2, no GPU | — | — |
| Splat file | **.spz** (MIT) primary, **.ply** fallback | — | — | — |

The only default that is **not** permissive is the panorama/inpainting generator (SDXL / OpenRAIL++). This is flagged as the single sign-off item because, as the survey below shows, every *high-quality* open panorama model today is a FLUX.1-dev derivative (NC), and the best commercial-usable option is SDXL-class (OpenRAIL). See §2 and PLAN.md §0.

---

## Stage 1 — Text → image (optional entry point)

| Candidate | License | VRAM (fp16 / fp8) | Quality | Maintenance |
|-----------|---------|-------------------|---------|-------------|
| **FLUX.1-schnell** | **Apache-2.0** ✅ | 12B params; ~24 GB fp16, **~10–13 GB fp8 + sequential CPU offload** → fits T4 | SOTA open, 1–4 step distilled | Black Forest Labs, very active |
| SDXL-base-1.0 | OpenRAIL++-M | ~7–10 GB fp16 | Strong, mature ecosystem (ControlNet/LoRA) | Stability, mature/stable |
| SD 1.5 | OpenRAIL-M | ~4 GB fp16 | Dated but tiny; good for CPU smoke tests | community |
| FLUX.1-dev | **NC** ❌ | like schnell | Higher fidelity than schnell | active |

**Default: FLUX.1-schnell.** Best quality among *permissive* (Apache-2.0) models, and the only top-tier T2I we can ship commercially without caveats. On T4 it runs with fp8 weights + `enable_sequential_cpu_offload` + VAE tiling. **Fallback: SDXL-base** (lighter, faster, but OpenRAIL) and **SD 1.5** for the CPU smoke test. Text entry can either go schnell→seed→outpaint (stage 2), or skip to a text-native 360 model; we default to the former for prompt/compositional control and because the image→360 path is needed anyway for image input.

## Stage 2 — Image → 360° panorama (the hard stage)

| Candidate | License | VRAM | Quality | Maintenance |
|-----------|---------|------|---------|-------------|
| **SDXL equirectangular outpainting** (our pipeline: place seed on equirect canvas → SDXL-inpaint with horizontal-wrap/circular padding, depth- & text-guided) | **OpenRAIL++** ⚠ | ~9–12 GB fp16 | Good; seam handling via circular padding/blending (per Diffusion360, PanoDiff) | base models mature |
| **DiT360** (Insta360, CVPR'26) | code MIT; **LoRA on FLUX.1-dev → NC** ❌ | FLUX-dev class (~16–24 GB, offload) | **Best** fidelity + seam handling (hybrid persp/pano training) | active (2026) |
| **Gimbal360** (Orange-3DV) | weights on HF; **LoRA on FLUX.1-Fill-dev → NC** ❌ | FLUX-Fill class | Strong single-image→360 (projects photo to equirect + inpaint) | 2025/26 |
| SDXL-360-diffusion (ProGamerGov) | states **MIT**, but is a finetune of OpenRAIL SDXL (merged-weight license is murky) *(verify at adoption)* | ~9–12 GB | Text→equirect, decent | hobby |
| PanoDiffusion / Diffusion360 / PanFusion | mixed (research) | SD2-class | reference techniques (circular padding, blending) | papers/repos |

**Default: SDXL equirectangular outpainting (OpenRAIL++), implemented by us.** Rationale: it is the only route that (a) handles the required **image→360** direction (not just text→360), (b) is commercially usable, and (c) reuses the same SDXL-inpaint checkpoint we need for stages 5 and 8, so we load one generator for three stages. We adopt the seam fixes the literature agrees on — **horizontal circular padding during denoising + circular VAE blending** (Diffusion360, PanoDiff). ⚠ OpenRAIL is not pure Apache → **sign-off required.**
**Optional high-quality NC backend: `--backend flux-pano`** = DiT360 or Gimbal360 (FLUX.1-dev). Clearly flagged NC, rented-GPU friendly.
**Future permissive path:** fine-tune a small panorama LoRA on **FLUX.1-schnell (Apache)** — allowed by §6 (optional LoRA), removes the OpenRAIL dependency. Tracked as post-v0.1 work.
**Also accept** a user-supplied equirectangular panorama or multi-image input, both of which skip stages 1–2.

## Stage 3 — Depth

| Candidate | License | VRAM | Quality | Maintenance |
|-----------|---------|------|---------|-------------|
| **Depth Anything V2 — Large** | **Apache-2.0** ✅ (S/B/L; **Giant is CC-BY-NC**) | ~3–4 GB fp16 | SOTA relative depth; sharp | ByteDance, very active |
| Depth Anything V2 — Base/Small | Apache-2.0 ✅ | <2 GB | good; Small is the CPU-smoke choice | active |
| Depth Anything V2 **metric** (indoor/outdoor fine-tunes) | Apache-2.0 ✅ *(verify per-checkpoint)* | ~4 GB | metric scale | active |
| UniDepth v2 | **CC-BY-NC-4.0** ❌ | ~6 GB | strong metric | ETH, active |
| Depth Pro (Apple) | Apple **ASCL** (research-ish) ❌ *(verify)* | ~8 GB | sharp metric, fast | Apple |
| DA² / DAP (Depth Any Panoramas) | *(verify)* | — | **360-native**, less distortion | 2025/26 |

**Default: Depth Anything V2 (Large), Apache-2.0.** It outputs *relative* (affine-invariant) depth, which is exactly what we need to unproject into a point cloud; absolute metric scale is recovered/assumed per-world and stored in `world.json`. Equirectangular distortion is handled by **tangent-plane projection**: slice the panorama into perspective tangent crops, run depth per crop, and fuse back to an equirect depth map (standard trick; CAPDepth/Depth-Anywhere). **Fallback: V2 Base/Small.** **NC backends** (UniDepth, Depth Pro, DA²/DAP 360-native) behind `--backend` for users who need true metric or best 360 geometry on a rented GPU.

## Stage 4 — Lift to 3D → Gaussian splats

| Candidate | License | Notes |
|-----------|---------|-------|
| **gsplat** (nerfstudio) | **Apache-2.0** ✅ | CUDA rasterizer + optimizer, PyTorch bindings; ~10% faster / 4× less memory than Inria ref; actively maintained (JMLR 2025). **This is our splat engine.** |
| Inria `diff-gaussian-rasterization` (graphdeco) | **research-only NC** ❌ | The original. Do **not** depend on it. (SuGaR/LucidDreamer/original-3DGS all use it → why we reimplement on gsplat.) |
| LucidDreamer (code, luciddreamer-cvlab) | **CC-BY-NC-SA-4.0** ❌ (NC + ShareAlike); also uses an Inria-derived rasterizer + ZoeDepth | Use the **paper only** (arXiv:2311.13384). Do NOT read/copy the repo code (ShareAlike would infect our Apache license). Our stage 5 is written from the paper on gsplat. |

**Default: gsplat.** We unproject RGB+depth to a colored point cloud (equirect → 3D rays × depth), initialize one Gaussian per point (color from pixel, scale from local point spacing, opacity≈1), then run a short gsplat optimization against the rendered input view(s) to clean up. Everything else in the category depends on the NC Inria kernel, so gsplat is both the best-licensed and best-maintained choice.

## Stage 5 — Fill occlusions

Algorithm (clean-room reimplementation of **LucidDreamer**'s "Dreaming + Alignment" and the inpaint-and-lift loop, on gsplat): render a novel view → detect holes/low-density regions → **SDXL-inpaint** the missing pixels (same checkpoint as stage 2) → estimate depth of the inpainted pixels (stage 3) → align depth to existing geometry at the seam → lift new pixels to Gaussians → add to the scene → brief gsplat refit. Iterate over a small orbit of novel cameras. License identical to stage 2 (OpenRAIL for SDXL-inpaint; FLUX.1-Fill-dev NC backend optional).

## Stage 6 — World expansion

Clean-room, built on stages 2–5 (informed by **WonderWorld** / **WonderJourney**, CVPR'25, as public references — we do not use their Inria-rasterizer code). User walks to an edge → we place a camera at the frontier, render the partial view, outpaint the unseen region (stage 5 machinery), estimate & **boundary-align depth** to the existing splats, lift, and merge into the global splat set as a new *tile* recorded in `world.json`. WonderWorld's "guided depth diffusion" (partial depth conditioning) is the technique to borrow for seamless joins.

## Stage 7 — Mesh extraction (optional)

| Candidate | License | Notes |
|-----------|---------|-------|
| **TSDF fusion + marching cubes** (Open3D MIT, trimesh MIT) | **Permissive** ✅ | Render depth from the fitted splats at several views → TSDF-fuse → marching cubes → decimate → **GLB**. Reuses depth we already compute; great for a coarse *collision/export* mesh. **Default.** |
| Poisson surface reconstruction (Open3D) | MIT ✅ | fallback from the same point cloud |
| SuGaR | code + **Inria rasterizer → NC** ❌ | higher-quality surface mesh; flagged optional backend only |
| 2DGS / GS2Mesh | research, rasterizer-dependent | note as future optional backends |

**Default: TSDF + marching cubes** → GLB via trimesh. Purpose is a coarse collision/export mesh, not a beauty mesh, so the lightweight permissive path wins.

## Stage 8 — Layout conditioning

Coarse 3D blocks from the browser block-out editor (boxes/planes/cylinders + text labels + style prompt) are rendered server-side to a **depth map** (and optionally a label/segmentation map). We condition the stage-2 generator with **ControlNet-depth for SDXL** so generation follows the blocked-out geometry, then run stages 2–5 as usual. ControlNet code (lllyasviel) is Apache-2.0; specific SDXL-depth control weights are OpenRAIL/Apache depending on publisher *(verify at adoption; prefer an Apache/`diffusers`-published checkpoint)*. Region edit/regenerate/delete operates on the splat set within a selected AABB and re-runs this path for just that region.

## Viewer — Spark (confirmed)

- **License: MIT** ✅ (World Labs). Compatible with Apache-2.0; permits commercial use. **Use it.**
- **Package:** `@sparkjsdev/spark` on npm (also CDN); site `sparkjs.dev`; repo `github.com/sparkjsdev/spark` (renamed from "Forge" in 2025 — avoid stale "forge" refs).
- **three.js:** requires **r180+**. Runs in any WebGL2 browser — desktop, mobile, headset (WebXR). **No GPU required**, satisfying "viewer must work on a laptop with no GPU."
- **API surface we rely on:** `SplatMesh` (loads a splat asset as a three.js object) and `PackedSplats` (in-memory splat buffers for our dynamic edits/expansion). Confirm exact signatures against the docs during M2 before coding.
- **Formats read:** PLY, **SPZ**, SPLAT, KSPLAT, SOGS — so our `.spz`/`.ply` worlds load directly, no custom loader.

## Splat file format

- **Primary `.spz`** — Niantic, **MIT**, ~10× smaller than PLY, read natively by Spark. We fit in gsplat, export PLY, and write SPZ. SPZ writers available: `3dgsconverter` (Python, MIT), PDAL `writers.spz`, Niantic C++ ref (MIT), Rust `spz` w/ Python bindings (MIT/Apache). Plan: write a small clean-room SPZ writer (format is fully documented) or vendor a permissive binding; **PLY is always written** as the lossless fallback.
- **`.ply`** — universal 3DGS interchange; what gsplat and every viewer/DCC tool accepts.

---

## Clean-room & license risk register

1. **Never depend on the Inria 3DGS rasterizer** (`diff-gaussian-rasterization`) or anything that imports it — research-only. gsplat replaces it everywhere.
2. **FLUX.1-dev / FLUX.1-Fill-dev are non-commercial.** Any model built on them (DiT360, Gimbal360) is NC → optional flagged backend, never default.
3. **SDXL / SD = OpenRAIL** — commercial OK but use-restricted; our one non-Apache default (panorama/inpaint). Needs sign-off; mitigation = future FLUX.1-schnell pano-LoRA.
4. **Depth Anything V2 Giant = CC-BY-NC**; we default to **Large (Apache)**.
5. **UniDepth, Depth Pro = NC/research** → optional metric backends only.
6. Record every adopted model+version+license in `docs/LICENSES.md`; re-verify the licenses marked *(verify at adoption)* before first download.

## Sources

- Spark: [github.com/sparkjsdev/spark](https://github.com/sparkjsdev/spark), [sparkjs.dev](https://sparkjs.dev/), [radiancefields.com/platforms/spark](https://radiancefields.com/platforms/spark)
- FLUX licensing: [FLUX.1-schnell is Apache-2.0](https://huggingface.co/unsloth/FLUX.1-schnell-FP8), [FLUX.1-Fill-dev non-commercial LICENSE](https://huggingface.co/black-forest-labs/FLUX.1-Fill-dev/blob/main/LICENSE.md), [flux licensing overview](https://deepwiki.com/black-forest-labs/flux/5.1-model-licenses-and-restrictions)
- Panorama: [Gimbal360 (HF)](https://huggingface.co/Orange-3DV-Team/Gimbal360), [DiT360 (GitHub)](https://github.com/Insta360-Research-Team/DiT360), [SDXL-360-diffusion (HF)](https://huggingface.co/ProGamerGov/sdxl-360-diffusion), [Diffusion360 paper](https://arxiv.org/pdf/2311.13141), [Survey on text→360 panorama](https://arxiv.org/html/2502.14799v1)
- Depth: [Depth Anything V2 license issue (S/B/L Apache, Giant NC)](https://github.com/DepthAnything/Depth-Anything-V2/issues/162), [UniDepth (GitHub, CC-BY-NC)](https://github.com/lpiccinelli-eth/unidepth), [Depth Pro](https://arxiv.org/html/2410.02073v1), [Depth Anywhere (360 distillation)](https://arxiv.org/html/2406.12849v2), [DAP (GitHub)](https://github.com/Insta360-Research-Team/DAP)
- Splats: [gsplat (Apache-2.0, JMLR'25)](https://github.com/nerfstudio-project/gsplat), [gsplat paper](https://arxiv.org/html/2409.06765v1)
- Scene gen references (**papers only** — repos are NC/unlicensed, not read for code): [LucidDreamer repo (CC-BY-NC-SA-4.0)](https://github.com/luciddreamer-cvlab/LucidDreamer), [LucidDreamer paper (arXiv:2311.13384)](https://arxiv.org/abs/2311.13384), [WonderWorld repo (no license)](https://github.com/KovenYu/WonderWorld), [WonderWorld paper (CVPR'25)](https://openaccess.thecvf.com/content/CVPR2025/papers/Yu_WonderWorld_Interactive_3D_Scene_Generation_from_a_Single_Image_CVPR_2025_paper.pdf)
- Mesh: [SuGaR (GitHub)](https://github.com/Anttwo/SuGaR), [2DGS paper](https://arxiv.org/pdf/2403.17888)
- SPZ format: [nianticlabs/spz (MIT)](https://github.com/nianticlabs/spz), [Scaniverse SPZ announcement](https://dev.scaniverse.com/news/spz-gaussian-splat-open-source-file-format), [3dgsconverter (Python)](https://github.com/francescofugazzi/3dgsconverter)

