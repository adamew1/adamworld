# LICENSES.md — dependency & model license register

Project license: **Apache-2.0**. Rule: a model may be a **default** only if its license is permissive (Apache/MIT/BSD). OpenRAIL defaults require maintainer sign-off; non-commercial (NC) licenses are optional flagged backends only. Re-verify every row marked *(verify)* before downloading weights.

## Models

| Model | Role | License | Class | Status |
|-------|------|---------|-------|--------|
| FLUX.1-schnell | Stage 1 default | Apache-2.0 | Permissive | ✅ default |
| SDXL-base-1.0 + SDXL-inpaint | Stages 2/5/8 default | CreativeML OpenRAIL++-M | OpenRAIL (commercial-ok, use-restricted) | ⚠ default **pending sign-off** |
| SD 1.5 | CPU smoke / fallback | CreativeML OpenRAIL-M | OpenRAIL | fallback/test only |
| Depth Anything V2 — Large/Base/Small | Stage 3 default | Apache-2.0 | Permissive | ✅ default |
| Depth Anything V2 — Giant | Stage 3 | CC-BY-NC-4.0 | NC | ❌ not used |
| Depth Anything V2 metric fine-tunes | Stage 3 metric | Apache-2.0 *(verify per checkpoint)* | Permissive | optional |
| gsplat | Stage 4/5/7 splat engine | Apache-2.0 | Permissive | ✅ core |
| Inria diff-gaussian-rasterization | — | research-only | NC | ❌ forbidden dependency |
| ControlNet (code, lllyasviel) | Stage 8 | Apache-2.0 | Permissive | ✅ |
| ControlNet-depth-SDXL weights | Stage 8 | OpenRAIL/Apache *(verify, prefer Apache)* | — | pending |
| DiT360 (LoRA on FLUX.1-dev) | Stage 2 NC backend | code MIT; weights FLUX.1-dev (NC) + Insta360 license | NC | optional `--backend flux-pano` |
| Gimbal360 (LoRA on FLUX.1-Fill-dev) | Stage 2 NC backend | FLUX.1-Fill-dev (NC) | NC | optional backend |
| FLUX.1-dev / FLUX.1-Fill-dev | — | FLUX.1 [dev] Non-Commercial | NC | backend only |
| UniDepth v2 | Stage 3 metric backend | CC-BY-NC-4.0 | NC | optional backend |
| Depth Pro (Apple) | Stage 3 metric backend | Apple ASCL *(verify)* | NC/research | optional backend |
| DA² / DAP (360 depth) | Stage 3 360 backend | *(verify)* | — | optional, pending |
| SuGaR / 2DGS | Stage 7 mesh backend | depends on Inria rasterizer | NC | optional backend, flagged |

## Libraries / tooling

| Component | License | Notes |
|-----------|---------|-------|
| Spark (`@sparkjsdev/spark`) | MIT | viewer; three.js r180+ |
| three.js | MIT | |
| Niantic SPZ format + ref impl | MIT | `.spz` IO |
| 3dgsconverter (PLY↔SPZ) | MIT *(verify)* | optional SPZ path |
| Open3D | MIT | TSDF/marching cubes (stage 7) |
| trimesh | MIT | GLB export |
| diffusers / transformers / accelerate | Apache-2.0 | model runtime |
| PyTorch | BSD-3 | |
| FastAPI / Starlette / pydantic | MIT / BSD | backend |
| Typer / Click | MIT / BSD | CLI |
| Vite | MIT | web build |

## Open questions to resolve at adoption
- SDXL-as-default sign-off (OpenRAIL, not Apache).
- Exact ControlNet-depth-SDXL checkpoint license — prefer an Apache/`diffusers` one.
- Per-checkpoint license of Depth Anything V2 *metric* fine-tunes.
- DA²/DAP (360 depth) licenses if we adopt a 360-native depth backend.
- SDXL-360-diffusion claims MIT but is an SDXL finetune (merged-weight license murky) — avoid as default.

