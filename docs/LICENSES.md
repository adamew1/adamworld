# LICENSES.md — dependency & model license register

**Code: Apache-2.0. Models are downloaded at runtime and remain under their own
licenses — AdamWorld does not redistribute weights.** The default pipeline uses
only Apache-2.0 and OpenRAIL(++) models; every non-commercial model is gated
behind `--allow-noncommercial` (`Settings.allow_noncommercial`) and never runs by
default.

License classes: **Permissive** = Apache/MIT/BSD. **OpenRAIL** = CreativeML
OpenRAIL(++), commercial-ok but use-restricted (our one non-permissive default,
signed off). **NC** = non-commercial — opt-in only.

## Models the pipeline can download — exact licenses (verified 2026-10-19)

| Model (HF repo) | Role | License (as on the model card) | Default? |
|-----------------|------|--------------------------------|----------|
| `black-forest-labs/FLUX.1-schnell` | stage 1 text→image | **apache-2.0** ✅ | optional text backend |
| `stabilityai/stable-diffusion-xl-base-1.0` | SDXL base | **CreativeML OpenRAIL++-M** ⚠ | default text backend |
| `diffusers/stable-diffusion-xl-1.0-inpainting-0.1` | stages 2/5/8 | **CreativeML OpenRAIL++-M** ⚠ | **default** (pano/inpaint) |
| `depth-anything/Depth-Anything-V2-Small-hf` | stage 3 depth | **apache-2.0** ✅ | **default** |
| `depth-anything/Depth-Anything-V2-Base-hf` | stage 3 (opt-in) | **cc-by-nc-4.0** ❌ | `--allow-noncommercial` |
| `depth-anything/Depth-Anything-V2-Large-hf` | stage 3 (opt-in) | **cc-by-nc-4.0** ❌ | `--allow-noncommercial` |
| `depth-anything/Depth-Anything-V2-Giant-hf` | stage 3 (opt-in) | **cc-by-nc-4.0** ❌ | `--allow-noncommercial` |
| `diffusers/controlnet-depth-sdxl-1.0` | stage 8 layout | **openrail++** ⚠ | default (layout) |

> Correction (2026-10-19): an earlier draft wrongly listed Depth Anything V2
> Base/Large as Apache. Only **Small** is Apache-2.0; Base/Large/Giant are
> **CC-BY-NC-4.0**. The default is now Small; the others are gated.

## Research candidates (not implemented backends)

| Model | Role | License |
|-------|------|---------|
| Depth Anything V2 Base/Large/Giant | higher-quality depth | CC-BY-NC-4.0 |
| `black-forest-labs/FLUX.1-dev`, `FLUX.1-Fill-dev` | `--backend flux-pano` base | FLUX.1 [dev] Non-Commercial |
| DiT360 / Gimbal360 LoRA | best 360 panorama | FLUX.1-dev (NC) + Insta360 license |
| UniDepth v2 | metric depth | CC-BY-NC-4.0 |
| Apple Depth Pro | metric depth | Apple ASCL (research) |

## LoRA adapters

**None are currently trained, bundled, or downloaded.** Any future panorama LoRA requires a separate base-model and training-data license review before a license can be selected; if added it will be listed
here with its training-data provenance.

## Libraries / tooling

| Component | License |
|-----------|---------|
| Spark (`@sparkjsdev/spark`), three.js, Niantic SPZ format | MIT |
| gsplat | Apache-2.0 |
| diffusers / transformers / accelerate / huggingface_hub | Apache-2.0 |
| PyTorch | BSD-3 |
| numpy, pillow, FastAPI, pydantic, Typer, Vite | BSD / MIT |
| Open3D, trimesh (optional mesh backends) | MIT |

## Clean-room notes

- The original Inria 3DGS rasterizer (`diff-gaussian-rasterization`) is research-only
  and is **never** a dependency; all rasterization/fitting uses **gsplat** (Apache-2.0).
- **LucidDreamer** (luciddreamer-cvlab) is **CC-BY-NC-SA-4.0** and **WonderWorld**
  publishes **no license** (all rights reserved). We used only their *papers*
  (methods) — development records state that no repository code was copied; our stage 5 (occlusion) and
  stage 6 (expansion) are original and run on gsplat/Depth Anything, not their
  Inria-derived rasterizers or ZoeDepth. See the H2 audit in `docs/PROGRESS.md`.


## Publication audit (2026-10-23)

Default code selects SDXL text and Depth Anything V2 Small. License source links:
- [Depth Anything V2 Small model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf): Apache-2.0; relative depth, not calibrated metric depth.
- [SDXL base license](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/blob/main/LICENSE.md): CreativeML Open RAIL++-M; use restrictions remain applicable.
- [SDXL inpaint model card](https://huggingface.co/diffusers/stable-diffusion-xl-1.0-inpainting-0.1): CreativeML Open RAIL++-M.
- [ControlNet depth SDXL model card](https://huggingface.co/diffusers/controlnet-depth-sdxl-1.0): openrail++; experimental layout backend.
- [FLUX.1-schnell model card](https://huggingface.co/black-forest-labs/FLUX.1-schnell): historical record lists Apache-2.0; live retrieval required authentication during this audit, so it was not independently reverified today.

Earlier experiments downloaded Depth Anything V2 Large under CC-BY-NC-4.0. Their outputs are not used as the public gallery. The current small default does not retroactively change those experiments. No LoRA is implemented or downloaded by the current code. The Giant repository identifier in the registry is not a verified available checkpoint. Research candidates above must not be interpreted as working optional backends. OpenRAIL model terms are separate from the Apache code license.
