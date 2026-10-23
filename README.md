# AdamWorld

A personal research prototype for turning images into explorable Gaussian-splat scenes.

AdamWorld combines image generation, depth estimation, spherical projection, a browser viewer, and portable scene files. The engineering focus is the pipeline and tooling: cached stages, a CLI and job API, scene metadata, a block-out editor, and PLY/SPZ/GLB export. It composes existing models; it does not train a world model from scratch.

**Status: experimental, pre-alpha.** The CPU demo is a synthetic test fixture, not AI-generated scenery. GPU experiments have produced viewable scenes, but panorama quality remains poor and several features are incomplete. This repository is a work in progress, not a production release.

**Code Apache-2.0, models under their own licenses.** No model weights are included. See [licenses](docs/LICENSES.md).

## What works and what remains

| Area | Current implementation | Limitation |
| --- | --- | --- |
| Image to scene | SDXL inpainting, relative depth, RGB-D unprojection | Seams, dark regions and distorted poles; depth is not calibrated metric depth |
| Splat refinement | Short gsplat photometric fit | Quality improvement is not established by the existing logs |
| Viewer | three.js + Spark, keyboard/mouse controls, touch look, WebXR entry | Requires WebGL2; hardware/browser support varies |
| Editing | Block-out editor and region deletion | Region regeneration is not implemented |
| Expansion | Generate, translate and concatenate a region | No demonstrated consistent boundary alignment |
| Persistence | Metadata, cached outputs and edit records | Not a complete undo/restore system or bit-exact regeneration guarantee |
| Export | Gaussian PLY/SPZ and coarse depth-grid GLB | GLB is an approximate surface, not a production mesh |
| GPU text/layout | SDXL text generation and ControlNet layout path | Experimental; dedicated T4 validation remains pending |
| Gallery | Static card grid and viewer links | Six mock fixtures only; no curated generated-world showcase yet |

The previous local v0.1.0 tag is a development checkpoint, not evidence of release readiness. See the [verification report](docs/VERIFICATION.md) and [progress](docs/PROGRESS.md).

## Try the CPU demo

Use Python 3.11 or 3.12 and Node.js 20+. From the repository root:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -e "./core[dev]"
adamworld generate --prompt "a forest clearing" --backend mock --resolution 256 --out worlds/demo
```

To view the synthetic scene, copy `worlds/demo/splats.spz` to `web/public/demo.spz` (create `public` if needed), then:

```bash
cd web
npm ci
npm run dev
```

Open `http://localhost:5173/?src=demo.spz`. The block-out editor is at `/editor.html`. In a second terminal at the repository root, run `adamworld serve --worlds worlds` for editor generation through the local API. Choose the mock backend for the CPU preview.

The API is a local development service, without authentication. Keep it bound to localhost; it accepts local file paths and is not designed for public hosting.

## Experimental GPU path

On a CUDA machine with Python 3.11 or 3.12:

```bash
python -m pip install -e "./core[gpu]"
adamworld generate --image photo.jpg --offload --resolution 768 --profile --out worlds/experiment
```

SDXL is the text-to-image default; Depth Anything V2 Small is the depth default. FLUX.1-schnell is an optional, unvalidated text backend. The advertised `flux-pano` and `large` alternatives in historical planning notes are not working quality upgrades. GPU dependency installation and native gsplat compilation are environment-dependent.

[Kaggle and Colab notebooks](notebooks/) are experimental runners. The [benchmark notes](docs/BENCHMARKS.md) distinguish historical measurements from verified quality claims. Allocator-reserved memory is not total GPU memory usage.

## Development

```bash
cd core
ruff check .
ruff format --check .
mypy adamworld
pytest -q
# In web/: npm run lint && npm run build
```

- [Architecture](docs/ARCHITECTURE.md)
- [Model and library licenses](docs/LICENSES.md)
- [Verification and limitations](docs/VERIFICATION.md)
- [Blender export](docs/export/blender.md) / [Unity export](docs/export/unity.md)
- [Contributing](CONTRIBUTING.md)

## Research direction

Improve panorama completion with controlled before/after evaluation; establish that hole filling actually executes and reduces disocclusions; measure fitting loss and held-out render quality; implement consistent expansion and region regeneration. These are open engineering and research tasks, not completed capabilities.
