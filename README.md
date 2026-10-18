# AdamWorld

**Turn a text prompt or an image into a persistent, explorable, editable 3D world** — walk through it in your browser, edit it, and export it to Blender or Unity.

AdamWorld is a free and open alternative to tools like Marble. It is an independent, clean-room project that composes open-source models and open model weights into a reproducible pipeline. It is designed to run its default pipeline on a **single free-tier GPU (one 16 GB T4 / P100 on Kaggle or Colab)**.

> **Status: pre-alpha, under active construction.** Following milestones M0→M7 (see `docs/PLAN.md`). This README will carry GIFs, a quickstart, and the full model table at the v0.1.0 release (M7).

## What it does

```
prompt / image / 360° panorama / block-out layout
        │
        ▼  (pipeline of cached, swappable stages)
  text→image → image→360° → depth → lift to 3D Gaussian splats
             → fill occlusions → expand at edges → (optional) mesh
        │
        ▼
   world/  (splats + metadata + versioned history)
        │
        ├── explore & edit in the browser (three.js + Spark, no GPU needed)
        └── export to PLY / SPZ / GLB → Blender / Unity
```

## Design principles

- **Clean-room & license-clean.** Only public papers, open code, and open weights. Default pipeline is Apache-2.0-compatible; any non-commercial model is an optional, clearly-flagged backend. See `docs/LICENSES.md`.
- **Fits free hardware.** Every default stage runs within ~16 GB VRAM via fp16/fp8, CPU offload, tiling, and sequential model loading. Bigger models live behind `--backend large`.
- **Reproducible.** Worlds carry their seed, parameters, and model versions in `world.json`, so a world can be regenerated from its metadata.
- **Resumable.** Every stage caches its output by content hash; a crashed or cut-off run resumes instead of restarting.
- **Viewer needs no GPU.** Only generation needs a GPU; exploring a finished world runs on any laptop.

## Quickstart (dev, CPU)

```bash
# Python core (CPU — mock backends, for development & tests)
cd core
python -m venv ../.venv && . ../.venv/Scripts/activate   # Windows: ..\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
pytest -q                          # runs the CPU smoke test
adamworld generate --prompt "a quiet forest clearing" --backend mock --out ../worlds/demo
adamworld serve                    # FastAPI backend on :8000

# Web viewer
cd ../web
npm install
npm run dev                        # viewer on :5173
```

GPU generation (real models) needs the `[gpu]` extra and a CUDA GPU:

```bash
pip install -e ".[gpu]"
adamworld generate --prompt "a quiet forest clearing" --out worlds/forest
```

## Hardware

| Path | GPU | Notes |
|------|-----|-------|
| Default pipeline | 1× **T4 / P100, ~16 GB** | fp8/fp16 + CPU offload + tiling; free Kaggle/Colab tier |
| `--backend large` | 24–80 GB | rented GPU; higher-fidelity backends |
| Viewer / editor | **none** | runs in any WebGL2 browser |

## Documentation

- `docs/PLAN.md` — architecture, modules, file formats, milestones
- `docs/RESEARCH.md` — model/library survey & the reasons behind each default
- `docs/LICENSES.md` — every model & dependency with its license
- `docs/PROGRESS.md` — what's done, what's next, known issues
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`

## License

Apache-2.0 (`LICENSE`). "Marble" and "World Labs" are trademarks of their respective owners; AdamWorld is not affiliated with or derived from them.
