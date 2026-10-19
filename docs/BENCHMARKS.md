# BENCHMARKS.md

Real numbers from the **default (GPU) backend** on a **free Kaggle NVIDIA Tesla T4**
(16 GB), via `notebooks/run_on_kaggle.py --mode full`. Image → 360° panorama (SDXL)
→ depth (Depth Anything V2 **Small**, Apache) → lift **+ gsplat fit** → occlusion.
Sequential CPU offload **on**. VRAM is **reserved** (allocator high-water), reset per
stage — the honest footprint.

Device: **Tesla T4** · backend: **default** · resolution: **768** · 294,911 splats

```
stage       time(s)   peak VRAM MB (reserved)
--------------------------------------------
panorama     200.65          634.0    # SDXL equirect outpaint + seam/pole passes
depth          2.65          202.0    # Depth Anything V2 Small, 10 tangent crops
lift         330.75          152.0    # numpy unproject + gsplat fit (300 iters)
occlusion      3.60          106.0    # novel-view inpaint loop (gsplat render)
--------------------------------------------
TOTAL        537.64          634.0
```

Notes:
- **Reserved peak ~634 MB** with sequential offload — SDXL weights live on CPU and
  stream to the GPU layer-by-layer, so the resident footprint is genuinely small.
  With `offload=False` the SDXL stage reserves ~5–7 GB (still within a T4); run
  `run_on_kaggle.py --mode bench` for the offload-off comparison.
- **The gsplat fit dominates wall time** (300 iters ≈ 330 s). It optimizes the
  Gaussian field but, at the default prune threshold, barely changes the count
  (294,912 → 294,911). Fewer iters / denser pruning are tunable; see `splat/fit.py`.
- **Occlusion** runs in seconds here because the (poor) panorama leaves few
  detected holes; on richer scenes it does more work.
- An earlier image→world run **without** fit/seam passes was ~132 s total — the
  extra time here is the fit + seam/pole passes.

> Quality caveat: the SDXL equirect panorama is **experimental** and low quality
> (SDXL is not pano-aware). See `docs/RESEARCH.md` — real 360 fidelity needs the
> FLUX-based `--backend flux-pano` (non-commercial) or a FLUX.1-schnell pano LoRA.
