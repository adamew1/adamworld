# BENCHMARKS.md

Real numbers from the **default (GPU) backend** on a **free Kaggle NVIDIA Tesla T4**
(16 GB), via `notebooks/run_on_kaggle.py --mode image`. Input: one seed image →
360° panorama → depth → lift. Sequential CPU offload on; occlusion (stage 5)
disabled for this isolation run.

Device: **Tesla T4** · backend: **default** · resolution: **768** · 294,912 splats

```
stage      time(s)   peak VRAM(MB)
----------------------------------
panorama    120.54          470.9    # SDXL equirectangular outpaint (OpenRAIL)
depth        11.26          884.8    # Depth Anything V2 (Large), 10 tangent crops fused
lift          0.12            9.1    # numpy RGB-D -> Gaussian unprojection
----------------------------------
TOTAL       131.93          884.8
```

Takeaways:
- **Fits a T4 with huge headroom** — peak ~885 MB with sequential offload, so higher
  resolution or disabling offload is feasible.
- **~2.2 min end to end**; the SDXL panorama dominates (one inpaint pass).
- **SPZ is ~9× smaller** than PLY for this world (2.2 MB vs 20 MB).

Quality note: plain SDXL inpaint is **not** equirectangular-aware, so the 360 has
seams/soft poles. Higher-fidelity panoramas use the FLUX-based `--backend flux-pano`
(non-commercial) or the planned FLUX.1-schnell panorama LoRA (Apache). See
`docs/RESEARCH.md` and `docs/LICENSES.md`.

> Re-run `python notebooks/run_on_kaggle.py --mode text` for the FLUX.1-schnell path,
> or drop `occlusion=False` to include stage 5 (needs gsplat).
