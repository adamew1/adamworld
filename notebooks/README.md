# notebooks/

Run AdamWorld's **default** (GPU) pipeline on free hosted GPUs.

| File | Use |
|------|-----|
| `kaggle_generate.ipynb` | Interactive: open on Kaggle, attach the `adamworld-core` dataset, Run All. |
| `colab_generate.ipynb` | Interactive: open in Colab (T4 runtime), upload `core/` or set `GIT_URL`, Run All. |
| `run_on_kaggle.py` | **Headless**: packages `core/`, pushes a GPU kernel, polls, downloads `world.zip`. |
| `_build_notebooks.py` | Regenerates the two `.ipynb` files (keeps cell sources in one place). |

## Headless Kaggle run

```bash
pip install kaggle                       # once (already in the project venv)
# put your token at ~/.kaggle/kaggle.json  (Kaggle -> Settings -> API -> Create New Token)
python notebooks/run_on_kaggle.py --mode image   # SDXL pano + Depth Anything + lift
python notebooks/run_on_kaggle.py --mode text    # + FLUX.1-schnell (large weights download)
```

The stage cache lives under the world's `cache/`, so a cut-off session resumes
on re-run instead of restarting. Weights download on first use to the Kaggle
session; nothing is committed to the repo.
