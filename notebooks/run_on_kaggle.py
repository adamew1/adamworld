"""Run the real AdamWorld pipeline on a free Kaggle GPU, headless, end to end.

What it does:
  1. packages the repo's ``core/`` (+ a seed image) as a private Kaggle dataset,
  2. pushes a GPU+internet kernel that installs deps and runs ``default`` generation,
  3. polls until the kernel finishes, then downloads the output (world.zip, log).

Prerequisites:
  - ``pip install kaggle`` (done in the project venv)
  - a Kaggle API token at ``~/.kaggle/kaggle.json`` (Kaggle -> Settings -> API ->
    "Create New Token"). The username is read from that file.

Usage:
  python notebooks/run_on_kaggle.py --mode image   # SDXL pano + DepthAnything + lift (fast-ish)
  python notebooks/run_on_kaggle.py --mode text    # + FLUX.1-schnell (large download)
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
STAGE = REPO / ".kaggle_run"
DATASET_SLUG = "adamworld-core"
KERNEL_SLUG = "adamworld-generate"


def _seed_image(path: Path) -> None:
    import numpy as np
    from PIL import Image

    h, w = 512, 512
    y = np.linspace(0, 1, h)[:, None, None]
    sky = np.array([135, 180, 235]) * (1 - y) + np.array([230, 225, 210]) * y
    img = np.tile(sky, (1, w, 1))
    img[int(h * 0.6) :] = [80, 120, 70]  # ground
    Image.fromarray(img.astype("uint8")).save(path)


# Shared setup: robust source discovery (dataset layout varies / may be zipped),
# sys.path install (no build step), deps, and the common imports.
_PRELUDE = r"""
import os, glob, sys, subprocess, zipfile
# Keep model weights OUT of /kaggle/working so the kernel output (and its download)
# stays tiny — only world.zip / the log come back.
os.environ['ADAMWORLD_CACHE'] = '/tmp/.adamworld_cache'
os.environ['HF_HOME'] = '/tmp/hf'
os.environ['HUGGINGFACE_HUB_CACHE'] = '/tmp/hf'
for r in glob.glob('/kaggle/input/*'):
    print('input:', r, '->', os.listdir(r)[:8], flush=True)
for z in glob.glob('/kaggle/input/**/*.zip', recursive=True):
    zipfile.ZipFile(z).extractall('/kaggle/working/src')
SRC = None
for p in (glob.glob('/kaggle/input/**/pyproject.toml', recursive=True)
          + glob.glob('/kaggle/working/src/**/pyproject.toml', recursive=True)):
    SRC = os.path.dirname(p); break
assert SRC, 'could not locate adamworld core (no pyproject.toml under /kaggle/input)'
print('SRC =', SRC, flush=True)
sys.path.insert(0, SRC)
subprocess.run([sys.executable,'-m','pip','install','-q','pydantic>=2','platformdirs',
    'diffusers==0.31.0','transformers==4.46.2','accelerate==1.1.1','safetensors',
    'huggingface_hub'], check=True)
import torch
print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0), flush=True)
_seeds = (glob.glob('/kaggle/input/**/seed.png', recursive=True)
          + glob.glob('/kaggle/working/src/**/seed.png', recursive=True))
SEED_IMG = _seeds[0] if _seeds else None
from pathlib import Path
from adamworld.pipeline import Pipeline
from adamworld.profiling import Profiler
from adamworld.config import Settings
from adamworld.types import GenerationRequest
"""


def _run_source(mode: str) -> str:
    use_image = "True" if mode == "image" else "False"
    prompt = "'a quiet forest clearing with a mossy stone well'"
    tail = f"""
IMAGE = SEED_IMG if {use_image} else None
req = GenerationRequest(prompt={prompt}, image_paths=[Path(IMAGE)] if IMAGE else [],
                        resolution=768, backend='default', seed=0)
prof = Profiler()
world = Pipeline(settings=Settings(offload=True), profiler=prof, occlusion=False,
                 progress=lambda s,f: print(f'{{f*100:6.1f}}%  {{s}}', flush=True)).generate(
    req, Path('/kaggle/working/world'))
print(prof.table(), flush=True)
import shutil
from adamworld.splat import GaussianCloud
print('splats:', len(GaussianCloud.load_ply(world.splats_ply)), flush=True)
shutil.make_archive('/kaggle/working/world', 'zip', world.root)
print('OK wrote world.zip', flush=True)
"""
    return _PRELUDE + tail


def _bench_source() -> str:
    """Benchmark: image->world at 768 with offload OFF then ON; honest reserved
    VRAM per stage + device-used (~nvidia-smi) for cross-check."""
    tail = """
def _dev_used_mb():
    free, total = torch.cuda.mem_get_info()
    return (total - free) / (1024 * 1024)
for off in (True, False):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    print(f"\\n===== offload={off} =====", flush=True)
    prof = Profiler()
    Pipeline(settings=Settings(offload=off), profiler=prof, occlusion=False).generate(
        GenerationRequest(prompt='a quiet forest clearing',
                          image_paths=[Path(SEED_IMG)] if SEED_IMG else [],
                          resolution=768, backend='default', seed=0),
        Path(f'/tmp/bench_{off}'))  # /tmp -> not in kernel output, keeps the download tiny
    print(prof.table(), flush=True)
    print('device used after run (MB, ~nvidia-smi):', round(_dev_used_mb(), 1), flush=True)
print('OK bench done', flush=True)
"""
    return _PRELUDE + tail


def _full_source() -> str:
    """Comprehensive validation: image->world with the seamless pano (H5), gsplat
    fit (H7), and occlusion fill (H6) all on. Reports splat count before/after fit
    and the reserved-VRAM profile. Needs gsplat (installed here)."""
    tail = """
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gsplat'], check=True)
import numpy as np, glob, shutil
from adamworld.splat import GaussianCloud
from adamworld.imaging import load_rgb

req = GenerationRequest(prompt='a quiet forest clearing with a mossy stone well',
                        image_paths=[Path(SEED_IMG)] if SEED_IMG else [],
                        resolution=768, backend='default', seed=0)
prof = Profiler()
world = Pipeline(settings=Settings(offload=True), profiler=prof, occlusion=True,
                 progress=lambda s, f: print(f'{f*100:5.1f}%  {s}', flush=True)).generate(
    req, Path('/kaggle/working/world'))
print(prof.table(), flush=True)

pano = sorted(glob.glob('/kaggle/working/world/cache/panorama/*.png'))[0]
depth = sorted(glob.glob('/kaggle/working/world/cache/depth/*.npy'))[0]
before = GaussianCloud.from_rgbd_equirect(load_rgb(pano), np.load(depth), max_points=800000)
after = GaussianCloud.load_ply('/kaggle/working/world/splats.ply')
print(f'SPLATS unprojected(before fit): {len(before)}  ->  after fit+occlusion: {len(after)}', flush=True)
shutil.make_archive('/kaggle/working/world', 'zip', '/kaggle/working/world')
print('OK wrote world.zip', flush=True)
"""
    return _PRELUDE + tail


def _notebook(source: str) -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": source.strip("\n").splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["image", "text", "bench", "full"], default="image")
    ap.add_argument("--poll", type=int, default=30, help="status poll interval (s)")
    ap.add_argument("--skip-dataset", action="store_true", help="reuse the existing dataset")
    ap.add_argument("--no-wait", action="store_true", help="push the kernel and exit (poll later)")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username")
    assert user, "No Kaggle username — is ~/.kaggle/kaggle.json present?"
    print("Kaggle user:", user)

    # --- 1. dataset (core/ + seed image) ---
    if not args.skip_dataset:
        ds_dir = STAGE / "dataset"
        if ds_dir.exists():
            shutil.rmtree(ds_dir)
        shutil.copytree(REPO / "core", ds_dir / "core", ignore=shutil.ignore_patterns(
            "__pycache__", "*.pyc", ".pytest_tmp", "*.egg-info", "build", "dist"))
        _seed_image(ds_dir / "seed.png")
        (ds_dir / "dataset-metadata.json").write_text(json.dumps({
            "title": "adamworld-core",
            "id": f"{user}/{DATASET_SLUG}",
            "licenses": [{"name": "Apache 2.0"}],
        }, indent=2))

        try:
            api.dataset_status(f"{user}/{DATASET_SLUG}")
            print("updating existing dataset ...")
            api.dataset_create_version(str(ds_dir), version_notes="update", dir_mode="zip")
        except Exception:
            print("creating dataset ...")
            api.dataset_create_new(str(ds_dir), dir_mode="zip")

        # Wait until the dataset finishes processing so the kernel can mount it.
        for _ in range(20):
            time.sleep(args.poll)
            try:
                st = str(api.dataset_status(f"{user}/{DATASET_SLUG}"))
                print("dataset status:", st, flush=True)
                if "ready" in st.lower():
                    break
            except Exception as exc:
                print("dataset not ready yet:", exc, flush=True)
    else:
        print("skipping dataset upload (reusing existing)", flush=True)

    # --- 2. kernel ---
    k_dir = STAGE / "kernel"
    if k_dir.exists():
        shutil.rmtree(k_dir)
    k_dir.mkdir(parents=True)
    source = {
        "bench": _bench_source,
        "full": _full_source,
    }.get(args.mode, lambda: _run_source(args.mode))()
    (k_dir / "adamworld-generate.ipynb").write_text(json.dumps(_notebook(source), indent=1))
    (k_dir / "kernel-metadata.json").write_text(json.dumps({
        "id": f"{user}/{KERNEL_SLUG}",
        "title": "adamworld-generate",
        "code_file": "adamworld-generate.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [f"{user}/{DATASET_SLUG}"],
    }, indent=2))
    print("pushing kernel ...")
    api.kernels_push(str(k_dir))
    if args.no_wait:
        print("pushed; exiting (poll with --mode poll or kernels_status).", flush=True)
        return

    # --- 3. poll + fetch ---
    kref = f"{user}/{KERNEL_SLUG}"
    while True:
        time.sleep(args.poll)
        st = api.kernels_status(kref)
        status = st.get("status") if isinstance(st, dict) else getattr(st, "status", st)
        s = str(status).lower()
        print("status:", status, flush=True)
        if any(k in s for k in ("complete", "error", "cancel")):
            break
    out = STAGE / "output"
    out.mkdir(exist_ok=True)
    try:
        api.kernels_output(kref, str(out))
        print("downloaded output to", out)
    except Exception as exc:  # e.g. Windows cp1252 log-write bug in the kaggle client
        print("kernels_output partial (continuing):", exc)
    print("open the .log to see the run; world.zip is the generated world.")


if __name__ == "__main__":
    main()
