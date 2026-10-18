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


def _run_source(mode: str) -> str:
    image = "'/kaggle/input/adamworld-core/seed.png'" if mode == "image" else "None"
    prompt = "'a quiet forest clearing with a mossy stone well'"
    return f"""
import os, glob
os.environ['ADAMWORLD_CACHE'] = '/kaggle/working/.adamworld_cache'
import subprocess, sys
SRC = next(iter(glob.glob('/kaggle/input/*/core')))
subprocess.run([sys.executable,'-m','pip','install','-q',SRC], check=True)
subprocess.run([sys.executable,'-m','pip','install','-q','diffusers==0.31.0',
    'transformers==4.46.2','accelerate==1.1.1','safetensors','huggingface_hub',
    'opencv-python-headless','gsplat'], check=True)

import torch
print('CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))

from pathlib import Path
from adamworld.pipeline import Pipeline
from adamworld.profiling import Profiler
from adamworld.config import Settings
from adamworld.types import GenerationRequest

req = GenerationRequest(prompt={prompt}, image_paths=[Path({image})] if {image} else [],
                        resolution=768, backend='default', seed=0)
prof = Profiler()
world = Pipeline(settings=Settings(offload=True), profiler=prof,
                 progress=lambda s,f: print(f'{{f*100:6.1f}}%  {{s}}', flush=True)).generate(
    req, Path('/kaggle/working/world'))
print(prof.table())
import shutil
from adamworld.splat import GaussianCloud
print('splats:', len(GaussianCloud.load_ply(world.splats_ply)))
shutil.make_archive('/kaggle/working/world', 'zip', world.root)
print('OK wrote world.zip')
"""


def _notebook(mode: str) -> dict:
    return {
        "cells": [
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": _run_source(mode).strip("\n").splitlines(keepends=True),
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
    ap.add_argument("--mode", choices=["image", "text"], default="image")
    ap.add_argument("--poll", type=int, default=30, help="status poll interval (s)")
    args = ap.parse_args()

    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    user = api.get_config_value("username")
    assert user, "No Kaggle username — is ~/.kaggle/kaggle.json present?"
    print("Kaggle user:", user)

    # --- 1. dataset (core/ + seed image) ---
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
    time.sleep(args.poll)

    # --- 2. kernel ---
    k_dir = STAGE / "kernel"
    if k_dir.exists():
        shutil.rmtree(k_dir)
    k_dir.mkdir(parents=True)
    (k_dir / "adamworld-generate.ipynb").write_text(json.dumps(_notebook(args.mode), indent=1))
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

    # --- 3. poll + fetch ---
    kref = f"{user}/{KERNEL_SLUG}"
    while True:
        time.sleep(args.poll)
        st = api.kernels_status(kref)
        status = st.get("status") if isinstance(st, dict) else getattr(st, "status", st)
        print("status:", status, flush=True)
        if str(status).lower() in ("complete", "error", "cancelacknowledged"):
            break
    out = STAGE / "output"
    out.mkdir(exist_ok=True)
    api.kernels_output(kref, str(out))
    print("downloaded output to", out)
    print("open the .log to see the run; world.zip is the generated world.")


if __name__ == "__main__":
    main()
