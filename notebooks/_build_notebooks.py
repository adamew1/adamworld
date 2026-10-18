"""Generate the Kaggle and Colab notebooks as valid .ipynb JSON.

Run:  python notebooks/_build_notebooks.py
Keeping the cell sources here (plain Python strings) avoids hand-editing JSON and
keeps the two notebooks in sync.
"""

from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def notebook(cells: list[dict]) -> dict:
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


INTRO_KAGGLE = """
# AdamWorld — generate a 3D world on a free Kaggle GPU

Turns a prompt or image into a Gaussian-splat world you can explore in the browser.
Runs the **default** pipeline on a single **T4 / P100 (~16 GB)**.

**Setup (one time):**
1. Enable the GPU: *Settings → Accelerator → GPU T4 x2* (one is used) or P100.
2. Enable internet: *Settings → Internet → On* (to download model weights).
3. Attach the `adamworld-core` dataset (the repo's `core/` folder) via *Add Input*,
   or run the fallback install cell.

The pipeline caches each stage by content hash under `/kaggle/working/world/cache`,
so if a session is cut off you can **Save & Run All** again and it resumes instead
of restarting.
"""

INSTALL_KAGGLE = """
import os, glob, subprocess, sys

# AdamWorld source: attached as a Kaggle dataset (preferred) or cloned if you set GIT_URL.
SRC = next(iter(glob.glob("/kaggle/input/*/core")), None)
GIT_URL = ""  # optional: "https://github.com/<you>/adamworld.git"

if SRC is None and GIT_URL:
    subprocess.run(["git", "clone", "--depth", "1", GIT_URL, "/kaggle/working/adamworld"], check=True)
    SRC = "/kaggle/working/adamworld/core"
assert SRC and os.path.isdir(SRC), "Attach the adamworld-core dataset or set GIT_URL."
print("source:", SRC)

# Install adamworld's base (light) deps — NOT the pinned torch; Kaggle already ships torch.
subprocess.run([sys.executable, "-m", "pip", "install", "-q", SRC], check=True)
# GPU model libraries (use Kaggle's preinstalled torch/CUDA).
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "diffusers==0.31.0", "transformers==4.46.2", "accelerate==1.1.1",
                "safetensors", "huggingface_hub", "opencv-python-headless"], check=True)
# gsplat builds against the installed torch/CUDA (prebuilt wheel if available).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gsplat"], check=True)
print("install done")
"""

ENV_KAGGLE = """
import os, torch
os.environ["ADAMWORLD_CACHE"] = "/kaggle/working/.adamworld_cache"
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
"""

PARAMS = """
# --- parameters ---
PROMPT     = "a quiet forest clearing with a mossy stone well, soft morning light"
IMAGE      = None          # or "/kaggle/input/.../photo.jpg" to start from an image
RESOLUTION = 1024          # lower to 512/768 if you hit OOM
BACKEND    = "default"     # "default" (SDXL/FLUX/DepthAnything) | "flux-pano" (NC) | "mock"
SEED       = 0
OFFLOAD    = True          # sequential CPU offload — required to fit FLUX/SDXL on 16 GB
"""

RUN = """
from pathlib import Path
from adamworld.pipeline import Pipeline
from adamworld.profiling import Profiler
from adamworld.config import Settings
from adamworld.types import GenerationRequest

settings = Settings(offload=OFFLOAD)
prof = Profiler()
def progress(stage, frac):
    print(f"{frac*100:6.1f}%  {stage}", flush=True)

req = GenerationRequest(
    prompt=PROMPT,
    image_paths=[Path(IMAGE)] if IMAGE else [],
    resolution=RESOLUTION,
    backend=BACKEND,
    seed=SEED,
)
world = Pipeline(settings=settings, profiler=prof, progress=progress).generate(
    req, Path("/kaggle/working/world")
)
print("\\n", prof.table(), sep="")
print("world:", world.root)
"""

OUTPUT = """
import shutil
from IPython.display import Image as IPyImage, display

# Thumbnail + splat stats.
display(IPyImage(filename=str(world.thumbnail)))
from adamworld.splat import GaussianCloud
print("splats:", len(GaussianCloud.load_ply(world.splats_ply)))

# Zip the world for download (Output tab -> world.zip).
shutil.make_archive("/kaggle/working/world", "zip", world.root)
print("wrote /kaggle/working/world.zip  — download from the Output panel.")
"""

INTRO_COLAB = """
# AdamWorld — generate a 3D world on a free Colab GPU

Runs the default pipeline on a Colab **T4 (~16 GB)**. Set *Runtime → Change runtime
type → T4 GPU* first. The stage cache lives under `./world/cache`, so re-running
resumes after a disconnect.
"""

INSTALL_COLAB = """
import os, subprocess, sys

# Provide the adamworld source: upload the repo's core/ folder, or set GIT_URL to clone.
GIT_URL = ""   # e.g. "https://github.com/<you>/adamworld.git"
SRC = "adamworld/core" if os.path.isdir("adamworld/core") else "core"
if not os.path.isdir(SRC) and GIT_URL:
    subprocess.run(["git", "clone", "--depth", "1", GIT_URL, "adamworld"], check=True)
    SRC = "adamworld/core"
assert os.path.isdir(SRC), "Upload core/ (File browser) or set GIT_URL to clone the repo."

subprocess.run([sys.executable, "-m", "pip", "install", "-q", SRC], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "diffusers==0.31.0", "transformers==4.46.2", "accelerate==1.1.1",
                "safetensors", "huggingface_hub", "opencv-python-headless", "gsplat"], check=True)
print("install done")
"""

ENV_COLAB = """
import os, torch
os.environ["ADAMWORLD_CACHE"] = os.path.abspath("./.adamworld_cache")
print("CUDA:", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
"""

RUN_COLAB = RUN.replace("/kaggle/working/world", "./world")
OUTPUT_COLAB = """
from IPython.display import Image as IPyImage, display
from google.colab import files
import shutil
display(IPyImage(filename=str(world.thumbnail)))
shutil.make_archive("world", "zip", world.root)
files.download("world.zip")
"""


def main() -> None:
    kaggle = notebook([
        md(INTRO_KAGGLE), code(INSTALL_KAGGLE), code(ENV_KAGGLE),
        code(PARAMS), code(RUN), code(OUTPUT),
    ])
    colab = notebook([
        md(INTRO_COLAB), code(INSTALL_COLAB), code(ENV_COLAB),
        code(PARAMS), code(RUN_COLAB), code(OUTPUT_COLAB),
    ])
    (HERE / "kaggle_generate.ipynb").write_text(json.dumps(kaggle, indent=1), encoding="utf-8")
    (HERE / "colab_generate.ipynb").write_text(json.dumps(colab, indent=1), encoding="utf-8")
    print("wrote kaggle_generate.ipynb and colab_generate.ipynb")


if __name__ == "__main__":
    main()
