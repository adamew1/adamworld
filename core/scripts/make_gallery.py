"""Assemble the gallery: generate N worlds and lay out web/public/gallery/.

Each world's thumbnail + splats.spz is copied into gallery/<id>/ and recorded in
gallery/worlds.json, which gallery.html reads. Uses the mock backend by default
(fast, CPU) for placeholder worlds; pass --backend default on a GPU for real ones.

    python scripts/make_gallery.py [--backend mock|default] [--resolution N]
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from adamworld.pipeline import Pipeline
from adamworld.profiling import Profiler
from adamworld.types import GenerationRequest

REPO = Path(__file__).resolve().parents[2]
GALLERY = REPO / "web" / "public" / "gallery"

WORLDS = [
    ("forest-glade", "a quiet forest glade with a mossy stone well, soft morning light"),
    ("desert-ruins", "ancient sandstone desert ruins at golden hour"),
    ("snow-cabin", "a cozy snow-covered cabin in a pine forest at dusk"),
    ("coral-reef", "a vibrant underwater coral reef with shafts of sunlight"),
    ("neon-alley", "a rainy neon-lit cyberpunk alley at night"),
    ("mountain-lake", "a calm alpine lake reflecting snowy mountains"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="mock")
    ap.add_argument("--resolution", type=int, default=256)
    args = ap.parse_args()

    GALLERY.mkdir(parents=True, exist_ok=True)
    manifest = []
    for i, (wid, prompt) in enumerate(WORLDS):
        print(f"[{i + 1}/{len(WORLDS)}] {wid}: {prompt}", flush=True)
        world = Pipeline(profiler=Profiler()).generate(
            GenerationRequest(prompt=prompt, resolution=args.resolution, backend=args.backend, seed=i),
            REPO / "worlds" / f"gal_{wid}",
        )
        dst = GALLERY / wid
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(world.thumbnail, dst / "thumbnail.jpg")
        shutil.copyfile(world.splats_spz, dst / "splats.spz")
        manifest.append(
            {
                "id": wid,
                "name": wid.replace("-", " ").title(),
                "thumb": f"gallery/{wid}/thumbnail.jpg",
                "spz": f"gallery/{wid}/splats.spz",
                "prompt": prompt,
            }
        )
    (GALLERY / "worlds.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"wrote {GALLERY / 'worlds.json'} ({len(manifest)} worlds)")


if __name__ == "__main__":
    main()
