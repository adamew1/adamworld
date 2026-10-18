"""Small image helpers (PIL + numpy) shared by stages and mock backends."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image


def seeded_rng(seed: int, *tokens: str) -> np.random.Generator:
    h = hashlib.sha256(str(seed).encode())
    for t in tokens:
        h.update(b"\x00")
        h.update(t.encode("utf-8"))
    return np.random.default_rng(int.from_bytes(h.digest()[:8], "little"))


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def save_rgb(path: Path, arr: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB").save(path)


def resize(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    im = Image.fromarray(arr.astype(np.uint8), "RGB").resize(
        (width, height), Image.Resampling.BILINEAR
    )
    return np.asarray(im, dtype=np.uint8)


def save_thumbnail(path: Path, arr: np.ndarray, max_w: int = 512) -> None:
    h, w = arr.shape[:2]
    scale = min(1.0, max_w / max(1, w))
    tw, th = max(1, int(w * scale)), max(1, int(h * scale))
    im = Image.fromarray(arr.astype(np.uint8), "RGB").resize((tw, th), Image.Resampling.BILINEAR)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    im.convert("RGB").save(path, "JPEG", quality=85)
