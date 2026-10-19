"""Runtime configuration: cache/weights locations and device selection.

Weights are stored *outside* the repo in a configurable cache directory and
downloaded on first use (see ``models.registry``). Nothing here imports torch
at module load, so the core stays importable on a machine with no GPU deps.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from platformdirs import user_cache_dir
from pydantic import BaseModel, Field

_ENV_CACHE = "ADAMWORLD_CACHE"


def _default_cache_dir() -> Path:
    env = os.environ.get(_ENV_CACHE)
    if env:
        return Path(env).expanduser()
    return Path(user_cache_dir("adamworld", "adamworld"))


@lru_cache(maxsize=1)
def detect_device() -> str:
    """Return ``"cuda"`` if a CUDA torch build is present and a GPU is visible.

    Falls back to ``"cpu"``. Importing torch is optional and done lazily so the
    core package works without the ``[gpu]`` extra installed.
    """
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class Settings(BaseModel):
    """Process-wide settings. Immutable within a run."""

    cache_dir: Path = Field(default_factory=_default_cache_dir)
    device: str = Field(default_factory=detect_device)
    # Default inference precision for GPU backends.
    dtype: str = "float16"
    # Enable aggressive CPU offload / sequential model loading to fit 16 GB.
    offload: bool = False
    # Opt in to non-commercial models (e.g. Depth Anything V2 Base/Large/Giant =
    # CC-BY-NC-4.0, FLUX.1-dev pano backends). Default-off keeps the pipeline
    # Apache/OpenRAIL-only. See docs/LICENSES.md.
    allow_noncommercial: bool = False

    @property
    def weights_dir(self) -> Path:
        d = self.cache_dir / "weights"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def model_config_summary(self) -> dict[str, str]:
        return {"device": self.device, "dtype": self.dtype, "offload": str(self.offload)}


def default_settings() -> Settings:
    return Settings()
