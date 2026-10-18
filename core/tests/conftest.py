from __future__ import annotations

# Redirect the temp root to a project-local dir BEFORE pytest's tmp_path factory
# first resolves it. This keeps `pytest -q` working even when the system temp
# (e.g. a locked %TEMP%\pytest-of-<user>) is unwritable on a given machine.
import os  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402

_TMP = Path(__file__).resolve().parents[1] / ".pytest_tmp"
_TMP.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = os.environ["TEMP"] = os.environ["TMP"] = str(_TMP)
tempfile.tempdir = str(_TMP)

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PIL import Image  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADAMWORLD_CACHE", str(tmp_path / "cache"))


@pytest.fixture
def sample_image(tmp_path: Path) -> Path:
    arr = (np.random.default_rng(0).integers(0, 256, (48, 64, 3))).astype(np.uint8)
    p = tmp_path / "seed.png"
    Image.fromarray(arr, "RGB").save(p)
    return p
