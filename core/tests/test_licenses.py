from __future__ import annotations

import pytest

from adamworld.config import Settings
from adamworld.models.registry import DEPTH_ANYTHING_APACHE, ModelRegistry


def test_depth_small_is_the_apache_default() -> None:
    assert DEPTH_ANYTHING_APACHE == {"small"}


def test_nc_depth_sizes_are_gated() -> None:
    reg = ModelRegistry(Settings(allow_noncommercial=False))
    for size in ("base", "large", "giant"):
        with pytest.raises(PermissionError, match="non-commercial"):
            reg.depth_anything(size)


def test_small_passes_license_gate() -> None:
    # Small clears the license gate; it then fails only because no GPU extra is
    # installed (NotImplementedError), proving the gate let it through.
    reg = ModelRegistry(Settings(allow_noncommercial=False))
    with pytest.raises(NotImplementedError):
        reg.depth_anything("small")


def test_optin_allows_nc_sizes() -> None:
    reg = ModelRegistry(Settings(allow_noncommercial=True))
    with pytest.raises(NotImplementedError):  # past the gate -> needs GPU
        reg.depth_anything("large")
