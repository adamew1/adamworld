from __future__ import annotations

from adamworld.cache import content_key, stage_cache_key
from adamworld.types import ImageRef


def test_content_key_stable_and_sensitive() -> None:
    assert content_key("a", "b") == content_key("a", "b")
    assert content_key("a", "b") != content_key("b", "a")


def test_stage_cache_key_depends_on_inputs_and_params() -> None:
    img_a = ImageRef(path="a.png", width=10, height=10, key="k1")
    img_b = ImageRef(path="a.png", width=10, height=10, key="k2")
    base = stage_cache_key("panorama", "1", img_a, {"backend": "mock"})
    assert base == stage_cache_key("panorama", "1", img_a, {"backend": "mock"})
    # Different upstream content -> different key (content chaining).
    assert base != stage_cache_key("panorama", "1", img_b, {"backend": "mock"})
    # Different params -> different key.
    assert base != stage_cache_key("panorama", "1", img_a, {"backend": "default"})
    # Version bump invalidates.
    assert base != stage_cache_key("panorama", "2", img_a, {"backend": "mock"})
