"""Content-addressed stage caching.

A stage's cache key hashes its name+version, the (already content-chained) inputs,
and the parameters that affect output. Re-running after a crash finds the cached
payload JSON and skips the work, so a cut-off session resumes instead of
restarting.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from pydantic import BaseModel


def content_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def stage_cache_key(
    name: str,
    version: str,
    inputs: BaseModel | None,
    params: dict[str, Any],
) -> str:
    inputs_json = inputs.model_dump_json() if inputs is not None else ""
    params_json = json.dumps(params, sort_keys=True, default=str)
    return content_key(name, version, inputs_json, params_json)
