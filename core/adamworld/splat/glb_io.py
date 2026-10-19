"""Minimal binary glTF (``.glb``) writer for a colored triangle mesh.

Self-contained (no trimesh/Open3D dependency): emits a single-mesh GLB with
POSITION, COLOR_0 (vertex colors), and indices — enough for Blender/Unity import.
Pure numpy.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np


def _pad4(b: bytes, fill: bytes) -> bytes:
    pad = (-len(b)) % 4
    return b + fill * pad


def write_glb(
    path: Path,
    verts: np.ndarray,  # (V, 3) float
    faces: np.ndarray,  # (F, 3) int
    colors: np.ndarray,  # (V, 3) float in [0, 1]
) -> tuple[int, int]:
    """Write a GLB triangle mesh. Returns (vertex_count, face_count)."""
    verts = np.ascontiguousarray(verts, "<f4")
    colors = np.ascontiguousarray(np.clip(colors, 0, 1), "<f4")
    faces = np.ascontiguousarray(faces, "<u4")
    v, f = verts.shape[0], faces.shape[0]

    pos_b = verts.tobytes()
    col_b = colors.tobytes()
    idx_b = faces.reshape(-1).tobytes()
    bin_blob = pos_b + col_b + idx_b

    vmin = verts.min(axis=0).tolist()
    vmax = verts.max(axis=0).tolist()

    gltf = {
        "asset": {"version": "2.0", "generator": "adamworld"},
        "buffers": [{"byteLength": len(bin_blob)}],
        "bufferViews": [
            {"buffer": 0, "byteOffset": 0, "byteLength": len(pos_b), "target": 34962},
            {"buffer": 0, "byteOffset": len(pos_b), "byteLength": len(col_b), "target": 34962},
            {
                "buffer": 0,
                "byteOffset": len(pos_b) + len(col_b),
                "byteLength": len(idx_b),
                "target": 34963,
            },
        ],
        "accessors": [
            {
                "bufferView": 0,
                "componentType": 5126,
                "count": v,
                "type": "VEC3",
                "min": vmin,
                "max": vmax,
            },
            {"bufferView": 1, "componentType": 5126, "count": v, "type": "VEC3"},
            {"bufferView": 2, "componentType": 5125, "count": f * 3, "type": "SCALAR"},
        ],
        "meshes": [
            {"primitives": [{"attributes": {"POSITION": 0, "COLOR_0": 1}, "indices": 2, "mode": 4}]}
        ],
        "nodes": [{"mesh": 0}],
        "scenes": [{"nodes": [0]}],
        "scene": 0,
    }

    json_chunk = _pad4(json.dumps(gltf, separators=(",", ":")).encode("utf-8"), b" ")
    bin_chunk = _pad4(bin_blob, b"\x00")
    total = 12 + 8 + len(json_chunk) + 8 + len(bin_chunk)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as out:
        out.write(struct.pack("<III", 0x46546C67, 2, total))  # "glTF", version 2
        out.write(struct.pack("<II", len(json_chunk), 0x4E4F534A))  # "JSON"
        out.write(json_chunk)
        out.write(struct.pack("<II", len(bin_chunk), 0x004E4942))  # "BIN\0"
        out.write(bin_chunk)
    return v, f
