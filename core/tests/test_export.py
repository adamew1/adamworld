from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from adamworld.meshing import env_mesh_from_equirect
from adamworld.splat.glb_io import write_glb


def test_write_glb_structure(tmp_path: Path) -> None:
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    faces = np.array([[0, 1, 2]], np.uint32)
    colors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], np.float32)
    v, f = write_glb(tmp_path / "m.glb", verts, faces, colors)
    assert (v, f) == (3, 1)

    data = (tmp_path / "m.glb").read_bytes()
    magic, version, total = struct.unpack("<III", data[:12])
    assert magic == 0x46546C67  # "glTF"
    assert version == 2
    assert total == len(data)

    # Parse the JSON chunk.
    jlen, jtype = struct.unpack("<II", data[12:20])
    assert jtype == 0x4E4F534A  # "JSON"
    gltf = json.loads(data[20 : 20 + jlen].decode("utf-8"))
    assert gltf["asset"]["version"] == "2.0"
    assert gltf["accessors"][0]["count"] == 3  # POSITION
    assert gltf["accessors"][2]["count"] == 3  # indices (1 tri * 3)
    assert gltf["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"] == 1


def test_env_mesh_from_equirect() -> None:
    h, w = 32, 64
    depth = np.full((h, w), 3.0, np.float32)  # near shell, all kept
    rgb = np.zeros((h, w, 3), np.uint8)
    verts, faces, colors = env_mesh_from_equirect(depth, rgb, max_rows=16, far_cap=25.0)
    assert verts.shape == (16 * 32, 3)
    assert colors.shape == verts.shape
    assert faces.shape[0] > 0 and faces.shape[1] == 3
    assert faces.max() < verts.shape[0]
    # All points on a radius-3 shell.
    np.testing.assert_allclose(np.linalg.norm(verts, axis=1), 3.0, atol=1e-4)
