from __future__ import annotations

import time
from pathlib import Path

from fastapi.testclient import TestClient

from adamworld.api import create_app


def _wait_done(client: TestClient, job_id: str, timeout: float = 30.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/jobs/{job_id}").json()
        if job["status"] in ("done", "error"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish in time")


def test_generate_and_fetch_world(tmp_path: Path) -> None:
    app = create_app(worlds_root=tmp_path / "worlds")
    client = TestClient(app)

    resp = client.post(
        "/worlds",
        json={"prompt": "a mossy ruin", "backend": "mock", "resolution": 64},
    )
    assert resp.status_code == 200
    body = resp.json()
    job = _wait_done(client, body["job_id"])
    assert job["status"] == "done", job.get("error")

    world_id = body["world_id"]
    got = client.get(f"/worlds/{world_id}").json()
    assert got["meta"]["input"]["kind"] == "prompt"
    assert "splats.ply" in got["assets"]

    # Asset + export download work.
    asset = client.get(f"/worlds/{world_id}/asset/splats.ply")
    assert asset.status_code == 200 and len(asset.content) > 0
    assert client.get(f"/worlds/{world_id}/export", params={"format": "ply"}).status_code == 200
    # spz is the primary format (written alongside ply).
    assert "splats.spz" in got["assets"]
    assert client.get(f"/worlds/{world_id}/export", params={"format": "spz"}).status_code == 200
    # glb (mesh) still unavailable until the mesh stage runs.
    assert client.get(f"/worlds/{world_id}/export", params={"format": "glb"}).status_code == 404


def test_unknown_job_and_world(tmp_path: Path) -> None:
    client = TestClient(create_app(worlds_root=tmp_path / "worlds"))
    assert client.get("/jobs/nope").status_code == 404
    assert client.get("/worlds/nope").status_code == 404


def test_layout_generate_and_region_delete(tmp_path: Path) -> None:
    client = TestClient(create_app(worlds_root=tmp_path / "worlds"))
    layout = {
        "style_prompt": "a hall",
        "blocks": [
            {
                "id": "b0",
                "shape": "box",
                "label": "room",
                "transform": [8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 8, 0, 0, 0, 0, 1],
            }
        ],
    }
    resp = client.post("/worlds", json={"layout": layout, "backend": "mock", "resolution": 64})
    body = resp.json()
    assert _wait_done(client, body["job_id"])["status"] == "done"

    wid = body["world_id"]
    got = client.get(f"/worlds/{wid}").json()
    assert got["meta"]["input"]["kind"] == "layout"

    # Region delete via the edit endpoint.
    edit = client.post(
        f"/worlds/{wid}/edit",
        json={"aabb_min": [-50, -50, -50], "aabb_max": [50, 50, 50], "delete": True},
    )
    assert edit.status_code == 200


def test_layout_api_honors_default_backend(tmp_path: Path, monkeypatch) -> None:
    from adamworld.pipeline import Pipeline

    observed = []

    def generate(self, layout, out_dir, **kwargs):
        observed.append(kwargs["backend"])

    monkeypatch.setattr(Pipeline, "generate_from_layout", generate)
    client = TestClient(create_app(worlds_root=tmp_path / "worlds"))
    response = client.post("/worlds", json={"layout": {"blocks": [], "style_prompt": "hall"}})
    assert response.status_code == 200
    assert _wait_done(client, response.json()["job_id"])["status"] == "done"
    assert observed == ["default"]
