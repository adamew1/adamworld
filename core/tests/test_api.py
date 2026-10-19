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
