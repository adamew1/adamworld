"""Render a GaussianCloud from a novel camera with gsplat (GPU).

Used by occlusion filling (stage 5) and expansion (stage 6) to find holes and
align depth. GPU-only; raises an actionable error without the ``[gpu]`` extra.
"""

from __future__ import annotations

import numpy as np

from adamworld import geometry
from adamworld.splat.gaussians import GaussianCloud


def _viewmat(cam_pos: np.ndarray, look_lonlat: tuple[float, float]) -> np.ndarray:
    """World-to-camera (OpenCV convention: x right, y down, z forward)."""
    forward, right, up = geometry._camera_basis(look_lonlat[0], look_lonlat[1])
    x_c = right
    y_c = -up
    z_c = forward
    rot = np.stack([x_c, y_c, z_c], axis=0)  # rows = camera axes in world
    t = -rot @ np.asarray(cam_pos, np.float32)
    m = np.eye(4, dtype=np.float32)
    m[:3, :3] = rot
    m[:3, 3] = t
    return m


def render_view(
    cloud: GaussianCloud,
    cam_pos: np.ndarray,
    look_lonlat: tuple[float, float],
    fov_deg: float,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (rgb uint8 (S,S,3), alpha (S,S), depth (S,S)) for a perspective view."""
    try:
        import torch  # noqa: PLC0415
        from gsplat import rasterization  # noqa: PLC0415
    except Exception as exc:  # pragma: no cover - env dependent
        raise NotImplementedError(
            "Novel-view rendering needs the GPU extra (gsplat). "
            "Install with: pip install -e '.[gpu]' on a CUDA GPU."
        ) from exc

    dev = "cuda"
    means = torch.tensor(cloud.means, device=dev)
    quats = torch.tensor(cloud.quats, device=dev)
    scales = torch.tensor(cloud.scales, device=dev)
    opac = torch.tensor(cloud.opacity, device=dev)
    colors = torch.tensor(cloud.colors, device=dev)

    f = 0.5 * size / np.tan(np.radians(fov_deg) / 2.0)
    k = torch.tensor(
        [[[f, 0, size / 2], [0, f, size / 2], [0, 0, 1]]], device=dev, dtype=torch.float32
    )
    viewmat = torch.tensor(_viewmat(cam_pos, look_lonlat)[None], device=dev, dtype=torch.float32)

    renders, alphas, _ = rasterization(
        means, quats, scales, opac, colors, viewmat, k, size, size, render_mode="RGB+ED"
    )
    rgb = (renders[0, ..., :3].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    depth = renders[0, ..., 3].cpu().numpy()
    alpha = alphas[0, ..., 0].cpu().numpy()
    return rgb, alpha, depth
