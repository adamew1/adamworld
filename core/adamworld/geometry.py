"""Spherical / equirectangular geometry — the CPU core of stages 2-3.

Conventions match ``splat.gaussians.from_rgbd_equirect``: y is up, a ray's
longitude is ``atan2(x, z)`` in ``[-pi, pi)`` across image width, latitude is
``asin(y)`` mapped top=+pi/2 to bottom=-pi/2. All functions are pure numpy so
they run and are tested on CPU; the diffusion/depth models that consume these
projections run on the GPU (stages 2-3 ``default`` backend).
"""

from __future__ import annotations

import numpy as np

# --- direction <-> equirect uv ----------------------------------------------


def dirs_to_equirect_uv(dirs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit directions (..., 3) -> (u, v) in [0, 1). Width wraps, height clamps."""
    x, y, z = dirs[..., 0], dirs[..., 1], dirs[..., 2]
    lon = np.arctan2(x, z)
    lat = np.arcsin(np.clip(y, -1.0, 1.0))
    u = (lon + np.pi) / (2.0 * np.pi)
    v = (0.5 * np.pi - lat) / np.pi
    return u, v


def equirect_uv_to_dirs(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(u, v) in [0, 1) -> unit directions (..., 3)."""
    lon = u * 2.0 * np.pi - np.pi
    lat = 0.5 * np.pi - v * np.pi
    cos_lat = np.cos(lat)
    return np.stack([cos_lat * np.sin(lon), np.sin(lat), cos_lat * np.cos(lon)], axis=-1)


# --- bilinear sampling of an equirect image ---------------------------------


def sample_equirect(img: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Bilinearly sample ``img`` (H, W, C) at (u, v) in [0,1). Wraps in u, clamps v."""
    h, w = img.shape[:2]
    img_f = img.astype(np.float32)
    fx = u * w - 0.5
    fy = v * h - 0.5
    x0 = np.floor(fx).astype(np.int64)
    y0 = np.floor(fy).astype(np.int64)
    wx = (fx - x0)[..., None]
    wy = (fy - y0)[..., None]
    x0m, x1m = np.mod(x0, w), np.mod(x0 + 1, w)
    y0c, y1c = np.clip(y0, 0, h - 1), np.clip(y0 + 1, 0, h - 1)
    c00, c10 = img_f[y0c, x0m], img_f[y0c, x1m]
    c01, c11 = img_f[y1c, x0m], img_f[y1c, x1m]
    top = c00 * (1 - wx) + c10 * wx
    bot = c01 * (1 - wx) + c11 * wx
    return top * (1 - wy) + bot * wy


# --- perspective (tangent) patches ------------------------------------------


def _camera_basis(lon0: float, lat0: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = equirect_uv_to_dirs(
        np.array((lon0 + np.pi) / (2 * np.pi)), np.array((0.5 * np.pi - lat0) / np.pi)
    )
    world_up = np.array([0.0, 1.0, 0.0], np.float32)
    if abs(float(forward[1])) > 0.999:  # looking near the poles
        world_up = np.array([0.0, 0.0, 1.0], np.float32)
    right = np.cross(world_up, forward)
    right /= np.linalg.norm(right) + 1e-9
    up = np.cross(forward, right)
    return forward.astype(np.float32), right.astype(np.float32), up.astype(np.float32)


def perspective_rays(lon0: float, lat0: float, fov_deg: float, size: int) -> np.ndarray:
    """Unit ray directions (size, size, 3) for a square perspective view."""
    forward, right, up = _camera_basis(lon0, lat0)
    t = np.tan(np.radians(fov_deg) / 2.0)
    px = (np.arange(size) + 0.5) / size * 2 - 1
    py = 1 - (np.arange(size) + 0.5) / size * 2
    gx, gy = np.meshgrid(px * t, py * t)
    rays = forward[None, None, :] + gx[..., None] * right + gy[..., None] * up
    return (rays / (np.linalg.norm(rays, axis=-1, keepdims=True) + 1e-9)).astype(np.float32)


def equirect_to_perspective(
    equirect: np.ndarray, lon0: float, lat0: float, fov_deg: float, size: int
) -> np.ndarray:
    """Render a perspective crop of an equirect image."""
    rays = perspective_rays(lon0, lat0, fov_deg, size)
    u, v = dirs_to_equirect_uv(rays)
    return sample_equirect(equirect, u, v)


def perspective_to_equirect_accumulate(
    values: np.ndarray,  # (size, size) or (size, size, C)
    lon0: float,
    lat0: float,
    fov_deg: float,
    out_h: int,
    out_w: int,
    accum: np.ndarray,
    weight: np.ndarray,
) -> None:
    """Splat a perspective patch's values back onto equirect accumulators.

    Uses a cosine-falloff weight (strong at patch center, feathering to edges)
    so overlapping patches blend seamlessly. Modifies ``accum``/``weight`` in place.
    """
    size = values.shape[0]
    rays = perspective_rays(lon0, lat0, fov_deg, size)
    u, v = dirs_to_equirect_uv(rays)
    xs = np.clip((u * out_w).astype(np.int64), 0, out_w - 1)
    ys = np.clip((v * out_h).astype(np.int64), 0, out_h - 1)

    pc = (np.arange(size) + 0.5) / size * 2 - 1
    gx, gy = np.meshgrid(pc, pc)
    w = np.clip(1.0 - np.sqrt(gx**2 + gy**2), 0.0, 1.0).astype(np.float32)

    vals = values.astype(np.float32)
    if vals.ndim == 2:
        np.add.at(accum, (ys, xs), vals * w)
    else:
        np.add.at(accum, (ys, xs), vals * w[..., None])
    np.add.at(weight, (ys, xs), w)


def ring_centers(n_horizontal: int = 8) -> list[tuple[float, float]]:
    """Tangent-view centers covering the sphere: a horizon ring + up + down."""
    centers = [(float(i) / n_horizontal * 2 * np.pi - np.pi, 0.0) for i in range(n_horizontal)]
    centers.append((0.0, float(np.radians(75))))  # up
    centers.append((0.0, float(np.radians(-75))))  # down
    return centers


# --- seed-image placement for panorama outpainting (stage 2) ----------------


def place_image_front(
    seed_rgb: np.ndarray, out_h: int, out_w: int, fov_deg: float = 90.0
) -> tuple[np.ndarray, np.ndarray]:
    """Paint a perspective seed image onto the front of an equirect canvas.

    Returns ``(canvas_uint8, mask)`` where ``mask`` is 1.0 where the seed is
    known (to be kept) and 0.0 where outpainting must fill. The front view faces
    ``+z`` (lon=0, lat=0).
    """
    sh, sw = seed_rgb.shape[:2]
    canvas = np.zeros((out_h, out_w, 3), np.float32)
    mask = np.zeros((out_h, out_w), np.float32)
    t = np.tan(np.radians(fov_deg) / 2.0)

    vv, uu = np.mgrid[0:out_h, 0:out_w]
    u = (uu + 0.5) / out_w
    v = (vv + 0.5) / out_h
    dirs = equirect_uv_to_dirs(u, v)
    fz = dirs[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        sx = dirs[..., 0] / fz
        sy = dirs[..., 1] / fz
    inside = (fz > 1e-3) & (np.abs(sx) <= t) & (np.abs(sy) <= t)
    px = ((sx / t * 0.5 + 0.5) * sw).astype(np.int64)
    py = ((0.5 - sy / t * 0.5) * sh).astype(np.int64)
    px = np.clip(px, 0, sw - 1)
    py = np.clip(py, 0, sh - 1)
    seed_f = seed_rgb.astype(np.float32)
    canvas[inside] = seed_f[py[inside], px[inside]]
    mask[inside] = 1.0
    return np.clip(canvas, 0, 255).astype(np.uint8), mask


def perspective_unproject(
    depth: np.ndarray,  # (S, S) radial depth
    colors: np.ndarray,  # (S, S, 3)
    fov_deg: float,
    cam_pos: np.ndarray,  # (3,)
    look_lonlat: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Unproject a perspective depth+color view at ``cam_pos`` into world points."""
    size = depth.shape[0]
    rays = perspective_rays(look_lonlat[0], look_lonlat[1], fov_deg, size)
    pts = np.asarray(cam_pos, np.float32)[None, None, :] + rays * depth[..., None]
    return pts.reshape(-1, 3).astype(np.float32), colors.reshape(-1, 3).astype(np.float32)


def fit_scale_shift(
    src: np.ndarray, ref: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, float, float]:
    """Least-squares a,b so a*src+b ~= ref over ``mask``. Aligns inpainted-view
    depth to the rendered depth of existing geometry before lifting new pixels."""
    s = src[mask].ravel().astype(np.float64)
    r = ref[mask].ravel().astype(np.float64)
    if s.size < 2:
        return src.astype(np.float32), 1.0, 0.0
    a_mat = np.stack([s, np.ones_like(s)], axis=1)
    (a, b), *_ = np.linalg.lstsq(a_mat, r, rcond=None)
    return (a * src + b).astype(np.float32), float(a), float(b)


def horizontal_wrap_pad(img: np.ndarray, pad: int) -> np.ndarray:
    """Circular-pad left/right by ``pad`` columns (seam-free 360 denoising)."""
    if pad <= 0:
        return img
    return np.concatenate([img[:, -pad:], img, img[:, :pad]], axis=1)
