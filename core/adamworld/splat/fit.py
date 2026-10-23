"""Short gsplat optimization of unprojected splats (stage 4 refinement, GPU).

The lift produces one Gaussian per sampled panorama pixel. This pass turns that
into a *fitted* splat field: it renders the current Gaussians from a set of
perspective views of the panorama, compares against the panorama itself
(equirect→perspective ground truth), and optimizes the Gaussian parameters
(means, log-scales, quaternions, colors, opacities) with Adam to minimize the
photometric error, then prunes transparent splats. gsplat's rasterizer is
differentiable, so this is standard 3DGS-style fitting — just short.

GPU-only (needs torch + gsplat); callers fall back to the unprojected cloud when
unavailable. Camera math matches ``splat.render`` / ``geometry``.
"""

from __future__ import annotations

import numpy as np

from adamworld.splat.gaussians import GaussianCloud


def fit_gaussians(
    cloud: GaussianCloud,
    rgb_equirect: np.ndarray,  # (H, W, 3) uint8 panorama = ground truth
    *,
    iters: int = 300,
    view_size: int = 256,
    fov_deg: float = 90.0,
    prune_opacity: float = 0.05,
) -> GaussianCloud:
    """Fit and prune splats; raise on missing dependencies or fitting failures."""
    try:
        import torch
        from gsplat import rasterization

        from adamworld import geometry
        from adamworld.splat.render import _viewmat
    except ImportError as exc:
        raise RuntimeError(
            "Splat fitting requires torch and gsplat; install the GPU extra."
        ) from exc

    if len(cloud) == 0:
        return cloud

    try:
        dev = "cuda"

        def _inv_sig(p: np.ndarray) -> np.ndarray:
            p = np.clip(p, 1e-4, 1 - 1e-4)
            return np.log(p / (1 - p))

        means = torch.tensor(cloud.means, device=dev, requires_grad=True)
        log_scales = torch.tensor(
            np.log(np.clip(cloud.scales, 1e-6, None)), device=dev, requires_grad=True
        )
        quats = torch.tensor(cloud.quats, device=dev, requires_grad=True)
        raw_op = torch.tensor(_inv_sig(cloud.opacity), device=dev, requires_grad=True)
        raw_col = torch.tensor(_inv_sig(cloud.colors), device=dev, requires_grad=True)

        # Ground-truth tangent views of the panorama + their cameras.
        looks = [(float(lon), 0.0) for lon in np.linspace(-np.pi, np.pi, 8, endpoint=False)]
        looks += [(0.0, float(np.radians(70))), (0.0, float(np.radians(-70)))]
        f = 0.5 * view_size / np.tan(np.radians(fov_deg) / 2.0)
        k = torch.tensor(
            [[f, 0, view_size / 2], [0, f, view_size / 2], [0, 0, 1]], device=dev
        ).float()
        cam0 = np.zeros(3, np.float32)
        gts, viewmats = [], []
        for look in looks:
            gt = geometry.equirect_to_perspective(
                rgb_equirect, look[0], look[1], fov_deg, view_size
            )
            gts.append(torch.tensor(gt.astype(np.float32) / 255.0, device=dev))
            viewmats.append(torch.tensor(_viewmat(cam0, look), device=dev).float())

        opt = torch.optim.Adam(
            [
                {"params": [means], "lr": 2e-4},
                {"params": [log_scales], "lr": 5e-3},
                {"params": [quats], "lr": 1e-3},
                {"params": [raw_op], "lr": 5e-2},
                {"params": [raw_col], "lr": 5e-3},
            ]
        )
        for it in range(iters):
            i = it % len(gts)
            render, _, _ = rasterization(
                means,
                quats / quats.norm(dim=-1, keepdim=True),
                torch.exp(log_scales),
                torch.sigmoid(raw_op),
                torch.sigmoid(raw_col),
                viewmats[i][None],
                k[None],
                view_size,
                view_size,
                render_mode="RGB",
            )
            loss = (render[0] - gts[i]).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()

        with torch.no_grad():
            op = torch.sigmoid(raw_op)
            keep = (op > prune_opacity).cpu().numpy()
            fitted = GaussianCloud(
                means=means.detach().cpu().numpy()[keep],
                colors=torch.sigmoid(raw_col).detach().cpu().numpy()[keep],
                opacity=op.detach().cpu().numpy()[keep],
                scales=torch.exp(log_scales).detach().cpu().numpy()[keep],
                quats=(quats / quats.norm(dim=-1, keepdim=True)).detach().cpu().numpy()[keep],
            )
        return fitted if len(fitted) > 0 else cloud
    except Exception:
        raise
