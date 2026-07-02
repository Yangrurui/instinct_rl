"""SSR 式2（支撑缺失度 rho）/ 式3（引导 reward）的纯张量实现。

区域高度图约定：height_scan 输出的每个 ray 是命中点世界高度 z，按行主序 reshape 成
(H, W) 网格；网格中心对齐足部扫描 site（即当前脚 xy）。sole patch 是网格中心处一块
SOLE_H x SOLE_W 的子窗口；对想象点，用中心相对偏移取子窗口。
"""

import torch


def _reshape_region(scan: torch.Tensor, region_grid) -> torch.Tensor:
    H, W = region_grid
    return scan.view(scan.shape[0], H, W)


def _sole_patch(region_hw: torch.Tensor, center_ij: torch.Tensor, sole_grid) -> torch.Tensor:
    """从 (N,H,W) 网格按每 env 的中心格 (i,j) 取 (N, SOLE_H*SOLE_W)。越界夹紧。"""
    N, H, W = region_hw.shape
    sh, sw = sole_grid
    di = torch.arange(sh, device=region_hw.device) - sh // 2
    dj = torch.arange(sw, device=region_hw.device) - sw // 2
    ii = (center_ij[:, 0:1] + di.view(1, -1)).clamp(0, H - 1)          # (N, sh)
    jj = (center_ij[:, 1:2] + dj.view(1, -1)).clamp(0, W - 1)          # (N, sw)
    ii = ii[:, :, None].expand(N, sh, sw).reshape(N, -1)
    jj = jj[:, None, :].expand(N, sh, sw).reshape(N, -1)
    batch = torch.arange(N, device=region_hw.device)[:, None].expand(N, sh * sw)
    return region_hw[batch, ii, jj]                                    # (N, sh*sw)


def _deficiency(patch: torch.Tensor, sole_z: torch.Tensor, epsilon_h: float) -> torch.Tensor:
    # rho = 1 - mean_k 1[sole_z - h_k < epsilon_h]
    supported = (sole_z.unsqueeze(-1) - patch < epsilon_h).float()
    return 1.0 - supported.mean(dim=-1)


def support_deficiency_at_center(scan, sole_z, region_grid, region_res, sole_grid, epsilon_h):
    """支撑脚：patch 取网格正中心。scan:(N,H*W), sole_z:(N,)."""
    region_hw = _reshape_region(scan, region_grid)
    N, H, W = region_hw.shape
    center = torch.tensor([[H // 2, W // 2]], device=scan.device).expand(N, 2)
    patch = _sole_patch(region_hw, center, sole_grid)
    return _deficiency(patch, sole_z, epsilon_h)


def support_deficiency_at_point(scan, offset_xy, region_grid, region_res, sole_grid, epsilon_h):
    """想象点：patch 中心 = 网格中心 + offset_xy(米, scanner 系)。
    scan:(N,H*W), offset_xy:(N,2). sole_z 用 patch 内最大高度（SSR: h^f=max_k h_k）。"""
    region_hw = _reshape_region(scan, region_grid)
    N, H, W = region_hw.shape
    # 网格行=x(前后)、列=y(左右)：以行主序为 (i沿x, j沿y)。offset->格数
    di = torch.round(offset_xy[:, 0] / region_res).long()
    dj = torch.round(offset_xy[:, 1] / region_res).long()
    center = torch.stack([H // 2 + di, W // 2 + dj], dim=-1).clamp(
        torch.tensor([0, 0], device=scan.device),
        torch.tensor([H - 1, W - 1], device=scan.device))
    patch = _sole_patch(region_hw, center, sole_grid)
    sole_z = patch.max(dim=-1)[0]
    return _deficiency(patch, sole_z, epsilon_h)


def expected_support_deficiency(scan, mu_xy, log_sigma, region_grid, region_res,
                                sole_grid, epsilon_h, num_samples: int = 8):
    """摆动脚：E_{p~N(mu,sigma^2 I)}[rho(p)]，离散高斯采样近似（SSR 式3）。
    mu_xy:(N,2) scanner 系相对偏移；log_sigma:(N,)."""
    sigma = torch.exp(log_sigma).unsqueeze(-1)                         # (N,1)
    eps = torch.randn(mu_xy.shape[0], num_samples, 2, device=scan.device)
    samples = mu_xy.unsqueeze(1) + sigma.unsqueeze(-1) * eps           # (N,S,2)
    rho = []
    for s in range(num_samples):
        rho.append(support_deficiency_at_point(
            scan, samples[:, s, :], region_grid, region_res, sole_grid, epsilon_h))
    return torch.stack(rho, dim=-1).mean(dim=-1)                       # (N,)


def guidance_reward(rho_tilde: torch.Tensor, sigma_f: float) -> torch.Tensor:
    """SSR 式3: r^f = exp(-(sum_i rho_tilde_i)^2 / sigma_f). rho_tilde:(N,2)."""
    s = rho_tilde.sum(dim=-1)
    return torch.exp(-(s ** 2) / sigma_f)
