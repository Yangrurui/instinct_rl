import torch
from instinct_rl.algorithms.foothold_math import (
    support_deficiency_at_center, expected_support_deficiency, guidance_reward,
)

REGION_H, REGION_W, REGION_RES = 21, 13, 0.025
SOLE_H, SOLE_W = 10, 5
EPS_H = 0.03


def _flat_region(value, n=4):
    # n 个 env，全平地高度=0 的区域图
    return torch.full((n, REGION_H * REGION_W), value)


def test_flat_ground_zero_deficiency():
    scan = _flat_region(0.0)                     # 地面 z=0
    sole_z = torch.zeros(4)                       # 鞋底 z=0
    rho = support_deficiency_at_center(
        scan, sole_z, region_grid=(REGION_H, REGION_W), region_res=REGION_RES,
        sole_grid=(SOLE_H, SOLE_W), epsilon_h=EPS_H)
    assert torch.allclose(rho, torch.zeros(4), atol=1e-6)


def test_deep_gap_full_deficiency():
    scan = _flat_region(-1.0)                    # 地面在鞋底下方 1m
    sole_z = torch.zeros(4)
    rho = support_deficiency_at_center(
        scan, sole_z, region_grid=(REGION_H, REGION_W), region_res=REGION_RES,
        sole_grid=(SOLE_H, SOLE_W), epsilon_h=EPS_H)
    assert torch.allclose(rho, torch.ones(4), atol=1e-6)


def test_guidance_reward_monotonic():
    # rho 越大 reward 越小；rho=0 -> 1
    r0 = guidance_reward(torch.zeros(4, 2), sigma_f=0.0625)
    r1 = guidance_reward(torch.full((4, 2), 0.5), sigma_f=0.0625)
    assert torch.allclose(r0, torch.ones(4), atol=1e-6)
    assert torch.all(r1 < r0)
