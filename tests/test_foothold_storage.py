import torch
from instinct_rl.storage.foothold_storage import compute_next_touchdown_targets


def test_next_touchdown_single_env_one_foot():
    # T=5, N=1, F=1: contact 序列 0,0,1,0,1 -> 触地边沿在 t=2 和 t=4
    contact = torch.tensor([[0.], [0.], [1.], [0.], [1.]]).view(5, 1, 1)   # (T,N,F)
    x = torch.tensor([[0.], [0.], [2.], [0.], [4.]]).view(5, 1, 1)          # (T,N,F)
    foot_pos_w = torch.stack([x, torch.zeros_like(x)], dim=-1)             # (5,1,1,2)
    base_pose = torch.zeros(5, 1, 3)  # base=原点, yaw=0 -> base 系==世界系
    dones = torch.zeros(5, 1, 1)
    target, valid = compute_next_touchdown_targets(
        contact=contact, foot_pos_w=foot_pos_w, base_pose_w=base_pose, dones=dones, horizon=5)
    # t=0,1,2 的下一次触地在 t=2 (x=2)；t=3,4 的在 t=4 (x=4)
    assert torch.allclose(target[0, 0, 0], torch.tensor([2., 0.]), atol=1e-5)
    assert torch.allclose(target[2, 0, 0], torch.tensor([2., 0.]), atol=1e-5)
    assert torch.allclose(target[3, 0, 0], torch.tensor([4., 0.]), atol=1e-5)
    assert valid[:, 0, 0].tolist() == [1, 1, 1, 1, 1]


def test_right_censored_masked():
    # 全程摆动，无触地 -> 全部 invalid
    contact = torch.zeros(4, 1, 1)
    foot_pos_w = torch.zeros(4, 1, 1, 2)
    base_pose = torch.zeros(4, 1, 3)
    dones = torch.zeros(4, 1, 1)
    _, valid = compute_next_touchdown_targets(contact, foot_pos_w, base_pose, dones, horizon=4)
    assert valid.sum() == 0


def test_done_boundary_blocks_future():
    # t=1 done；t=2 触地不应作为 t<=1 的目标
    contact = torch.tensor([[0.], [0.], [1.]]).view(3, 1, 1)
    x = torch.tensor([[0.], [0.], [9.]]).view(3, 1, 1)
    foot_pos_w = torch.stack([x, torch.zeros_like(x)], dim=-1)
    base_pose = torch.zeros(3, 1, 3)
    dones = torch.tensor([[0.], [1.], [0.]]).view(3, 1, 1)
    _, valid = compute_next_touchdown_targets(contact, foot_pos_w, base_pose, dones, horizon=3)
    assert valid[0, 0, 0] == 0 and valid[1, 0, 0] == 0 and valid[2, 0, 0] == 1
