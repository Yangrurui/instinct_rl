"""落脚点想象的独立缓冲（仿 AmpStorage）：存 privileged 特征与几何量，
在有序 [T,N] 上前向回填"下一次触地点"作为 NLL 目标（SSR 式1）。"""

from collections import namedtuple

import torch


def compute_next_touchdown_targets(contact, foot_pos_w, base_pose_w, dones, horizon):
    """
    Args:
        contact:    (T,N,F) in {0,1}
        foot_pos_w: (T,N,F,2) 世界系 xy
        base_pose_w:(T,N,3)  [x,y,yaw] 世界系
        dones:      (T,N,1)
    Returns:
        target_b: (T,N,F,2) 目标触地点在【查询步 t 的 base 系】
        valid:    (T,N,F)   1 表示 horizon 内、同 episode 内找到触地边沿
    说明：触地边沿定义为 contact[t']==1 且 contact[t'-1]==0；t'=0 视为非边沿(False)。
    target[t] = 首个 t'>=t 的触地边沿处 foot_pos_w；跨越 dones 边界不算。
    """
    T, N, F = contact.shape
    device = contact.device
    # 触地边沿
    prev = torch.zeros_like(contact)
    prev[1:] = contact[:-1]
    edge = ((contact == 1) & (prev == 0)).float()                     # (T,N,F)

    target_w = torch.zeros(T, N, F, 2, device=device)
    valid = torch.zeros(T, N, F, device=device)

    future_pos = torch.zeros(N, F, 2, device=device)
    future_valid = torch.zeros(N, F, device=device)
    for t in reversed(range(T)):
        # dones[t]==1: t 是本 episode 末步，来自 t+1 的 carry 属于下一 episode，切断
        keep = (dones[t] == 0).float().view(N, 1)                      # (N,1)
        future_valid = future_valid * keep
        future_pos = future_pos * keep.unsqueeze(-1)

        e = edge[t]                                                    # (N,F)
        this_valid = torch.clamp(e + future_valid, max=1.0)
        this_pos = torch.where(e.unsqueeze(-1) > 0, foot_pos_w[t], future_pos)
        target_w[t] = this_pos
        valid[t] = this_valid
        future_valid = this_valid
        future_pos = this_pos

    # 世界 -> 查询步 base 系
    bx = base_pose_w[..., 0:1].unsqueeze(2)                            # (T,N,1,1)
    by = base_pose_w[..., 1:2].unsqueeze(2)
    yaw = base_pose_w[..., 2:3].unsqueeze(2)                           # (T,N,1,1)
    dx = target_w[..., 0:1] - bx
    dy = target_w[..., 1:2] - by
    c = torch.cos(-yaw); s = torch.sin(-yaw)
    px = c * dx - s * dy
    py = s * dx + c * dy
    target_b = torch.cat([px, py], dim=-1)                            # (T,N,F,2)
    return target_b, valid


class FootholdStorage:
    class Transition:
        def __init__(self):
            self.feat = None
            self.foot_contact = None
            self.foot_pos_w = None
            self.base_pose_w = None
            self.terrain_level = None
            self.dones = None

        def clear(self):
            self.__init__()

    MiniBatch = namedtuple("MiniBatch", ["feat", "target", "valid", "terrain_level"])

    def __init__(self, num_envs, num_transitions_per_env, feat_size, num_feet=2, device="cpu"):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.num_feet = num_feet
        T, N = num_transitions_per_env, num_envs
        self.feat = torch.zeros(T, N, feat_size, device=device)
        self.foot_contact = torch.zeros(T, N, num_feet, device=device)
        self.foot_pos_w = torch.zeros(T, N, num_feet, 2, device=device)
        self.base_pose_w = torch.zeros(T, N, 3, device=device)
        self.terrain_level = torch.zeros(T, N, 1, device=device)
        self.dones = torch.zeros(T, N, 1, device=device)
        self.step = 0
        self._target = None
        self._valid = None

    def add_transitions(self, tr: "FootholdStorage.Transition"):
        if self.step >= self.num_transitions_per_env:
            raise AssertionError("Foothold buffer overflow")
        self.feat[self.step].copy_(tr.feat)
        self.foot_contact[self.step].copy_(tr.foot_contact)
        self.foot_pos_w[self.step].copy_(tr.foot_pos_w.view(self.num_envs, self.num_feet, 2))
        self.base_pose_w[self.step].copy_(tr.base_pose_w)
        self.terrain_level[self.step].copy_(tr.terrain_level)
        self.dones[self.step].copy_(tr.dones.view(-1, 1))
        self.step += 1

    def compute_targets(self):
        self._target, self._valid = compute_next_touchdown_targets(
            self.foot_contact, self.foot_pos_w, self.base_pose_w, self.dones,
            horizon=self.num_transitions_per_env)

    def clear(self):
        self.step = 0
        self._target = None
        self._valid = None

    def mini_batch_generator(self, num_mini_batches, num_epochs=5):
        T, N = self.num_transitions_per_env, self.num_envs
        batch = T * N
        mb = batch // num_mini_batches
        feat = self.feat.reshape(batch, -1)
        target = self._target.reshape(batch, self.num_feet, 2)
        valid = self._valid.reshape(batch, self.num_feet)
        level = self.terrain_level.reshape(batch, 1)
        for _ in range(num_epochs):
            idx = torch.randperm(num_mini_batches * mb, device=self.device)
            for i in range(num_mini_batches):
                sel = idx[i * mb:(i + 1) * mb]
                yield FootholdStorage.MiniBatch(feat[sel], target[sel], valid[sel], level[sel])
