"""FootholdAlgoMixin：SSR imagined foothold guidance 的算法插件（仿 WasabiAlgoMixin）。

- 独立网络 FootholdImaginationModel + 独立 Adam 优化器 + 独立 FootholdStorage。
- compute_auxiliary_reward: 用想象输出算 r^f（式3）注入 rollout reward（键名 foothold_reward）。
- update: PPO/AMP 更新后，回填 GT 并用 NLL（式1）训练想象网络。
"""

import torch
import torch.optim as optim

from instinct_rl.algorithms.foothold_math import (
    expected_support_deficiency, guidance_reward, support_deficiency_at_center,
)
from instinct_rl.modules.foothold_imagination import FootholdImaginationModel
from instinct_rl.storage.foothold_storage import FootholdStorage
from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size


class FootholdAlgoMixin:
    def __init__(
        self,
        *args,
        foothold_state_key="foothold",
        imagination_feature_components=(
            "foot_region_scan_l", "foot_region_scan_r", "foot_contact",
            "foot_pos_b", "foot_vel_b", "base_lin_vel", "last_action",
        ),
        region_grid=(13, 21),
        region_res=0.025,
        sole_grid=(5, 10),
        epsilon_h=0.03,
        sigma_f=0.0625,
        num_swing_samples=8,
        terrain_level_gate=5,
        imagination_hidden_sizes=(256, 128),
        imagination_lr=5e-4,
        foothold_reward_coef=0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.foothold_state_key = foothold_state_key
        self.imagination_feature_components = list(imagination_feature_components)
        self.region_grid = tuple(region_grid)
        self.region_res = region_res
        self.sole_grid = tuple(sole_grid)
        self.epsilon_h = epsilon_h
        self.sigma_f = sigma_f
        self.num_swing_samples = num_swing_samples
        self.terrain_level_gate = terrain_level_gate
        self.imagination_hidden_sizes = tuple(imagination_hidden_sizes)
        self.imagination_lr = imagination_lr
        self.foothold_reward_coef = foothold_reward_coef  # 供 PPO.process_env_step 的 *_coef 机制

    def init_storage(self, num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards=1):
        super().init_storage(num_envs, num_transitions_per_env, obs_format, num_actions, num_rewards)
        self._foothold_segments = obs_format[self.foothold_state_key]
        feat_size = get_subobs_size(self._foothold_segments, self.imagination_feature_components)
        self.imagination = FootholdImaginationModel(
            input_size=feat_size, hidden_sizes=self.imagination_hidden_sizes).to(self.device)
        self.imagination_optimizer = optim.Adam(self.imagination.parameters(), lr=self.imagination_lr)
        self.foothold_storage = FootholdStorage(
            num_envs, num_transitions_per_env, feat_size=feat_size, num_feet=2, device=self.device)
        self.foothold_transition = FootholdStorage.Transition()

    # ---- 从 obs_pack["foothold"] 切片 ----
    def _fh_slice(self, fh, names):
        return get_subobs_by_components(fh, names, self._foothold_segments)

    def _imagination_input(self, fh):
        return self._fh_slice(fh, self.imagination_feature_components)

    def process_env_step(self, rewards, dones, infos, next_obs, next_critic_obs):
        fh = infos["observations"][self.foothold_state_key]
        self.foothold_transition.feat = self._imagination_input(fh).clone()
        self.foothold_transition.foot_contact = self._fh_slice(fh, ["foot_contact"]).clone()
        self.foothold_transition.foot_pos_w = self._fh_slice(fh, ["foot_pos_w"]).clone()
        self.foothold_transition.base_pose_w = self._fh_slice(fh, ["base_pose_w"]).clone()
        self.foothold_transition.terrain_level = self._fh_slice(fh, ["terrain_level"]).clone()
        self.foothold_transition.dones = dones.clone()
        self.foothold_storage.add_transitions(self.foothold_transition)
        self.foothold_transition.clear()
        super().process_env_step(rewards, dones, infos, next_obs, next_critic_obs)

    @torch.no_grad()
    def compute_auxiliary_reward(self, obs_pack):
        aux = super().compute_auxiliary_reward(obs_pack)
        fh = obs_pack[self.foothold_state_key]
        feat = self._imagination_input(fh)
        mu, log_sigma = self.imagination(feat)                          # (N,2,2),(N,2)

        scan_l = self._fh_slice(fh, ["foot_region_scan_l"])
        scan_r = self._fh_slice(fh, ["foot_region_scan_r"])
        contact = self._fh_slice(fh, ["foot_contact"])                  # (N,2)
        foot_pos_b = self._fh_slice(fh, ["foot_pos_b"]).view(-1, 2, 2)  # (N,2,2)
        sole_z = self._fh_slice(fh, ["foot_sole_z"])                    # (N,2) 每脚鞋底平面高度
        level = self._fh_slice(fh, ["terrain_level"]).squeeze(-1)       # (N,)

        rho = torch.zeros_like(contact)                                 # (N,2)
        for i, scan in enumerate((scan_l, scan_r)):
            rho_stance = support_deficiency_at_center(
                scan, sole_z[:, i], self.region_grid, self.region_res, self.sole_grid, self.epsilon_h)
            # 想象点相对 scanner 中心的偏移 = mu(base) - 当前脚(base)
            offset = mu[:, i, :] - foot_pos_b[:, i, :]
            rho_swing = expected_support_deficiency(
                scan, offset, log_sigma[:, i], self.region_grid, self.region_res,
                self.sole_grid, self.epsilon_h, num_samples=self.num_swing_samples)
            rho[:, i] = contact[:, i] * rho_stance + (1.0 - contact[:, i]) * rho_swing

        rf = guidance_reward(rho, self.sigma_f)                         # (N,)
        gate = (level >= self.terrain_level_gate).float()
        aux["foothold_reward"] = (rf * gate).unsqueeze(-1)
        return aux

    def update(self, *args, **kwargs):
        mean_losses, average_stats = super().update(*args, **kwargs)
        self.foothold_storage.compute_targets()
        num_updates = self.num_learning_epochs * self.num_mini_batches
        for mb in self.foothold_storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs):
            mu, log_sigma = self.imagination(mb.feat)                   # (B,2,2),(B,2)
            sq = ((mb.target - mu) ** 2).sum(dim=-1)                    # (B,2)
            nll = sq / (2.0 * torch.exp(2.0 * log_sigma)) + 2.0 * log_sigma  # (B,2), SSR 式1
            gate = (mb.terrain_level.squeeze(-1) >= self.terrain_level_gate).float().unsqueeze(-1)
            m = mb.valid * gate                                         # (B,2)
            loss = (nll * m).sum() / m.sum().clamp(min=1.0)
            self.imagination_optimizer.zero_grad()
            loss.backward()
            self.imagination_optimizer.step()
            mean_losses["imagination_nll"] = mean_losses.get("imagination_nll", 0.0) + loss.detach() / num_updates
        self.foothold_storage.clear()
        return mean_losses, average_stats

    def state_dict(self):
        sd = super().state_dict()
        sd["imagination"] = self.imagination.state_dict()
        sd["imagination_optimizer"] = self.imagination_optimizer.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "imagination" in state_dict:
            self.imagination.load_state_dict(state_dict["imagination"])
        if "imagination_optimizer" in state_dict:
            self.imagination_optimizer.load_state_dict(state_dict["imagination_optimizer"])
