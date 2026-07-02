"""落脚点想象网络：privileged state + action -> 每脚未来接触点高斯 (mu_xy, log_sigma)。

对应 SSR Table 8 的 Foothold Imagination Model（MLP [256,128]）。它不属于 actor_critic，
由 FootholdAlgoMixin 持有并用独立优化器训练（与 AMP discriminator 同构）。
"""

import torch
import torch.nn as nn

from instinct_rl.modules.mlp import MlpModel


class FootholdImaginationModel(nn.Module):
    def __init__(self, input_size: int, hidden_sizes=(256, 128), nonlinearity: str = "ELU",
                 log_sigma_min: float = -4.0, log_sigma_max: float = 1.0):
        super().__init__()
        self.net = MlpModel(
            input_size=input_size,
            output_size=2 * 3,  # 每脚 (mu_x, mu_y, log_sigma)
            hidden_sizes=list(hidden_sizes),
            nonlinearity=nonlinearity,
        )
        self.log_sigma_min = log_sigma_min
        self.log_sigma_max = log_sigma_max

    def forward(self, x: torch.Tensor):
        out = self.net(x).view(-1, 2, 3)
        mu = out[..., :2]
        log_sigma = out[..., 2].clamp(self.log_sigma_min, self.log_sigma_max)
        return mu, log_sigma
