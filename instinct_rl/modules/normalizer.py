# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright (c) 2020 Preferred Networks, Inc.

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None, clip_obs: float | None = 10.0):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
            clip_obs (float or None): After normalization, clamp outputs to [-clip_obs, clip_obs]. None disables.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.clip_obs = clip_obs
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def _valid_rows(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 1:
            return torch.isfinite(x).all(dim=-1)
        return torch.isfinite(x)

    def _recover_stats_if_needed(self) -> None:
        if (
            torch.isfinite(self._mean).all()
            and torch.isfinite(self._var).all()
            and torch.isfinite(self._std).all()
        ):
            return
        self._mean.zero_()
        self._var.fill_(1.0)
        self._std.fill_(1.0)
        self.count.zero_()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """
        # Do not call _recover_stats_if_needed() here: its data-dependent
        # torch.isfinite().all() branch breaks torch.export / ONNX. Training
        # already recovers inside update(); inference sanitizes via nan_to_num.
        if self.training:
            with torch.no_grad():
                self.update(x)

        mean = torch.nan_to_num(self._mean, nan=0.0)
        std = torch.nan_to_num(self._std, nan=1.0).clamp_min(self.eps)
        out = (x - mean) / (std + self.eps)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        # Hard-clip normalized obs so extreme physics spikes cannot drive actor/critic to Inf/NaN.
        if self.clip_obs is not None:
            out = out.clamp(-self.clip_obs, self.clip_obs)
        return out

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return

        self._recover_stats_if_needed()

        valid_mask = self._valid_rows(x)
        x = x[valid_mask]
        if x.shape[0] == 0:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var.clamp_min(0.0))

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean

    def export(self, path):
        np.savez(
            path,
            mean=self._mean.cpu().numpy(),
            std=self._std.cpu().numpy(),
            eps=self.eps,
            until=self.until,
            clip_obs=self.clip_obs if self.clip_obs is not None else np.nan,
        )


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize
    the scale of the rewards so that the value function can learn quickly. We did this by dividing
    the rewards by a running estimate of the standard deviation of the sum of discounted rewards.
    """

    def __init__(self, shape, eps=1e-2, gamma=0.99, until=None):
        super().__init__()

        # Reward path should not inherit obs clip.
        self.emp_norm = EmpiricalNormalization(shape, eps, until, clip_obs=None)
        self.disc_avg = DiscountedAverage(gamma)

    def forward(self, rew):
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        if self.training:
            # update discounected rewards
            avg = self.disc_avg.update(rew)

            # update moments from discounted rewards
            self.emp_norm.update(avg)

        std = torch.nan_to_num(self.emp_norm._std, nan=1.0).clamp_min(self.emp_norm.eps)
        if std > 0:
            return rew / std
        return rew


class DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t

    Args:
        gamma (float): Discount factor.
    """

    def __init__(self, gamma):
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        rew = torch.nan_to_num(rew, nan=0.0, posinf=0.0, neginf=0.0)
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        self.avg = torch.nan_to_num(self.avg, nan=0.0, posinf=0.0, neginf=0.0)
        return self.avg
