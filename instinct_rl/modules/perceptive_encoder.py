"""PerceptiveEncoder: CNN+FiLM+AttentionPool for depth history fused with dynamics."""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from instinct_rl.utils.utils import get_subobs_by_components, get_subobs_size


class FiLMBlock(nn.Module):
    """Feature-wise Linear Modulation: condition signal generates per-channel scale/bias."""

    def __init__(self, feature_channels: int, condition_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.generator = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, feature_channels * 2),
        )

    def forward(self, feature_map: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """Apply FiLM: gamma * x + beta.

        Args:
            feature_map: [B, C, H, W]
            condition: [B, cond_dim]
        Returns:
            [B, C, H, W]
        """
        gamma_beta = self.generator(condition)  # [B, C*2]
        gamma, beta = gamma_beta.chunk(2, dim=1)  # [B, C] each
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)
        beta = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * feature_map + beta


class SpatialAttentionPool(nn.Module):
    """Learned query attends over spatial positions, returns weighted feature vector."""

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, feature_dim))
        nn.init.normal_(self.query, std=0.02)
        self.key_conv = nn.Conv2d(feature_dim, hidden_dim, 1)
        self.attn_conv = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, feature_map: torch.Tensor) -> torch.Tensor:
        """Weighted spatial pooling.

        Args:
            feature_map: [B, C, H, W]
        Returns:
            [B, C]
        """
        B, C, H, W = feature_map.shape
        keys = self.key_conv(feature_map)  # [B, hidden, H, W]
        attn_logits = self.attn_conv(keys)  # [B, 1, H, W]
        attn = attn_logits.flatten(2).softmax(dim=-1)  # [B, 1, H*W]
        values = feature_map.flatten(2)  # [B, C, H*W]
        pooled = torch.bmm(values, attn.transpose(1, 2)).squeeze(-1)  # [B, C]
        return pooled


class PerceptiveEncoder(nn.Module):
    """Terrain+dynamics fused encoder for RL policy.

    Replaces ParallelLayer as the encoder_class_name in EncoderActorCriticMixin.

    Pipeline:
        depth_history [B,T,H,W] → SmallCNN → FiLM(dynamics) → AttentionPool → TerrainMLP(128)
        dynamics_obs  [B,D]    → DynamicsMLP(128)
        concat → [B, 256]

    Block configs in encoder_configs:
        One block entry with class_name="PerceptiveEncoder" that lists depth and
        dynamics component names, plus CNN/FiLM/MLP hyperparameters.
    """

    def __init__(
        self,
        input_segments: Dict[str, tuple],
        block_configs: Dict[str, dict],
        sequential_idx: int = 0,
    ):
        super().__init__()
        self.input_segments = input_segments
        self._sequential_idx = sequential_idx

        if len(block_configs) != 1:
            raise ValueError(
                f"PerceptiveEncoder expects exactly one config block, got {len(block_configs)}"
            )
        cfg = deepcopy(next(iter(block_configs.values())))
        cfg.pop("class_name", None)

        depth_component_names = cfg.pop("depth_component_names")
        dynamics_component_names = cfg.pop("dynamics_component_names")
        self.depth_component_names = list(depth_component_names)
        self.dynamics_component_names = list(dynamics_component_names)

        depth_shape = input_segments[self.depth_component_names[0]]
        # depth_shape: (num_frames, H, W)
        if len(depth_shape) == 3:
            in_channels, H_in, W_in = int(depth_shape[0]), int(depth_shape[1]), int(depth_shape[2])
        else:
            raise ValueError(f"Expected depth_shape (T,H,W), got {depth_shape}")

        dynamics_dim = get_subobs_size(
            {k: input_segments[k] for k in self.dynamics_component_names}
        )

        # --- CNN config ---
        depth_channels: list = cfg.pop("depth_channels", [32, 64])
        depth_kernel_sizes: list = cfg.pop("depth_kernel_sizes", [3, 3])
        depth_strides: list = cfg.pop("depth_strides", [2, 2])
        depth_paddings: list = cfg.pop("depth_paddings", [1, 1])

        # --- FiLM config ---
        film_hidden_dim: int = cfg.pop("film_hidden_dim", 64)

        # --- Attention pool config ---
        attn_hidden_dim: int = cfg.pop("attn_hidden_dim", 64)

        # --- MLP configs ---
        terrain_mlp_hidden: list = cfg.pop("terrain_mlp_hidden", [128])
        dynamics_mlp_hidden: list = cfg.pop("dynamics_mlp_hidden", [256, 128])
        output_size: int = cfg.pop("output_size", 256)
        nonlinearity: str = cfg.pop("nonlinearity", "ReLU")
        self._takeout_input_components: bool = cfg.pop("takeout_input_components", True)
        self._output_size = output_size

        if cfg:
            raise ValueError(f"Unrecognized PerceptiveEncoder config keys: {list(cfg.keys())}")

        activation = getattr(nn, nonlinearity)

        # ---- Build CNN ----
        cnn_layers = []
        prev_ch = in_channels
        H_out, W_out = H_in, W_in
        for oc, k, s, p in zip(depth_channels, depth_kernel_sizes, depth_strides, depth_paddings):
            cnn_layers.append(nn.Conv2d(prev_ch, oc, kernel_size=k, stride=s, padding=p))
            cnn_layers.append(activation(inplace=True))
            prev_ch = oc
            H_out = (H_out + 2 * p - k) // s + 1
            W_out = (W_out + 2 * p - k) // s + 1
        self.cnn = nn.Sequential(*cnn_layers)
        self._cnn_out_channels = depth_channels[-1]

        # ---- FiLM ----
        self.film = FiLMBlock(self._cnn_out_channels, dynamics_dim, film_hidden_dim)

        # ---- Spatial Attention Pool ----
        self.attn_pool = SpatialAttentionPool(self._cnn_out_channels, attn_hidden_dim)

        # ---- Terrain MLP ----
        t_mlp = []
        t_in = self._cnn_out_channels
        for h in terrain_mlp_hidden:
            t_mlp.append(nn.Linear(t_in, h))
            t_mlp.append(activation(inplace=True))
            t_in = h
        self.terrain_mlp = nn.Sequential(*t_mlp) if t_mlp else nn.Identity()
        self._terrain_out_dim = t_in

        # ---- Dynamics MLP ----
        d_mlp = []
        d_in = dynamics_dim
        for h in dynamics_mlp_hidden:
            d_mlp.append(nn.Linear(d_in, h))
            d_mlp.append(activation(inplace=True))
            d_in = h
        self.dynamics_mlp = nn.Sequential(*d_mlp) if d_mlp else nn.Identity()
        self._dynamics_out_dim = d_in

        # ---- Build output segment ----
        assert self._terrain_out_dim + self._dynamics_out_dim == output_size, (
            f"terrain({self._terrain_out_dim}) + dynamics({self._dynamics_out_dim}) "
            f"!= output_size({output_size})"
        )

        if self._takeout_input_components:
            self.output_segment = deepcopy(self.input_segments)
            for name in self.depth_component_names + self.dynamics_component_names:
                self.output_segment.pop(name, None)
        else:
            self.output_segment = deepcopy(self.input_segments)

        latent_name = f"parallel_latent_{self._sequential_idx}_perceptive"
        self.output_segment[latent_name] = (output_size,)
        self._latent_name = latent_name
        self.numel_output = get_subobs_size(self.output_segment)

    @property
    def output_size(self):
        return self._output_size

    def forward(self, flat_input: torch.Tensor) -> torch.Tensor:
        B = flat_input.shape[0]

        # --- Depth branch ---
        depth_flat = get_subobs_by_components(
            flat_input, self.depth_component_names, self.input_segments
        )
        depth_shape = self.input_segments[self.depth_component_names[0]]
        if len(depth_shape) == 3:
            T, H, W = int(depth_shape[0]), int(depth_shape[1]), int(depth_shape[2])
        else:
            C, H, W = int(depth_shape[0]), int(depth_shape[1]), int(depth_shape[2])
            T = C
        depth_batch = depth_flat.reshape(B, T, H, W)

        feature_map = self.cnn(depth_batch)  # [B, C', H', W']

        # --- FiLM with dynamics ---
        dynamics_flat = get_subobs_by_components(
            flat_input, self.dynamics_component_names, self.input_segments
        )
        feature_map = self.film(feature_map, dynamics_flat)

        # --- Attention pool ---
        terrain_features = self.attn_pool(feature_map)  # [B, C']
        terrain_embedding = self.terrain_mlp(terrain_features)  # [B, terrain_out]

        # --- Dynamics branch ---
        dynamics_embedding = self.dynamics_mlp(dynamics_flat)  # [B, dynamics_out]

        # --- Fuse ---
        fused = torch.cat([terrain_embedding, dynamics_embedding], dim=-1)  # [B, output_size]

        # --- Build output matching output_segment ---
        leading_dim = flat_input.shape[:-1]
        outputs = []
        for name, shape in self.output_segment.items():
            if name == self._latent_name:
                outputs.append(fused.reshape(*leading_dim, -1))
            else:
                outputs.append(
                    get_subobs_by_components(flat_input, [name], self.input_segments).reshape(
                        *leading_dim, -1
                    )
                )
        return torch.cat(outputs, dim=-1)

    def __str__(self):
        return (
            f"PerceptiveEncoder("
            f"cnn_out={self._cnn_out_channels}, "
            f"terrain_out={self._terrain_out_dim}, "
            f"dynamics_out={self._dynamics_out_dim})"
        )
