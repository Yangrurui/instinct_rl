"""DepthProprioRecurrentGruEncoder: depth CNN + single-frame proprio with cross-step GRU memory."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from instinct_rl.modules.actor_critic_recurrent import Memory
from instinct_rl.modules.conv2d import Conv2dModel
from instinct_rl.modules.mlp import MlpModel


class DepthProprioRecurrentGruEncoder(nn.Module):
  """Encode latest depth + current proprio with episode-persistent GRU temporal fusion.

  Unlike DepthProprioGruEncoder, proprio is a single frame and temporal context is
  carried by Memory (cross-step hidden state), not an input history window.
  """

  is_recurrent = True

  def __init__(
    self,
    depth_shape: Sequence[int],
    proprio_dim: int,
    depth_channels: List[int],
    depth_kernel_sizes: List[int],
    depth_strides: List[int],
    depth_paddings: List[int],
    depth_embedding_dim: int,
    proprio_hidden_sizes: List[int],
    proprio_embedding_dim: int,
    gru_hidden_dim: int,
    gru_num_layers: int,
    fusion_hidden_sizes: List[int],
    output_size: int,
    nonlinearity: str = "ELU",
  ):
    super().__init__()
    if isinstance(nonlinearity, str):
      nonlinearity_cls = getattr(nn, nonlinearity)
    else:
      nonlinearity_cls = nonlinearity

    if len(depth_shape) == 3:
      in_channels, height, width = int(depth_shape[0]), int(depth_shape[1]), int(depth_shape[2])
    elif len(depth_shape) == 2:
      in_channels, height, width = 1, int(depth_shape[0]), int(depth_shape[1])
    else:
      raise ValueError(f"Expected depth_shape (T,H,W) or (H,W), got {depth_shape}")

    self.depth_embedding_dim = int(depth_embedding_dim)
    self.proprio_embedding_dim = int(proprio_embedding_dim)
    self.gru_hidden_dim = int(gru_hidden_dim)

    self.depth_cnn = Conv2dModel(
      in_channels=in_channels,
      channels=depth_channels,
      kernel_sizes=depth_kernel_sizes,
      strides=depth_strides,
      paddings=depth_paddings,
      nonlinearity=nonlinearity_cls,
      use_maxpool=False,
    )
    conv_out_size = self.depth_cnn.conv_out_size(height, width)
    self.depth_head = MlpModel(
      conv_out_size,
      hidden_sizes=[],
      output_size=depth_embedding_dim,
      nonlinearity=nonlinearity_cls,
    )

    proprio_mlp_hidden = list(proprio_hidden_sizes) + [proprio_embedding_dim]
    self.proprio_mlp = MlpModel(
      proprio_dim,
      hidden_sizes=proprio_mlp_hidden[:-1],
      output_size=proprio_mlp_hidden[-1],
      nonlinearity=nonlinearity_cls,
    )

    gru_input_dim = depth_embedding_dim + proprio_embedding_dim
    self.temporal = Memory(
      input_size=gru_input_dim,
      type="gru",
      num_layers=gru_num_layers,
      hidden_size=gru_hidden_dim,
    )

    fusion_hidden = list(fusion_hidden_sizes) + [output_size]
    self.fusion_mlp = MlpModel(
      gru_hidden_dim,
      hidden_sizes=fusion_hidden[:-1],
      output_size=fusion_hidden[-1],
      nonlinearity=nonlinearity_cls,
    )
    self._output_size = output_size

  @property
  def output_size(self) -> int:
    return self._output_size

  @property
  def hidden_states(self):
    return self.temporal.hidden_states

  def reset(self, dones=None):
    self.temporal.reset(dones)

  def _reshape_depth_for_cnn(self, depth: torch.Tensor) -> torch.Tensor:
    if depth.dim() == 2:
      batch_size = depth.shape[0]
      depth = depth.reshape(batch_size, 1, -1)
      side = int(depth.shape[-1] ** 0.5)
      depth = depth.reshape(batch_size, 1, side, side)
    elif depth.dim() == 3:
      depth = depth.unsqueeze(1)
    elif depth.dim() == 4 and depth.shape[1] > 1:
      depth = depth[:, :1]
    return depth

  def _encode_depth(self, depth: torch.Tensor) -> torch.Tensor:
    if depth.dim() == 5:
      t_steps, batch_size = depth.shape[0], depth.shape[1]
      flat = depth.reshape(t_steps * batch_size, *depth.shape[2:])
      flat = self._reshape_depth_for_cnn(flat)
      depth_embed = self.depth_head(self.depth_cnn(flat).view(t_steps * batch_size, -1))
      return depth_embed.reshape(t_steps, batch_size, self.depth_embedding_dim)

    depth = self._reshape_depth_for_cnn(depth)
    batch_size = depth.shape[0]
    return self.depth_head(self.depth_cnn(depth).view(batch_size, -1))

  def _encode_proprio(self, proprio: torch.Tensor) -> torch.Tensor:
    if proprio.dim() == 3:
      t_steps, batch_size, proprio_dim = proprio.shape
      proprio_embed = self.proprio_mlp(proprio.reshape(t_steps * batch_size, proprio_dim))
      return proprio_embed.reshape(t_steps, batch_size, self.proprio_embedding_dim)
    return self.proprio_mlp(proprio)

  def forward(
    self,
    depth: torch.Tensor,
    proprio: torch.Tensor,
    masks=None,
    hidden_states=None,
  ) -> torch.Tensor:
    """Run depth CNN + proprio MLP + cross-step GRU fusion.

    Single-step (collection): hidden_states is None, inputs are [B, ...].
    Batch (policy update): hidden_states given, inputs are [T, B, ...]; masks must be None
    so Memory returns padded [T, B, H] for AC-level unpad.
    """
    depth_embed = self._encode_depth(depth)
    proprio_embed = self._encode_proprio(proprio)
    fused = torch.cat([proprio_embed, depth_embed], dim=-1)
    gru_out = self.temporal(fused, masks=masks, hidden_states=hidden_states)
    return self.fusion_mlp(gru_out)
