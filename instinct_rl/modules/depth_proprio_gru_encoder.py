"""DepthProprioGruEncoder: latest depth CNN + proprio history GRU fusion block."""

from __future__ import annotations

from typing import List, Sequence

import torch
import torch.nn as nn

from instinct_rl.modules.conv2d import Conv2dModel
from instinct_rl.modules.mlp import MlpModel


class DepthProprioGruEncoder(nn.Module):
  """Encode latest depth frame and proprioception history into a latent vector.

  Pipeline:
    depth [B,1,H,W] -> CNN -> depth_embed [B, D_d]
    proprio [B,T,D_p] -> per-frame MLP -> proprio_embed [B,T,D_p']
    repeat depth_embed over T -> concat with proprio_embed -> [B,T,D_d+D_p']
    GRU -> hidden [B, H] -> fusion MLP -> latent [B, output_size]
  """

  def __init__(
    self,
    depth_shape: Sequence[int],
    proprio_dim: int,
    proprio_seq_len: int,
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

    self.proprio_seq_len = int(proprio_seq_len)
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
    self.gru = nn.GRU(
      input_size=gru_input_dim,
      hidden_size=gru_hidden_dim,
      num_layers=gru_num_layers,
      batch_first=True,
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

  def forward(self, depth: torch.Tensor, proprio_history: torch.Tensor) -> torch.Tensor:
    """Run depth CNN + proprio GRU fusion.

    Args:
      depth: [B, C, H, W], [B, H, W], or [B, C*H*W] (latest frame only).
      proprio_history: [B, T, D] proprio sequence for GRU.
    """
    batch_size = proprio_history.shape[0]
    seq_len = proprio_history.shape[1]

    if depth.dim() == 2:
      depth = depth.reshape(batch_size, 1, -1)
      side = int(depth.shape[-1] ** 0.5)
      depth = depth.reshape(batch_size, 1, side, side)
    elif depth.dim() == 3:
      depth = depth.unsqueeze(1)
    elif depth.dim() == 4 and depth.shape[1] > 1:
      depth = depth[:, :1]

    depth_embed = self.depth_head(self.depth_cnn(depth).view(batch_size, -1))

    proprio_embed = self.proprio_mlp(
      proprio_history.reshape(batch_size * seq_len, -1)
    ).reshape(batch_size, seq_len, self.proprio_embedding_dim)

    depth_repeated = depth_embed.unsqueeze(1).expand(-1, seq_len, -1)
    gru_input = torch.cat([proprio_embed, depth_repeated], dim=-1)
    _, hidden = self.gru(gru_input)
    gru_out = hidden[-1]
    return self.fusion_mlp(gru_out)
