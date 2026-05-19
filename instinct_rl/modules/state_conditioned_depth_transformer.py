from __future__ import annotations

from typing import Any, Sequence

import torch
import torch.nn as nn

from .mlp import MlpModel


class StateConditionedDepthTransformerHeadModel(torch.nn.Module):
  """Depth: (B,C,H,W) → (B, H*W, C) → ``Linear``(C, d_model) for multi-token K/V; then cross-attn and head.

  Each spatial position becomes one token with a shared ``nn.Linear`` on the channel stack (equivalent to
  1×1 conv over C). State MLP is Q; ``nn.MultiheadAttention`` over the spatial token sequence; then
  ``concat(latent, raw state)`` through the output MLP.

  Optional **spatial** downsampling: ``AvgPool2d`` with ``kernel_size = stride = spatial_pool_size`` is
  applied to ``(B, C, H, W)`` before flattening, so the token count becomes ``(H/k)*(W/k)`` with
  ``k = spatial_pool_size`` (1 means no pool).

  Config fields ``num_layers`` / ``dim_feedforward`` / ``activation`` are reserved; legacy keys (e.g. old
  ``channels`` / ``frame_hidden_sizes``) are accepted via ``**kwargs`` and ignored for checkpoint/config
  compatibility.
  """

  def __init__(
    self,
    input_shapes: Sequence[torch.Size],
    output_size: int,
    state_hidden_sizes: Sequence[int] = (),
    d_model: int = 128,
    num_heads: int = 4,
    num_layers: int = 1,
    dim_feedforward: int = 256,
    dropout: float = 0.1,
    activation: str = "relu",
    nonlinearity: str = "ReLU",
    spatial_pool_size: int = 1,
    output_hidden_sizes: Sequence[int] = (),
    **kwargs: Any,
  ) -> None:
    super().__init__()
    _ = (num_layers, dim_feedforward, activation, kwargs)

    if len(input_shapes) != 2:
      msg = "StateConditionedDepthTransformerHeadModel expects two input shapes: depth and state."
      raise ValueError(msg)
    visual_shape, state_shape = input_shapes
    if len(visual_shape) != 3:
      raise ValueError(f"visual input shape must be (C, H, W); got {tuple(visual_shape)}")
    if len(state_shape) != 1:
      raise ValueError(f"state input shape must be flat; got {tuple(state_shape)}")

    self._c, self._h, self._w = (int(v) for v in visual_shape)
    self._state_dim = int(state_shape[0])
    mlp_nonlinearity = getattr(nn, nonlinearity) if isinstance(nonlinearity, str) else nonlinearity

    if d_model % num_heads != 0:
      raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
    if spatial_pool_size < 1:
      raise ValueError(f"spatial_pool_size must be >= 1; got {spatial_pool_size}")
    self._spatial_pool_size = int(spatial_pool_size)
    self._spatial_pool: nn.AvgPool2d | None
    if self._spatial_pool_size > 1:
      k = self._spatial_pool_size
      self._spatial_pool = nn.AvgPool2d(kernel_size=k, stride=k)
    else:
      self._spatial_pool = None

    self._depth_proj = nn.Linear(self._c, d_model)
    self._state_head = MlpModel(
        input_size=self._state_dim,
        hidden_sizes=list(state_hidden_sizes),
        output_size=d_model,
        nonlinearity=mlp_nonlinearity,
    )
    self._d_model = d_model
    self._cross_attn = nn.MultiheadAttention(
        embed_dim=d_model,
        num_heads=num_heads,
        dropout=dropout,
        batch_first=True,
    )
    self._output_head = MlpModel(
        input_size=d_model + self._state_dim,
        hidden_sizes=list(output_hidden_sizes),
        output_size=output_size,
        nonlinearity=mlp_nonlinearity,
    )

  def _encode_depth_tokens(self, depth: torch.Tensor) -> torch.Tensor:
    if depth.dim() != 4:
      msg = f"expected depth input shape (B, C, H, W); got shape {tuple(depth.shape)}"
      raise ValueError(msg)
    b, c, h, w = depth.shape
    if (c, h, w) != (self._c, self._h, self._w):
      msg = (
        f"depth input must match configured shape ({self._c}, {self._h}, {self._w}); "
        f"got ({c}, {h}, {w})"
      )
      raise ValueError(msg)
    x = self._spatial_pool(depth) if self._spatial_pool is not None else depth
    b2, c2, h2, w2 = x.shape
    seq = x.view(b2, c2, h2 * w2).transpose(1, 2).contiguous()
    return self._depth_proj(seq)

  def forward(self, depth: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    if state.dim() != 2 or state.shape[-1] != self._state_dim:
      msg = f"expected state input shape (B, {self._state_dim}); got {tuple(state.shape)}"
      raise ValueError(msg)
    depth_tok = self._encode_depth_tokens(depth)
    q = self._state_head(state).unsqueeze(1)
    latent, _ = self._cross_attn(q, depth_tok, depth_tok, need_weights=False)
    latent = latent.squeeze(1)
    return self._output_head(torch.cat([latent, state], dim=-1))
