""" MLP module w/ dropout and configurable activation layer

Hacked together by / Copyright 2020 Ross Wightman
"""
from typing import Any
from functools import partial
import jax.numpy as jnp
from flax import linen as nn
from cipf import utils


class Mlp(nn.Module):
  hidden_dim: int
  output_dim: int
  drop_rate: float = 0.0
  use_conv: bool = False
  use_bias: bool = True
  act_layer: Any = nn.silu
  norm_layer: Any = nn.LayerNorm

  def setup(self):
    linear_layer = partial(nn.Conv, kernel_size=(1, 1)) if self.use_conv else nn.Dense
    self.fc1 = linear_layer(self.hidden_dim, use_bias=self.use_bias)
    self.drop1 = nn.Dropout(rate=self.drop_rate)
    self.norm = self.norm_layer() if self.norm_layer is not None else None
    self.fc2 = linear_layer(self.output_dim, use_bias=self.use_bias)
    self.drop2 = nn.Dropout(rate=self.drop_rate)

  @nn.compact
  def __call__(self, x, **kwargs):
    training = kwargs.get('training', False)
    x = self.fc1(x)
    x = self.act_layer(x)
    x = self.drop1(x, deterministic=not training)
    x = self.norm(x) if self.norm is not None else x
    x = self.fc2(x)
    x = self.drop2(x, deterministic=not training)
    return x


class GluMlp(nn.Module):
  """ MLP w/ GLU style gating
  See: https://arxiv.org/abs/1612.08083, https://arxiv.org/abs/2002.05202
  """
  hidden_dim: int
  output_dim: int
  drop_rate: float = 0.0
  use_conv: bool = False
  use_bias: bool = True
  act_layer: Any = nn.silu
  norm_layer: Any = nn.LayerNorm
  gate_last: bool = False

  def setup(self):
    use_bias = utils.to_2tuple(self.use_bias)
    drop_rate = utils.to_2tuple(self.drop_rate)
    linear_layer = partial(nn.Conv, kernel_size=(1, 1)) if self.use_conv else nn.Dense
    self.chunk_dim = 1 if self.use_conv else -1

    self.fc1 = linear_layer(self.hidden_dim, use_bias=use_bias[0])
    self.act = self.act_layer
    self.drop1 = nn.Dropout(drop_rate[0])
    self.norm = self.norm_layer() if self.norm_layer is not None else None
    self.fc2 = linear_layer(self.output_dim, use_bias=use_bias[1])
    self.drop2 = nn.Dropout(drop_rate[1])

  @nn.compact
  def __call__(self, x, **kwargs):
    training = kwargs.get('training', False)
    x = self.fc1(x)
    x1, x2 = jnp.split(x, 2, axis=self.chunk_dim)
    x = x1 * self.act(x2) if self.gate_last else self.act(x1) * x2
    x = self.drop1(x, deterministic=not training)
    x = self.norm(x) if self.norm is not None else x
    x = self.fc2(x)
    x = self.drop2(x, deterministic=not training)
    return x


class SwiGluMlp(nn.Module):
  hidden_dim: int
  output_dim: int
  drop_rate: float = 0.0
  use_bias: bool = True
  act_layer: Any = nn.silu
  norm_layer: Any = nn.LayerNorm

  def setup(self):
    use_bias = utils.to_2tuple(self.use_bias)
    drop_rate = utils.to_2tuple(self.drop_rate)

    self.fc1_g = nn.Dense(self.hidden_dim, use_bias=use_bias[0],
                          kernel_init=nn.initializers.normal(stddev=1e-6),
                          bias_init=nn.initializers.ones)
    self.fc1_x = nn.Dense(self.hidden_dim, use_bias=use_bias[0])
    self.act = self.act_layer
    self.drop1 = nn.Dropout(drop_rate[0])
    if self.norm_layer is not None:
      self.norm = self.norm_layer()
    self.fc2 = nn.Dense(self.output_dim, use_bias=use_bias[1])
    self.drop2 = nn.Dropout(drop_rate[1])

  @nn.compact
  def __call__(self, x, **kwargs):
    training = kwargs.get('training', False)
    x_gate = self.fc1_g(x)
    x = self.fc1_x(x)
    x = self.act(x_gate) * x
    x = self.drop1(x, deterministic=not training)
    x = self.norm(x) if self.norm_layer is not None else x
    x = self.fc2(x)
    x = self.drop2(x, deterministic=not training)
    return x
