from typing_extensions import Optional, Callable

import jax.numpy as jnp
from flax import linen as nn
from cipf import utils

class PatchEmbed(nn.Module):
  image_size: Optional[int] = 128
  patch_size: int = 16
  embed_dim: int = 768
  flatten: bool = True
  use_bias: bool = True
  norm_layer: Callable = nn.LayerNorm

  def setup(self):
    self.proj = nn.Conv(
        features=self.embed_dim,
        kernel_size=(self.patch_size, self.patch_size),
        strides=(self.patch_size, self.patch_size),
        use_bias=self.use_bias,
    )
    self.norm = self.norm_layer()

  @nn.compact
  def __call__(self, x, **kwargs):
    B, H, W, _ = x.shape
    hpad = (self.patch_size - H % self.patch_size) % self.patch_size
    wpad = (self.patch_size - W % self.patch_size) % self.patch_size
    x = jnp.pad(x, [(0, 0), (hpad, hpad), (wpad, wpad), (0, 0)])
    x = self.proj(x)
    x = self.norm(x)
    if self.flatten:
      x = jnp.reshape(x, (B, -1, self.embed_dim)) # NHWC -> NLC
    return x

  @property
  def num_patches(self):
    return self.grid_size[0] * self.grid_size[1]

  @property
  def grid_size(self):
    image_size = utils.to_2tuple(self.image_size)
    patch_size = utils.to_2tuple(self.patch_size)
    return tuple([s // p for s, p in zip(image_size, patch_size)])
