from typing import List, Optional
import jax.numpy as jnp
from flax import linen as nn


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
  """
  grid_size: int of the grid height and width
  return:
  pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
  """
  grid_h = jnp.arange(grid_size, dtype=jnp.float32)
  grid_w = jnp.arange(grid_size, dtype=jnp.float32)
  grid = jnp.meshgrid(grid_w, grid_h)  # here w goes first
  grid = jnp.stack(grid, axis=0)

  grid = grid.reshape([2, 1, grid_size, grid_size])
  pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
  if cls_token:
    pos_embed = jnp.concatenate([jnp.zeros([1, embed_dim]), pos_embed], axis=0)
  return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
  assert embed_dim % 2 == 0

  # use half of dimensions to encode grid_h
  emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
  emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

  emb = jnp.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
  return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
  """
  embed_dim: output dimension for each position
  pos: a list of positions to be encoded: size (M,)
  out: (M, D)
  """
  assert embed_dim % 2 == 0
  omega = jnp.arange(embed_dim // 2, dtype=jnp.float32)
  omega /= embed_dim / 2.
  omega = 1. / 10000**omega  # (D/2,)

  pos = pos.reshape(-1)  # (M,)
  out = jnp.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

  emb_sin = jnp.sin(out) # (M, D/2)
  emb_cos = jnp.cos(out) # (M, D/2)

  emb = jnp.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
  return emb


def build_fourier_pos_embed(feat_shape: List[int],
                            bands: Optional[jnp.ndarray] = None,
                            num_bands: int = 64,
                            max_res: int = 224,
                            temperature: int = 10000,
                            linear_bands: bool = False,
                            include_grid: bool = False,
                            in_pixels: bool = True,
                            ref_feat_shape: Optional[List[int]] = None
) -> List[jnp.ndarray]:
  if bands is None:
    if in_pixels:
      bands = pixel_freq_bands(num_bands, float(max_res), linear_bands=linear_bands)
    else:
      bands = freq_bands(num_bands, temperature=temperature, step=1)
  if in_pixels:
    t = [jnp.linspace(-1., 1., s, dtype=jnp.float32) for s in feat_shape]
  else:
    t = [jnp.arange(s, dtype=jnp.float32) for s in feat_shape]

  if ref_feat_shape is not None:
    t = [x / f * r for x, f, r in zip(t, feat_shape, ref_feat_shape)]

  grid = jnp.stack(jnp.meshgrid(*t), axis=-1)
  grid = jnp.expand_dims(grid, axis=-1)
  pos = grid * bands

  pos_sin, pos_cos = jnp.sin(pos), jnp.cos(pos)
  out = [grid, pos_sin, pos_cos] if include_grid else [pos_sin, pos_cos]
  return out


def pixel_freq_bands(num_bands: int,
                     max_freq: float = 224.,
                     linear_bands: bool = True):
  if linear_bands:
    bands = jnp.linspace(1.0, max_freq / 2, num_bands, dtype=jnp.float32)
  else:
    bands = 2 ** jnp.linspace(0.0, jnp.log(max_freq / 2) - 1, num_bands, dtype=jnp.float32)
  return bands * jnp.pi


def freq_bands(num_bands: int,
               temperature: float = 10000,
               step: int = 2):
  exp = jnp.arange(0, num_bands, step, dtype=jnp.float32) / num_bands
  bands = 1. / (temperature ** exp)
  return bands


def rot(x: jnp.ndarray):
  return jnp.stack([-x[..., 1::2], x[..., ::2]], axis=-1).reshape(x.shape)


def apply_rot_embed_cat(x: jnp.ndarray, emb):
  sin_emb, cos_emb = jnp.split(emb, 2, axis=-1)
  return x * cos_emb + rot(x) * sin_emb


def apply_keep_indices_nlc(x: jnp.ndarray, pos_embed, keep_indices):
  keep_indices = jnp.expand_dims(keep_indices, axis=-1)
  pos_embed = jnp.expand_dims(pos_embed, axis=0)
  pos_embed = jnp.broadcast_to(pos_embed, (x.shape[0], -1, -1))
  pos_embed = jnp.take(1, keep_indices)
  return pos_embed


def build_rotary_pos_embed(feat_shape: List[int],
                           bands: Optional[jnp.ndarray] = None,
                           embed_dim: int = 64,
                           max_res: int = 224,
                           temperature: int = 10000,
                           linear_bands: bool = False,
                           in_pixels: bool = True,
                           ref_feat_shape: Optional[List[int]] = None):
  sin_emb, cos_emb = build_fourier_pos_embed(
      feat_shape,
      bands=bands,
      num_bands=embed_dim // 4,
      max_res=max_res,
      temperature=temperature,
      linear_bands=linear_bands,
      in_pixels=in_pixels,
      ref_feat_shape=ref_feat_shape,
  )
  num_spatial_dim = 1
  for x in feat_shape:
    num_spatial_dim *= x
  sin_emb = jnp.reshape(sin_emb, (num_spatial_dim, -1)).repeat(2, axis=-1)
  cos_emb = jnp.reshape(cos_emb, (num_spatial_dim, -1)).repeat(2, axis=-1)
  return sin_emb, cos_emb
