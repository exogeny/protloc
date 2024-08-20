from typing_extensions import Any, Dict, Optional
from functools import partial

import jax.numpy as jnp
from flax import linen as nn
from cipf import nn_layers
from cipf import utils

class Attention(nn.Module):
  dim: int
  num_heads: int
  qkv_bias: bool = False
  qk_norm: bool = False
  attn_drop_rate: float = 0.0
  proj_drop_rate: float = 0.0
  norm_layer: Any = nn.LayerNorm
  fused_attn: bool = False

  def setup(self):
    self.head_dim = self.dim // self.num_heads
    self.scale = self.head_dim ** -0.5
    self.qkv = nn.Dense(self.dim * 3, use_bias=self.qkv_bias)
    self.q_norm = self.norm_layer()
    self.k_norm = self.norm_layer()
    self.proj = nn.Dense(self.dim)
    self.proj_drop = nn.Dropout(rate=self.proj_drop_rate)
    if not self.fused_attn:
      self.attn_drop = nn.Dropout(rate=self.attn_drop_rate)

  @nn.compact
  def __call__(self, x, **kwargs):
    training = kwargs.get('training', False)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).transpose(
        (2, 0, 3, 1, 4))
    q, k, v = jnp.split(qkv, 3, axis=0)
    q, k, v = jnp.squeeze(q, axis=0), jnp.squeeze(k, axis=0), jnp.squeeze(v, axis=0)
    q, k = self.q_norm(q), self.k_norm(k)

    if self.fused_attn:
      x = nn_layers.scaled_dot_product_attention(
          q, k, v,
          dropout_rate=self.attn_drop_rate if training else 0.,
          training=training,
      )
    else:
      q = q * self.scale
      attn = q @ k.transpose((0, 1, 3, 2))
      attn = nn.activation.softmax(attn, axis=-1)
      attn = self.attn_drop(attn, deterministic=not training)
      x = attn @ v

    x = x.transpose((0, 2, 1, 3)).reshape(B, N, C)
    x = self.proj(x)
    x = self.proj_drop(x, deterministic=not training)
    return x


class LayerScale(nn.Module):
  dim: int
  init_values: float = 1e-5

  def setup(self):
    self.scale = self.param('scale',
                            nn.initializers.constant(self.init_values),
                            (self.dim))

  def __call__(self, x, **kwargs):
    return x * self.scale.value


class Block(nn.Module):
  dim: int
  num_heads: int
  mlp_ratio: float = 4.0
  qkv_bias: bool = False
  qk_norm: bool = False
  proj_drop_rate: float = 0.0
  attn_drop_rate: float = 0.0
  init_values: Optional[float] = None
  drop_path: float = 0.0
  act_layer: Any = nn.gelu
  norm_layer: Any = nn.LayerNorm
  mlp_layer: Any = nn_layers.Mlp

  def setup(self):
    self.attn = Attention(
        dim=self.dim,
        num_heads=self.num_heads,
        qkv_bias=self.qkv_bias,
        qk_norm=self.qk_norm,
        attn_drop_rate=self.attn_drop_rate,
        proj_drop_rate=self.proj_drop_rate,
        norm_layer=self.norm_layer,
    )
    self.norm1 = self.norm_layer()
    self.norm2 = self.norm_layer()
    self.mlp = self.mlp_layer(
        int(self.dim * self.mlp_ratio),
        self.dim,
        act_layer=self.act_layer,
        drop_rate=self.proj_drop_rate,
    )
    if self.init_values is not None:
      self.ls1 = LayerScale(self.dim, init_values=self.init_values)
      self.ls2 = LayerScale(self.dim, init_values=self.init_values)
    else:
      self.ls1 = lambda x: x
      self.ls2 = lambda x: x
    if self.drop_path > 0.:
      self.drop_path1 = nn_layers.DropPath(self.drop_path)
      self.drop_path2 = nn_layers.DropPath(self.drop_path)
    else:
      self.drop_path1 = lambda x, **kwargs: x
      self.drop_path2 = lambda x, **kwargs: x

  @nn.compact
  def __call__(self, x, **kwargs):
    x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))), **kwargs)
    x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))), **kwargs)
    return x


class VisionTransformer(nn.Module):
  image_size: int = 224
  patch_size: int = 16
  embed_dim: int = 768
  depth: int = 12
  num_heads: int = 12
  mlp_ratio: float = 4.0
  qkv_bias: bool = True
  use_bias: bool = True
  use_cls_token: bool = True
  pos_drop_rate: float = 0.0
  norm_layer: Any = nn.LayerNorm

  def setup(self):
    self.patch_embed =  nn_layers.PatchEmbed(
        image_size=self.image_size,
        patch_size=self.patch_size,
        embed_dim=self.embed_dim,
        use_bias=self.use_bias,
        norm_layer=self.norm_layer,
    )
    num_patches = self.patch_embed.num_patches
    self.pos_embed = jnp.expand_dims(
        utils.get_2d_sincos_pos_embed(
            self.embed_dim,
            int(num_patches ** 0.5),
            cls_token=self.use_cls_token
        ), axis=0)
    self.blocks = [
        Block(
            dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            norm_layer=self.norm_layer,
        ) for _ in range(self.depth)
    ]
    self.norm = self.norm_layer()
    self.cls_token = self.param('cls_token',
                                nn.initializers.normal(stddev=0.02),
                                (1, 1, self.embed_dim))

  @nn.compact
  def __call__(self, batch: Dict[str, jnp.ndarray], **kwargs):
    x = batch['image']
    B = x.shape[0]

    x = self.patch_embed(x, **kwargs)
    x = x + self.pos_embed[:, 1:, :]
    if self.use_cls_token:
      # append cls token
      cls_token = self.cls_token + self.pos_embed[:, :1, :]
      cls_token = cls_token.repeat(x.shape[0], axis=0)
      x = jnp.concatenate([cls_token, x], axis=1)

    for blk in self.blocks:
      x = blk(x)

    return {
      'latent': self.norm(x),
    }

  @staticmethod
  def metrics_fn(attributes, outputs, batch):
    pass


def vit_base_patch16(**kwargs):
  model = VisionTransformer(
      patch_size=16,
      embed_dim=768,
      depth=12,
      num_heads=12,
      mlp_ratio=4,
      qkv_bias=True,
      norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
      **kwargs,
  )
  return model


def vit_large_patch16(**kwargs):
  model = VisionTransformer(
      patch_size=16,
      embed_dim=768,
      depth=24,
      num_heads=12,
      mlp_ratio=4,
      qkv_bias=True,
      norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
      **kwargs,
  )
  return model


def vit_huge_patch14(**kwargs):
  model = VisionTransformer(
      patch_size=14,
      embed_dim=1280,
      depth=32,
      num_heads=16,
      mlp_ratio=4,
      qkv_bias=True,
      norm_layer=partial(nn.LayerNorm, epsilon=1e-6),
      **kwargs,
  )
  return model
