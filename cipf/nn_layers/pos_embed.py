from typing import Optional, List
import jax.numpy as jnp
from flax import linen as nn

from cipf import utils


class RotaryEmbeddingCat(nn.Module):
  """Rotary position embedding w/ concatenatd sin & cos

  The following impl/resources were referenced for this impl:
    * https://github.com/lucidrains/vit-pytorch/blob/6f3a5fcf0bca1c5ec33a35ef48d97213709df4ba/vit_pytorch/rvt.py
    * https://blog.eleuther.ai/rotary-embeddings/
    * https://github.com/huggingface/pytorch-image-models/blob/20fe56bd9072af61d9f5404ce8b08e24ff10a807/timm/layers/pos_embed_sincos.py#L362
  """
  embed_dim: int
  max_res: int = 224
  temperature: int = 10000
  in_pixels: bool = True
  linear_bands: bool = False
  feat_shape: Optional[List[int]] = None
  ref_feat_shape: Optional[List[int]] = None

  def setup(self):
    if self.feat_shape is None:
      if self.in_pixels:
        bands = utils.pixel_freq_bands(
            self.embed_dim // 4,
            float(self.max_res),
            linear_bands=self.linear_bands,
        )
      else:
        bands = utils.freq_bands(
            self.embed_dim // 4,
            temperature=self.temperature,
            step=1,
        )
      self.bands = bands
      self.pos_embed = None
    else:
      embeds = utils.build_rotary_pos_embed(
          feat_shape=self.feat_shape,
          embed_dim=self.embed_dim,
          max_res=self.max_res,
          linear_bands=self.linear_bands,
          in_pixels=self.in_pixels,
          ref_feat_shape=self.ref_feat_shape,
      )
      self.bands = None
      self.pos_embed = jnp.concatenate(embeds, axis=-1)

  def get_embed(self, shape: Optional[List[int]] = None):
    if self.bands is not None and shape is not None:
      embeds = utils.build_rotary_pos_embed(
          shape,
          self.bands,
          in_pixels=self.in_pixels,
          ref_feat_shape=self.ref_feat_shape,
      )
      return jnp.concatenate(embeds, axis=-1)
    elif self.pos_embed is not None:
      return self.pos_embed
    assert False, 'requires pre-computed pos_embed or valid shape w/ pre-computed bands'

  def __call__(self, x: jnp.ndarray):
    pos_embed = self.get_embed(x.shape[1:-1])
    return utils.apply_rot_embed_cat(x, pos_embed)
