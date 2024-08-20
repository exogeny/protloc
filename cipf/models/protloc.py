from typing import Any, Callable, Tuple
from functools import partial
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from cipf import nn_layers
from cipf import utils
from cipf.models import quantizer
from cipf.nn_layers import losses


def dt_bias_initializer(rng, shape, **kwargs):
  dt_min = kwargs.get('dt_min', 0.001)
  dt_max = kwargs.get('dt_max', 0.1)

  dt = jnp.exp(
      random.uniform(rng, shape=shape)
      * (jnp.log(dt_max) - jnp.log(dt_min))
      + jnp.log(dt_min)
  )
  dt = jnp.clip(dt, min=1e-4)
  inv_dt = dt + jnp.log(-jnp.expm1(-dt))
  return inv_dt


def a_initializer(rng, shape, **kwargs):
  a_min = kwargs.get('a_min', 1)
  a_max = kwargs.get('a_max', 16)
  a = random.uniform(rng, shape, minval=a_min, maxval=a_max)
  a_log = jnp.log(a)
  return a_log


class SSDBlock(nn.Module):
  dim: int
  num_heads: int
  block_len: int = 1
  has_cls_token: bool = False
  norm_layer: Any = nn.LayerNorm

  def setup(self):
    self.head_dim = self.dim // self.num_heads
    self.proj_dt = nn.Dense(self.num_heads, use_bias=False)
    self.proj = nn.Dense(self.dim, use_bias=False)
    self.norm = self.norm_layer()

  @nn.compact
  def __call__(self, X, B, C, **kwargs):
    batch, seqlen, dim = X.shape
    B = B.reshape(batch, seqlen, self.num_heads, -1)
    C = C.reshape(batch, seqlen, self.num_heads, -1)
    X = X.reshape(batch, seqlen, self.num_heads, -1)

    # initial states
    initial_states = kwargs.get('initial_states', None)

    # dt from learnable parameters and v
    dt_bias = self.param(
        'dt_bias',
        dt_bias_initializer,
        (1, seqlen, self.num_heads),
    )
    dt = self.proj_dt(X.reshape(batch, seqlen, -1))
    dt = nn.softplus(dt + dt_bias)

    # a_log from learnable parameters
    a_log = self.param('a_log', a_initializer, (self.num_heads, ))
    A = -jnp.exp(a_log)

    x, states = nn_layers.ssd_minimal_discrete(
        X * jnp.expand_dims(dt, axis=-1),
        A * dt, C, B,
        block_len=self.block_len,
        has_cls_token=self.has_cls_token,
        initial_states=initial_states,
    )

    x = x.reshape(batch, seqlen, dim)
    x = self.proj(self.norm(x))
    return x, states


class MambaBlock(nn.Module):
  embeding_dim: int
  state_dim: int
  num_heads: int = 8
  block_len: int = 8
  has_cls_token: bool = False
  mlp_ratio: float = 4.0
  use_bias: bool = False
  act_layer: Any = nn.silu
  norm_layer: Any = nn.LayerNorm
  mlp_layer: Any = nn_layers.Mlp

  def setup(self):
    self.ssd = SSDBlock(
        dim=self.embeding_dim,
        num_heads=self.num_heads,
        block_len=self.block_len,
        has_cls_token=self.has_cls_token,
        norm_layer=self.norm_layer,
    )
    self.qkv = nn.Dense(self.embeding_dim + self.state_dim * 2,
                        use_bias=self.use_bias)
    self.norm1 = self.norm_layer()
    self.norm2 = self.norm_layer()
    self.mlp = self.mlp_layer(
        int(self.embeding_dim * self.mlp_ratio),
        self.embeding_dim,
        use_bias=self.use_bias,
        act_layer=self.act_layer,
        norm_layer=self.norm_layer,
        drop_rate=0.0,
    )

  @nn.compact
  def __call__(self, x, **kwargs):
    shortcut = x
    x = self.qkv(self.norm1(x))
    b, c, x = jnp.split(x, [self.state_dim, self.state_dim * 2], axis=-1)
    x, states = self.ssd(X=x, B=b, C=c, **kwargs)
    x = shortcut + self.mlp(self.norm2(x), **kwargs)
    return x, states


class VectorQuantizer(nn.Module):
  embedding_dim: int
  embedding_rows: int
  embedding_cols: int
  commitment_cost: float
  dtype = jnp.float32

  def setup(self):
    self.embeddings = self.param(
        'embeddings',
        nn.initializers.variance_scaling(
            scale=1.0,
            mode='fan_in',
            distribution='uniform'
        ),
        # (self.embedding_dim, self.embedding_cols),
        (self.embedding_rows, self.embedding_cols),
    )
    self.cls_proj = nn.Dense(self.embedding_rows)

  @partial(jax.vmap, in_axes=(None, 0, 0))
  def quantize(self, query, embeddings):
    distance = (
        jnp.sum(query ** 2, axis=1, keepdims=True)
        - 2 * jnp.matmul(query, embeddings)
        + jnp.sum(embeddings ** 2, axis=0, keepdims=True))
    indices = jnp.argmax(-distance, axis=1)
    encodings = jax.nn.one_hot(indices, self.embedding_cols, dtype=query.dtype)
    embeddings = jnp.transpose(embeddings, (1, 0))

    quantized = jnp.take(embeddings, indices, axis=0)
    e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - query) ** 2)
    q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(query)) ** 2)

    quantized = query + jax.lax.stop_gradient(quantized - query)
    avg_probs = jnp.mean(encodings, axis=0)
    perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))
    return perplexity, quantized, e_latent_loss, q_latent_loss, indices

  @nn.compact
  def __call__(self, x, **kwargs):
    # get the embeddings from cls tokens
    cls_token = x[:, :1, :] # B 1 C
    pos_token = x[:, 1:, :] # B L C

    cls_matrix = jnp.matmul(jnp.transpose(cls_token, (0, 2, 1)), cls_token)
    cls_matrix = jax.lax.stop_gradient(cls_matrix)
    cls_proj = self.cls_proj(cls_matrix) # B C C -> B C N
    cls_proj = nn.softmax(cls_proj, axis=-1)
    embeddings = jnp.einsum('bcn,nm->bcm', cls_proj, self.embeddings)

    # get the embeddings from pos tokens
    (
      perplexity,
      quantized,
      e_latent_loss,
      q_latent_loss,
      pos_indices,
    ) = self.quantize(pos_token, embeddings)

    quantized = jnp.concatenate([cls_token, quantized], axis=1)
    e_latent_loss = e_latent_loss.mean()
    q_latent_loss = q_latent_loss.mean()
    perplexity = perplexity.mean()

    return {
      'quantized': quantized,
      'metrics/perplexity': perplexity,
      'metrics/e_latent_loss': e_latent_loss,
      'metrics/q_latent_loss': q_latent_loss,
      'metrics/embeddings@histogram': self.embeddings,
      'metrics/pos_indices@histogram': pos_indices,
      'metrics/quantized@histogram': quantized,
      'metrics/pos_token@histogram': pos_token,
      'metrics/cls_token@histogram': cls_token,
      'vq_loss': e_latent_loss * self.commitment_cost + q_latent_loss,
    }


class ProtLoc(nn.Module):
  image_size: int = 224
  patch_size: int = 7
  embedding_dim: int = 64
  state_dim: int = 64
  embedding_rows: int = 4096
  embedding_cols: int = 4096
  depth: int = 12
  num_heads: int = 8
  block_len: int = 8
  mlp_ratio: float = 4.
  commitment_cost: float = 0.25
  vq_mode: str = 'cls-vq'
  use_bias: bool = True
  vq_weight: float = 1.00
  affine_prob: float = 0.75
  cellular_reconstruction: bool = True
  dtype = jnp.float32

  def setup(self):
    use_bias = self.use_bias
    norm_layer = nn.RMSNorm

    self.act_layer = nn.silu
    self.p_patch_embed = nn_layers.PatchEmbed(
        image_size=self.image_size,
        patch_size=self.patch_size,
        embed_dim=self.embedding_dim,
        use_bias=use_bias,
        norm_layer=norm_layer,
    )
    self.c_patch_embed = nn_layers.PatchEmbed(
        image_size=self.image_size,
        patch_size=self.patch_size,
        embed_dim=self.embedding_dim,
        use_bias=use_bias,
        norm_layer=norm_layer,
    )

    self.p_blocks = [
        MambaBlock(
            embeding_dim=self.embedding_dim,
            state_dim=self.state_dim,
            num_heads=self.num_heads,
            block_len=self.block_len,
            has_cls_token=True,
            mlp_ratio=self.mlp_ratio,
            use_bias=use_bias,
            act_layer=self.act_layer,
            norm_layer=norm_layer,
        ) for _ in range(self.depth)
    ]
    self.c_blocks = [
        MambaBlock(
            embeding_dim=self.embedding_dim,
            state_dim=self.state_dim,
            num_heads=self.num_heads,
            block_len=self.block_len,
            has_cls_token=False,
            mlp_ratio=self.mlp_ratio,
            use_bias=use_bias,
            act_layer=self.act_layer,
            norm_layer=norm_layer,
        ) for _ in range(self.depth)
    ]
    self.m_layer = nn.Dense(self.embedding_dim, use_bias=use_bias)
    self.p_norm = norm_layer()

    self.p_decoder_embed = nn.Dense(
        self.embedding_dim,
        use_bias=use_bias,
    )

    self.decoder_blocks = [
        MambaBlock(
            embeding_dim=self.embedding_dim,
            state_dim=self.state_dim,
            num_heads=self.num_heads,
            block_len=self.block_len,
            has_cls_token=True,
            mlp_ratio=self.mlp_ratio,
            use_bias=use_bias,
            act_layer=self.act_layer,
            norm_layer=norm_layer,
        ) for _ in range(self.depth)
    ]

    self.p_decoder_norm = norm_layer()
    self.p_decoder_pred = nn.Dense(
        self.patch_size**2,
        use_bias=use_bias,
    )

    if self.cellular_reconstruction:
      self.c_norm = norm_layer()
      self.c_decoder_embed = nn.Dense(
          self.embedding_dim,
          use_bias=use_bias,
      )
        # decoder for cellular, 2 channels: nucleus and cell
      self.c_decoder_norm = norm_layer()
      self.c_decoder_pred = nn.Dense(
          self.patch_size**2 * 2,
          use_bias=use_bias,
      )

    if self.vq_mode == 'cls-vq':
      self.vq_layer = VectorQuantizer(
          embedding_dim=self.embedding_dim,
          embedding_rows=self.embedding_rows,
          embedding_cols=self.embedding_cols,
          commitment_cost=self.commitment_cost,
      )
    elif self.vq_mode == 'vq':
      self.vq_layer = quantizer.VectorQuantizer(
          embedding_dim=self.embedding_dim,
          num_embeddings=self.embedding_rows,
          commitment_cost=self.commitment_cost,
      )
    else:
      self.vq_layer = lambda x, **kwargs: {'quantized': x, 'vq_loss': 0.0}

    num_patches = self.p_patch_embed.num_patches
    self.pos_embed = jnp.expand_dims(
        utils.get_2d_sincos_pos_embed(
            self.embedding_dim,
            int(num_patches ** 0.5),
            cls_token=True,
        ), axis=0)
    self.cls_token = self.param(
        'cls_token',
        nn.initializers.normal(stddev=0.02),
        (1, 1, self.embedding_dim),
    )

  def empty_data(self, image_size):
    return {
      'image': jnp.empty(
          (1, self.image_size, self.image_size, 3),
          dtype=self.dtype,
      ),
    }

  @nn.compact
  def __call__(self, batch, **kwargs):
    rng = kwargs.get('mrng', random.PRNGKey(0))
    encode_only = kwargs.get('encode_only', False)
    affine_transform = kwargs.pop('affine_transform', None)
    if affine_transform is None:
      affine_transform = kwargs.get('training', False)

    # Prepare the input data
    image = batch['image']
    reference = batch.get('reference', None)
    B = image.shape[0]

    if reference is not None:
      protein1, nucleus1, cell1 = jnp.split(reference, 3, axis=-1)
    elif affine_transform:
      reference = utils.random_affine(image, rng, self.affine_prob)
      protein1, nucleus1, cell1 = jnp.split(reference, 3, axis=-1)
    else:
      protein1, nucleus1, cell1 = jnp.split(image, 3, axis=-1)

    contour1 = jnp.concatenate([nucleus1, cell1], axis=-1)

    px1 = self.p_patch_embed(protein1, **kwargs) # NLC
    cx1 = self.c_patch_embed(contour1, **kwargs)

    # add pos embed
    px1 = px1 + self.pos_embed[:, 1:, :]
    cx1 = cx1 + self.pos_embed[:, 1:, :]

    if not encode_only:
      protein2, nucleus2, cell2 = jnp.split(image, 3, axis=-1)
      contour2 = jnp.concatenate([nucleus2, cell2], axis=-1)
      cx2 = self.c_patch_embed(contour2, **kwargs)
      cx2 = cx2 + self.pos_embed[:, 1:, :]

    # add cls token, just add for protein channel
    cls_token = self.cls_token + self.pos_embed[:, :1, :]
    cls_token = cls_token.repeat(B, axis=0)
    zeros = jnp.zeros_like(cls_token)
    px1 = jnp.concatenate([cls_token, px1], axis=1)

    outputs = {}
    # Apply attention to get the values
    cx2_lists = []
    p1states, c1states, c2states = None, None, None
    for pblock, cblock in zip(self.p_blocks, self.c_blocks):
      px1, p1states = pblock(px1, initial_states=p1states, **kwargs)
      cx1, c1states = cblock(cx1, initial_states=c1states, **kwargs)
      if not encode_only:
        cx2, c2states = cblock(cx2, initial_states=c2states, **kwargs)
        cx2_lists.append(cx2)
      px1 = self.act_layer(px1 + jnp.concatenate([zeros, cx1], axis=1))

    px1 = self.m_layer(px1)
    # quantizer path
    output = self.vq_layer(px1, **kwargs)
    latent = output['quantized']
    outputs.update({
      'latent': latent,
      'losses/vq_loss': output['vq_loss'],
    })
    outputs.update({k: v for k, v in output.items() if 'metrics' in k})

    if not encode_only:
      # decoder path
      px2 = self.p_decoder_embed(self.p_norm(latent))
      px2 = px2 + self.pos_embed

      p2states = p1states
      for pblock, cx2 in zip(self.decoder_blocks, reversed(cx2_lists)):
        px2, p2states = pblock(px2, initial_states=p2states, **kwargs)
        px2 = self.act_layer(px2 + jnp.concatenate([zeros, cx2], axis=1))
      px2 = self.p_decoder_norm(px2)
      px2_pred = self.p_decoder_pred(px2)
      px2_pred = px2_pred[:, 1:, :]
      px2_pred = utils.unpatchify(px2_pred, self.patch_size)

      if self.cellular_reconstruction:
        c_latent = self.c_norm(cx2_lists[-1])
        cx2 = self.c_decoder_embed(c_latent)
        cx2 = cx2 + self.pos_embed[:, 1:, :] # no cls token
        cx2 = self.c_decoder_norm(cx2)
        cx2_pred = self.c_decoder_pred(cx2)
        cx2_pred = utils.unpatchify(cx2_pred, self.patch_size)
        outputs['cpred'] = cx2_pred

      outputs.update({
        'pimage': protein2,
        'cimage': contour2,
        'ppred': px2_pred,
      })
    return outputs

  @staticmethod
  def metrics_fn(attributes, outputs, batch):
    mask = batch.get('__mask', None)
    pimage = outputs['pimage']
    cimage = outputs['cimage']
    ppred = outputs['ppred']

    if attributes.get('cellular_reconstruction', False):
      cpred = outputs['cpred']
      crec_err = utils.masked_reduce_mean(
          losses.mean_squared_error, mask, cpred, cimage)
    else:
      crec_err = 0.0

    metrics = {k: v for k, v in outputs.items() if 'metrics' in k}

    prec_err = utils.masked_reduce_mean(
        losses.mean_squared_error, mask, ppred, pimage)

    vq_loss = outputs['losses/vq_loss']
    vq_weight = attributes.get('vq_weight', 1.0)

    ssim = losses.ssim(ppred, pimage)

    metrics.update({
      'losses/total_loss': prec_err + crec_err + vq_weight * vq_loss,
      'metrics/crec_err': crec_err,
      'metrics/prec_err': prec_err,
      'metrics/prec_ssim': ssim,
      'losses/vq_loss': vq_loss,
      'images/ppred@image': ppred[:3],
      'images/pimage@image': pimage[:3],
      'images/cimage@image': cimage[:3],
    })
    return metrics
