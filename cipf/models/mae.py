import numpy as np
import jax
import jax.numpy as jnp
from jax import random
from flax import linen as nn

from cipf import nn_layers
from cipf import utils
from cipf.models import vit
from cipf.nn_layers import losses

class MaskedAutoEncoder(nn.Module):
  """A simple protein map model."""
  image_size: int = 128
  num_channels: int = 3
  patch_size: int = 16
  embed_dim: int = 1024
  depth: int = 24
  num_heads: int = 16
  decoder_embed_dim: int = 512
  decoder_depth: int = 8
  decoder_num_heads: int = 16
  mlp_ratio: float = 4.
  norm_pix_loss: bool = False
  use_cls_token: bool = True
  dtype = jnp.float32

  def setup(self):
    # Encoder
    self.patch_embed = nn_layers.PatchEmbed(
        image_size=self.image_size,
        patch_size=self.patch_size,
        embed_dim=self.embed_dim,
    )
    num_patches = self.patch_embed.num_patches
    self.pos_embed = jnp.expand_dims(
        utils.get_2d_sincos_pos_embed(
            self.embed_dim,
            int(num_patches ** 0.5),
            cls_token=self.use_cls_token
        ), axis=0)
    self.blocks = [
        vit.Block(
            dim=self.embed_dim,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            norm_layer=nn.LayerNorm
        ) for _ in range(self.depth)
    ]
    self.norm = nn.LayerNorm()
    self.cls_token = self.param('cls_token',
                                nn.initializers.normal(stddev=0.02),
                                (1, 1, self.embed_dim))

    # Decoder
    self.decoder_embed = nn.Dense(
        self.decoder_embed_dim,
        use_bias=True
    )
    self.mask_token = self.param('mask_token',
                                 nn.initializers.normal(stddev=0.02),
                                 (1, 1, self.decoder_embed_dim))
    self.decoder_pos_embed = jnp.expand_dims(
        utils.get_2d_sincos_pos_embed(
            self.decoder_embed_dim,
            int(num_patches ** 0.5),
            cls_token=self.use_cls_token
        ), axis=0)
    self.decoder_blocks = [
        vit.Block(
            self.decoder_embed_dim,
            self.decoder_num_heads,
            self.mlp_ratio,
            qkv_bias=True,
            norm_layer=nn.LayerNorm
        ) for _ in range(self.decoder_depth)
    ]
    self.decoder_norm = nn.LayerNorm()
    self.decoder_pred = nn.Dense(
        self.patch_size**2 * self.num_channels,
        use_bias=True
    )

  def empty_data(self, image_size):
    return {
      'image': jnp.empty(
          (1, image_size, image_size, self.num_channels),
          dtype=self.dtype
      ),
    }

  def random_masking(self, x, mask_ratio, rng):
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    noise = random.uniform(rng, (N, L))

    # sort noise for each sample
    ids_shuffle = jnp.argsort(noise, axis=1)  # ascend: small is keep, large is remove
    ids_restore = jnp.argsort(ids_shuffle, axis=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = jnp.take_along_axis(x, ids_keep[:, :, None], axis=1)

    # generate the binary mask: 0 is keep, 1 is remove
    mask = np.ones((N, L))
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = jnp.take_along_axis(mask, ids_restore, axis=1)

    return x_masked, mask, ids_restore

  def encoder(self, x, mask_ratio, **kwargs):
    training = kwargs.get('training', False)

    # embed patches, [N, L, D]
    x = self.patch_embed(x, **kwargs)

    # add pos embed w/o cls token
    x = x + self.pos_embed[:, 1:, :]

    if training:
      # masking: length -> length * mask_ratio
      rng = kwargs.get('mrng', jax.random.PRNGKey(0))
      x, mask, ids_restore = self.random_masking(x, mask_ratio, rng)
    else:
      mask, ids_restore = None, None

    if self.use_cls_token:
      # append cls token
      cls_token = self.cls_token + self.pos_embed[:, :1, :]
      cls_token = cls_token.repeat(x.shape[0], axis=0)
      x = jnp.concatenate([cls_token, x], axis=1)

    # apply Transformer blocks
    for block in self.blocks:
      x = block(x, **kwargs)
    x = self.norm(x)

    return x, mask, ids_restore

  def decoder(self, x, ids_restore, **kwargs):
    # embed tokens
    x = self.decoder_embed(x)

    if ids_restore is not None:
      # append mask tokens to sequence
      masks_tokens = jnp.tile(self.mask_token,
                              (x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1))
      x_ = jnp.concatenate([x[:, 1:, :], masks_tokens], axis=1) # no cls token
      x_ = jnp.take_along_axis(x_, ids_restore[:, :, None], axis=1)
      x = jnp.concatenate([x[:, :1, :], x_], axis=1) # add cls token

    # add pos embed
    x = x + self.decoder_pos_embed

    # apply Transformer blocks
    for block in self.decoder_blocks:
      x = block(x, **kwargs)
    x = self.decoder_norm(x)

    # predictor projection
    x = self.decoder_pred(x)

    # remove cls token
    x = x[:, 1:, :]

    return x

  @nn.compact
  def __call__(self, batch, **kwargs):
    image = batch['image']
    mask_ratio = kwargs.get('mask_ratio', 0.75)
    latent, mask, ids_restore = self.encoder(image, mask_ratio, **kwargs)
    pred = self.decoder(latent, ids_restore, **kwargs)
    return {
        'latent': latent,
        'pred': pred,
        'mask': mask,
        'ids_restore': ids_restore,
    }


  @staticmethod
  def metrics_fn(attributes, outputs, batch):
    pred = outputs['pred']
    mask = outputs['mask']
    image = batch['image']
    patch_size = attributes.patch_size
    norm_pix_loss = attributes.norm_pix_loss
    if isinstance(batch, tuple):
      batch, _ = batch

    target = utils.patchify(image, patch_size)
    if norm_pix_loss:
      mean = target.mean(axis=-1, keepdims=True)
      var = target.var(axis=-1, keepdims=True)
      target = (target - mean) / (var + 1e-6) ** 0.5

    loss = (pred - target)**2
    loss = loss.mean(axis=-1)  # [N, L], mean loss per patch
    if mask is not None:
      loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    else:
      loss = loss.mean()

    # logging for images, take 3 images
    targt_image = image[:3]
    if mask is not None:
      pred_patches = jnp.where(mask[:3, :, None], pred[:3], target[:3])
    else:
      pred_patches = pred[:3]
    pred_image = utils.unpatchify(pred_patches, patch_size)

    ppred = utils.unpatchify(pred, patch_size)[:, :, :, :1]
    ptarg = utils.unpatchify(target, patch_size)[:, :, :, :1]
    prec_err = jnp.mean((ppred - ptarg) ** 2)

    nmse = losses.nmse(ppred, ptarg).mean()
    ssim = losses.ssim(ppred, ptarg).mean()

    return {
        'losses/total_loss': loss,
        'metrics/prec_err': prec_err,
        'metrics/prec_nmse': nmse,
        'metrics/prec_ssim': ssim,
        'images/pred@image': pred_image,
        'images/target@image': targt_image,
    }
