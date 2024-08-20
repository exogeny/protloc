from typing import Any, Callable
import functools

from jax import numpy as jnp
from flax import linen as nn

from cipf.models import quantizer
from cipf.nn_layers import losses

ModuleDef = Any


class ResidualBlock(nn.Module):
  num_hiddens: int
  num_residual_hiddens: int
  num_residual_layers: int
  conv: ModuleDef
  act: Callable
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    layers = []
    for _ in range(self.num_residual_layers):
      conv3 = self.conv(
          self.num_residual_hiddens,
          kernel_size=(3, 3),
          strides=(1, 1),
      )
      conv1 = self.conv(
          self.num_hiddens,
          kernel_size=(1, 1),
          strides=(1, 1),
      )
      layers.append((conv3, conv1))
    self.layers = layers

  @nn.compact
  def __call__(self, x):
    h = x
    for conv3, conv1 in self.layers:
      conv3_out = conv3(self.act(h))
      conv1_out = conv1(self.act(conv3_out))
      h += conv1_out
    return self.act(h)


class Encoder(nn.Module):
  num_hiddens: int
  num_residual_hiddens: int
  num_residual_layers: int
  conv: ModuleDef
  act: Callable
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.enc1 = self.conv(
        self.num_hiddens // 2,
        kernel_size=(4, 4),
        strides=(2, 2),
    )
    self.enc2 = self.conv(
        self.num_hiddens,
        kernel_size=(4, 4),
        strides=(2, 2),
    )
    self.enc3 = self.conv(
        self.num_hiddens,
        kernel_size=(3, 3),
        strides=(1, 1),
    )

    residual_blocks = []
    for _ in range(self.num_residual_layers):
      residual_blocks.append(ResidualBlock(
          self.num_hiddens,
          self.num_residual_hiddens,
          self.num_residual_layers,
          conv=self.conv,
          act=self.act,
      ))
    self.residual_blocks = nn.Sequential(residual_blocks)

  @nn.compact
  def __call__(self, x):
    h = self.act(self.enc1(x))
    h = self.act(self.enc2(h))
    h = self.act(self.enc3(h))
    return self.residual_blocks(h)


class Decoder(nn.Module):
  num_hiddens: int
  num_residual_hiddens: int
  num_residual_layers: int
  conv: ModuleDef
  upsample: ModuleDef
  act: Callable
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.dec1 = self.conv(
        self.num_hiddens,
        kernel_size=(3, 3),
        strides=(1, 1),
    )
    self.dec2 = self.upsample(
        self.num_hiddens // 2,
        kernel_size=(4, 4),
        strides=(2, 2),
    )
    self.dec3 = self.upsample(
        3,
        kernel_size=(4, 4),
        strides=(2, 2),
    )
    self.residual = ResidualBlock(
        self.num_hiddens,
        self.num_residual_hiddens,
        self.num_residual_layers,
        conv=self.conv,
        act=self.act,
    )

  @nn.compact
  def __call__(self, x):
    h = self.dec1(x)
    h = self.residual(h)
    h = self.act(self.dec2(h))
    r = self.dec3(h)
    return r


class VQVAE(nn.Module):
  embedding_dim: int
  num_embeddings: int
  commitment_cost: float
  num_hiddens: int
  num_residual_hiddens: int
  num_residual_layers: int
  data_variance: float
  decay: float
  use_bias: bool = True
  use_ema: bool = True
  protein_only: bool = False
  dtype = jnp.float32

  def setup(self):
    conv = functools.partial(
        nn.Conv,
        dtype=self.dtype,
    )
    upsample = functools.partial(
        nn.ConvTranspose,
        dtype=self.dtype,
    )

    self.pre_vq_conv1 = conv(
        self.embedding_dim,
        kernel_size=(1, 1),
        strides=(1, 1),
    )
    self.enc = Encoder(
        self.num_hiddens,
        self.num_residual_hiddens,
        self.num_residual_layers,
        conv=conv,
        act=nn.relu,
    )
    self.dec = Decoder(
        self.num_hiddens,
        self.num_residual_hiddens,
        self.num_residual_layers,
        conv=conv,
        upsample=upsample,
        act=nn.relu,
    )

    if self.use_ema:
      self.quantizer = quantizer.VectorQuantizerEMA(
          embedding_dim=self.embedding_dim,
          num_embeddings=self.num_embeddings,
          commitment_cost=self.commitment_cost,
          decay=self.decay,
      )
    else:
      self.quantizer = quantizer.VectorQuantizer(
          embedding_dim=self.embedding_dim,
          num_embeddings=self.num_embeddings,
          commitment_cost=self.commitment_cost,
      )

  def empty_data(self, image_size):
    return {
      'image': jnp.zeros((1, image_size, image_size, 3), dtype=self.dtype)
    }

  @nn.compact
  def __call__(self, batch, **kwargs):
    x = batch['image']
    z = self.pre_vq_conv1(self.enc(x))
    outputs = self.quantizer(z, **kwargs)
    x_recon = self.dec(outputs['quantized'])

    rec_err = jnp.mean((x_recon - x) ** 2) / self.data_variance
    outputs.update({
        'image': x,
        'latent': outputs['quantized'],
        'recon': x_recon,
        'rec_err': rec_err,
        'total_loss': rec_err + outputs['vq_loss'],
    })
    return outputs

  @staticmethod
  def metrics_fn(config, outputs, batch):
    del batch
    image = outputs['image']
    recon = outputs['recon']
    # only take the first channel[protein channel] for metrics
    pimage = image[:, :, :, 0]
    precon = recon[:, :, :, 0]

    print(pimage.shape, precon.shape)
    prec_err = losses.mean_squared_error(pimage, precon).mean()
    ssim = losses.ssim(pimage, precon).mean()

    return {
        'losses/total_loss': outputs['total_loss'],
        'losses/recon': outputs['rec_err'],
        'losses/vq': outputs['vq_loss'],
        'images/image@image': outputs['image'][:3],
        'images/recon@image': outputs['recon'][:3],
        'metrics/perplexity': outputs['perplexity'],
  
        'metrics/prec_err': prec_err,
        'metrics/prec_ssim': ssim,
    }
