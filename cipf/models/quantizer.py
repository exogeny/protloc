from jax.typing import ArrayLike

import jax
from jax import numpy as jnp
from flax import linen as nn


class VectorQuantizer(nn.Module):
  """Vector Quantizer Module.

  Implements the algorithm presented in
  'Neural Discrete Representation Learning' by van den Oord et al.
  https://arxiv.org/abs/1711.00937

  Reference: https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
  """
  embedding_dim: int
  num_embeddings: int
  commitment_cost: float
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.embeddings = self.param(
        'embeddings',
        nn.initializers.variance_scaling(
            scale=1.0,
            mode='fan_in',
            distribution='uniform'
        ),
        (self.embedding_dim, self.num_embeddings),
    )

  @nn.compact
  def __call__(self, x, **kwargs):
    del kwargs
    embeddings = self.embeddings
    flatten = jnp.reshape(x, (-1, self.embedding_dim))
    distance = (
        jnp.sum(flatten ** 2, axis=1, keepdims=True)
        - 2 * jnp.matmul(flatten, embeddings)
        + jnp.sum(embeddings ** 2, axis=0, keepdims=True)
    )

    encoding_indices = jnp.argmax(-distance, axis=1)
    encodings = jax.nn.one_hot(encoding_indices,
                               self.num_embeddings,
                               dtype=distance.dtype)

    encoding_indices = jnp.reshape(encoding_indices, x.shape[:-1])
    quantized = self.quantize(encoding_indices)

    e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - x) ** 2)
    q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(x)) ** 2)
    loss = q_latent_loss + self.commitment_cost * e_latent_loss

    # Straight Through Estimator
    quantized = x + jax.lax.stop_gradient(quantized - x)
    avg_probs = jnp.mean(encodings, axis=0)
    perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))

    return {
        'quantized': quantized,
        'vq_loss': loss,
        'metrics/perplexity': perplexity,
        'encoding_indices': encoding_indices,
    }

  def quantize(self, encoding_indices):
    w = jnp.transpose(self.embeddings, [1, 0])
    return jnp.take(w, encoding_indices, axis=0)


class ExponentialMovingAverage(nn.Module):
  """Maintains an exponential moving average for a value.

  Note this module uses debiasing by default. If you don't want this please use
  an alternative implementation.

  This module keeps track of a hidden exponential moving average that is
  initialized as a vector of zeros which is then normalized to give the average.
  This gives us a moving average which isn't biased towards either zero or the
  initial value. Reference (https://arxiv.org/pdf/1412.6980.pdf)

  Initially:

      hidden_0 = 0

  Then iteratively:

      hidden_i = (hidden_{i-1} - value) * (1 - decay)
      average_i = hidden_i / (1 - decay^i)
  """
  decay: float
  shape: jnp.shape

  def setup(self):
    self.counter = self.variable(
        'batch_stats',
        'counter',
        lambda shape: jnp.zeros(shape, dtype=jnp.float32),
        ()
    )
    self.hidden = self.variable(
        'batch_stats',
        'hidden',
        lambda shape: jnp.zeros(shape, dtype=jnp.float32),
        self.shape,
    )
    self.average = self.variable(
        'batch_stats',
        'average',
        lambda shape: jnp.zeros(shape, dtype=jnp.float32),
        self.shape,
    )

  @nn.compact
  def __call__(self, value):
    counter = self.counter.value + 1
    hidden = self.hidden.value * self.decay + value * (1 - self.decay)
    average = hidden / (1. - jnp.power(self.decay, counter))
    self.counter.value = counter
    self.hidden.value = hidden
    self.average.value = average
    return average


class VectorQuantizerEMA(nn.Module):
  """Vector Quantizer with Exponential Moving Average.

  Implements the algorithm presented in
  'Neural Discrete Representation Learning' by van den Oord et al.
  https://arxiv.org/abs/1711.00937

  Reference:https://github.com/google-deepmind/sonnet/blob/v2/sonnet/src/nets/vqvae.py
  """
  embedding_dim: int
  num_embeddings: int
  commitment_cost: float
  decay: float
  epsilon: float = 1e-5
  dtype: jnp.dtype = jnp.float32

  def setup(self):
    self.embeddings = self.variable(
        'batch_stats',
        'embeddings',
        nn.initializers.variance_scaling(
            scale=1.0,
            mode='fan_in',
            distribution='uniform'
        ),
        jax.random.PRNGKey(42),
        (self.embedding_dim, self.num_embeddings),
    )
    self.ema_cluster_size = ExponentialMovingAverage(
        decay=self.decay,
        shape=(self.num_embeddings,),
    )
    self.ema_dw = ExponentialMovingAverage(
        decay=self.decay,
        shape=(self.embedding_dim, self.num_embeddings),
    )

  @nn.compact
  def __call__(self, x, **kwargs):
    # default training to True for initializing ema_cluster_size and ema_dw
    training = kwargs.get('training', False)

    flatten = jnp.reshape(x, (-1, self.embedding_dim))
    distance = (
        jnp.sum(flatten ** 2, axis=1, keepdims=True)
        - 2 * jnp.matmul(flatten, self.embeddings.value)
        + jnp.sum(self.embeddings.value ** 2, axis=0, keepdims=True))

    encoding_indices = jnp.argmax(-distance, axis=1)
    encodings = jax.nn.one_hot(encoding_indices,
                               self.num_embeddings,
                               dtype=distance.dtype)

    encoding_indices = jnp.reshape(encoding_indices, x.shape[:-1])
    quantized = self.quantize(encoding_indices)

    e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - x) ** 2)
    q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(x)) ** 2)
    loss = q_latent_loss + self.commitment_cost * e_latent_loss

    if training:
      updated_ema_cluster_size = self.ema_cluster_size(jnp.sum(encodings, axis=0))
      dw = jnp.matmul(jnp.transpose(flatten), encodings)
      updated_ema_dw = self.ema_dw(dw)

      n = jnp.sum(updated_ema_cluster_size)
      updated_ema_cluster_size = (
          (updated_ema_cluster_size + self.epsilon) /
          (n + self.num_embeddings * self.epsilon) * n
      )
      normalized_updated_ema_dw = (
          updated_ema_dw / jnp.reshape(updated_ema_cluster_size, [1, -1])
      )
      self.embeddings.value = normalized_updated_ema_dw

    quantized = x + jax.lax.stop_gradient(quantized - x)
    avg_probs = jnp.mean(encodings, axis=0)
    perplexity = jnp.exp(-jnp.sum(avg_probs * jnp.log(avg_probs + 1e-10)))

    return {
        'quantized': quantized,
        'vq_loss': loss,
        'perplexity': perplexity,
        'encodings': encodings,
        'encoding_indices': encoding_indices,
    }

  def quantize(self, encoding_indices):
    w = jnp.transpose(self.embeddings.value, [1, 0])
    return jnp.take(w, encoding_indices, axis=0)
