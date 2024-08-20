from typing import Any, Dict, List, Optional, Union, Literal
import jax
import jax.numpy as jnp
from flax import linen as nn

from cipf import core
from cipf.nn_layers import losses


class BaseClassifier(nn.Module):
  use_bias: bool = True
  threshold: float = 0.5
  num_classes: int = 1000
  multi_label: bool = True
  classifier_name: Optional[str] = None
  encoder_cls: Optional[Dict[str, Any]] = None
  dtype = jnp.float32

  def setup(self):
    self.encoder = core.init_model_from_config(
        self.encoder_cls, use_bias=self.use_bias)

  def empty_data(self, image_size):
    return {
      'image': jnp.zeros((1, image_size, image_size, 3), self.dtype),
      self.classifier_name: jnp.zeros((1, self.num_classes), self.dtype),
    }

  @staticmethod
  def metrics_fn(self, outputs, batch):
    labels = batch[self.classifier_name]
    logits = outputs['logits']
    mask = batch.get('__mask', None)
    metrics = {k: v for k, v in outputs.items() if k.startswith('metrics/')}
    total_losses = {k: v for k, v in outputs.items() if k.startswith('losses/')}
    total_loss = jnp.sum(jnp.asarray(list(total_losses.values())))
    if self.multi_label:
      bce = losses.focal_binary_cross_entropy(logits, labels).mean()
      y_prob = nn.sigmoid(logits)
      y_pred = jnp.greater_equal(y_prob, self.threshold)
    else:
      bce = losses.softmax_cross_entropy_with_logits(logits, labels).mean()
      y_pred = jnp.argmax(logits, axis=-1)
      y_pred = jax.nn.one_hot(y_pred, self.num_classes)

    mcm = losses.compute_multilabel_confusion_matrix(y_pred, labels)

    total_loss = bce + total_loss
    metrics.update(total_losses)
    metrics.update({
      'mcm@mcm': mcm,
      'losses/bce': bce,
      'losses/total_loss': total_loss,
    })
    return metrics


class FineTuneModel(BaseClassifier):
  pool_type: Literal['global_pool', 'cls_token', 'all_pool', 'dense'] = 'global_pool'

  def setup(self):
    super().setup()
    # self.norm = nn.LayerNorm(epsilon=1e-6)
    self.norm = nn.RMSNorm(epsilon=1e-6)
    self.classifier = nn.Dense(
        self.num_classes,
        use_bias=self.use_bias,
        kernel_init=nn.initializers.truncated_normal(stddev=2e-5),
    )
    if self.pool_type == 'dense':
      self.pool = nn.Dense(1, use_bias=self.use_bias)

  @nn.compact
  def __call__(self, batch, **kwargs):
    training = kwargs.get('training', False)
    kwargs.update(
    {
        'affine_transform': training,
        'encode_only': True,
    })
    outputs = self.encoder(batch, **kwargs)
    latent = outputs['latent']
    if self.pool_type == 'global_pool':
      features = jnp.mean(latent[:, 1:, :], axis=1)
      features = self.norm(features)
    elif self.pool_type == 'cls_token':
      features = self.norm(latent[:, 0])
    elif self.pool_type == 'all_pool':
      features = jnp.mean(latent, axis=1)
      features = self.norm(features)
    elif self.pool_type == 'dense':
      latent = jnp.reshape(latent, (latent.shape[0], -1, latent.shape[-1]))
      features = self.pool(jnp.transpose(latent, (0, 2, 1)))
      features = jnp.squeeze(features, axis=-1)
      features = self.norm(features)
    else:
      raise ValueError(f'Invalid pool_type: {self.pool_type}')
    outputs['logits'] = self.classifier(features)
    return outputs


class LinearProbeModel(FineTuneModel):
  def setup(self):
    super().setup()

  @nn.compact
  def __call__(self, batch, **kwargs):
    training = kwargs.pop('training', False)
    kwargs.update(
    {
        'training': False,
        'affine_transform': training,
        'encode_only': True,
    })
    outputs = self.encoder(batch, **kwargs)
    latent = outputs['latent']
    latent = jax.lax.stop_gradient(latent)
    if self.pool_type == 'global_pool':
      features = jnp.mean(latent[:, 1:, :], axis=1)
      features = self.norm(features)
    elif self.pool_type == 'cls_token':
      features = self.norm(latent[:, 0])
    elif self.pool_type == 'all_pool':
      features = jnp.mean(latent, axis=1)
      features = self.norm(features)
    elif self.pool_type == 'dense':
      features = self.pool(jnp.transpose(latent, (0, 2, 1)))
      features = jnp.squeeze(features, axis=-1)
      features = self.norm(features)
    else:
      raise ValueError(f'Invalid pool_type: {self.pool_type}')

    # clear all losses for freeze the parameters in encoder.
    outputs = {
        k: v for k, v in outputs.items() if not str(k).startswith('losses')
    }
    outputs['logits'] = self.classifier(features)
    return outputs
