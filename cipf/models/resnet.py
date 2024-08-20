from typing import Any, Callable, Optional, Sequence, Union
import functools

import jax.numpy as jnp
from flax import linen as nn

from cipf.nn_layers import losses
from cipf.models import nn_blocks

ModuleDef = Any


class ResNet(nn.Module):
  stage_sizes: Sequence[int]
  block_cls: ModuleDef
  num_classes: int
  classifier_name: str
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: ModuleDef = nn.Conv
  class_weights: Optional[Union[float, Sequence[float]]] = 1.0

  def empty_data(self, image_size):
    return {
      'image': jnp.empty((1, image_size, image_size, 3))
    }

  @nn.compact
  def __call__(self, batch, **kwargs):
    image = batch['image']
    training = kwargs.get('training', False)
    conv = functools.partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = functools.partial(
        nn.BatchNorm,
        use_running_average=not training,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
        axis_name='batch',
    )

    x = conv(self.num_filters,
             (7, 7),
             (2, 2),
             padding=[(3, 3), (3, 3)],
             name='conv_init',
            )(image)
    x = norm(name='bn_init')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x = self.block_cls(
            self.num_filters * 2 ** i,
            strides=strides,
            conv=conv,
            norm=norm,
            act=self.act,
        )(x)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x

  @staticmethod
  def metrics_fn(attributes, outputs, batch):
    label = batch[attributes.classifier_name]
    class_weights = jnp.array(attributes.class_weights)
    # total_loss = losses.binary_cross_entropy_with_logits(outputs, label, class_weights).mean()
    # total_loss = losses.focal_loss(outputs, label).mean()
    total_loss = losses.focal_binary_cross_entropy(outputs, label).mean()
    metrics = {
      'losses/total_loss': total_loss,
    }

    y_pred = jnp.greater_equal(nn.sigmoid(outputs), 0.5)
    mcm = losses.compute_multilabel_confusion_matrix(y_pred, label)
    metrics.update({
      'mcm@mcm': mcm,
    })
    return metrics


ResNet18 = functools.partial(
    ResNet,
    stage_sizes=[2, 2, 2, 2],
    block_cls=nn_blocks.ResidualBlock
)
ResNet34 = functools.partial(
    ResNet,
    stage_sizes=[3, 4, 6, 3],
    block_cls=nn_blocks.ResidualBlock
)
ResNet50 = functools.partial(
    ResNet,
    stage_sizes=[3, 4, 6, 3],
    block_cls=nn_blocks.BottleneckResidualBlock
)
ResNet101 = functools.partial(
    ResNet,
    stage_sizes=[3, 4, 23, 3],
    block_cls=nn_blocks.BottleneckResidualBlock
)
ResNet152 = functools.partial(
    ResNet,
    stage_sizes=[3, 8, 36, 3],
    block_cls=nn_blocks.BottleneckResidualBlock
)
ResNet200 = functools.partial(
    ResNet,
    stage_sizes=[3, 24, 36, 3],
    block_cls=nn_blocks.BottleneckResidualBlock
)


def init_from_config(config):
  model_name = config.name
  if model_name.lower() == 'resnet18':
    model = ResNet18
  elif model_name.lower() == 'resnet34':
    model = ResNet34
  elif model_name.lower() == 'resnet50':
    model = ResNet50
  elif model_name.lower() == 'resnet101':
    model = ResNet101
  elif model_name.lower() == 'resnet152':
    model = ResNet152
  elif model_name.lower() == 'resnet200':
    model = ResNet200
  return model(num_classes=config.num_classes)
