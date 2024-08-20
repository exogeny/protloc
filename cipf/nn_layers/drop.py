from typing import Optional, Tuple, Union
import jax.numpy as jnp
from jax import random
from flax import linen as nn


def drop_path(x, rng, drop_prob, scale_by_keep=False, training: bool = False):
  """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
  """
  if drop_prob == 0. or not training:
    return x
  keep_prob = 1 - drop_prob
  random_tensor = random.bernoulli(rng, keep_prob, x.shape)
  if keep_prob > 0.0 and scale_by_keep:
    random_tensor = random_tensor / keep_prob
  return x * random_tensor


class DropPath(nn.Module):
  drop_prob: float = 0.0
  scale_by_keep: bool = True

  @nn.compact
  def __call__(self, x, **kwargs):
    rng = kwargs.get('mrng', random.PRNGKey(0))
    training = kwargs.get('training', False)
    return drop_path(x, rng, self.drop_prob, self.scale_by_keep, training)


class PatchDropout(nn.Module):
  prob: float = 0.5
  num_prefix_tokens: int = 1
  ordered: bool = False
  return_indices: bool = False

  @nn.compact
  def __call__(self, x, **kwargs) -> Union[jnp.ndarray, Tuple[jnp.ndarray, Optional[jnp.ndarray]]]:
    training = kwargs.get('training', False)
    if not training or self.prob == 0.:
      if self.return_indices:
        return x, None
      return x

    if self.num_prefix_tokens:
      prefix_tokens, x = x[:, :self.num_prefix_tokens], x[:, self.num_prefix_tokens:]
    else:
      prefix_tokens = None

    B, L = x.shape[:2]
    num_keep = max(1, int(L * (1. - self.prob)))
    keep_indices = jnp.argsort(random.randint(B, L), axis=-1)[:, :num_keep]
    if self.ordered:
      keep_indices = jnp.sort(keep_indices, axis=-1)[0]
    x = jnp.take_along_axis(x, keep_indices[..., None], axis=-1)

    if prefix_tokens is not None:
      x = jnp.concatenate((prefix_tokens, x), axis=1)

    if self.return_indices:
      return x, keep_indices
    return x
