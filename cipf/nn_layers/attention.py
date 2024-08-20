import jax.lax
import jax.nn
import jax.numpy as jnp
from flax import linen as nn


def scaled_dot_product_attention(query,
                                 key,
                                 value,
                                 attention_mask=None,
                                 dropout_rate=0.0,
                                 is_causal=False,
                                 scale=None,
                                 training=False):
  L, S = query.shape[-2], key.shape[-2]
  scale_factor = scale or 1 / jnp.sqrt(query.shape[-1])
  attention_bias = jnp.zeros((L, S), dtype=query.dtype)
  if is_causal:
    assert (attention_mask is None,
            'Causal mask and attention_mask cannot be used together.')
    temp_mask = jnp.tril(jnp.ones((L, S), dtype=jnp.bool_), 0)
    attention_bias = jax.lax.select(
        jnp.logical_not(temp_mask),
        attention_bias,
        jax.lax.broadcast(float('-inf'), (L, S))
    ) # masked fill with -inf

  if attention_mask is not None:
    if attention_mask.dtype == jnp.bool_:
      attention_bias = jax.lax.select(
          attention_mask,
          attention_bias,
          jax.lax.broadcast(float('-inf'), (L, S))
      )
    else:
      attention_bias = attention_bias + attention_mask

  attention_weights = query @ key.transpose((0, 1, 3, 2)) * scale_factor
  attention_weights = attention_weights + attention_bias
  attention_weights = jax.nn.softmax(attention_weights, axis=-1)
  attention_weights = nn.Dropout(rate=dropout_rate)(attention_weights,
                                                    deterministic=not training)
  return attention_weights @ value
