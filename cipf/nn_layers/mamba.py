# Reference: https://github.com/state-spaces/mamba/blob/main/mamba_ssm/modules/ssd_minimal.py
from typing import Optional
import jax.numpy as jnp

def segsum(x: jnp.ndarray) -> jnp.ndarray:
  T = x.shape[-1]
  x = jnp.repeat(jnp.expand_dims(x, axis=-1), T, axis=-1)
  mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_), -1)
  x = jnp.where(mask, x, 0)
  x_segsum = jnp.cumsum(x, axis=-2)
  mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_), 0)
  x_segsum = jnp.where(mask, x_segsum, -jnp.inf)
  return x_segsum


def rearrange_cls_token(x, block_len):
  batch = x.shape[0]
  cls_token, pos_token = x[:, :1], x[:, 1:]
  cls_token = jnp.repeat(cls_token, block_len, axis=1)
  cls_token = jnp.reshape(cls_token, (batch, 1, block_len) + x.shape[2:])
  pos_token = jnp.reshape(pos_token, (batch, -1, block_len) + x.shape[2:])
  x = jnp.concatenate([cls_token, pos_token], axis=1)
  return x


def ssd_minimal_discrete(X, A, B, C, block_len,
                         has_cls_token: bool = False,
                         initial_states: Optional[jnp.ndarray] = None):
  """Minimal SSD implementation for discrete states.
  Arguments:
    X: (batch, length, num_heads, d_head)
    A: (batch, length, num_heads)
    B: (batch, length, num_heads, d_state)
    C: (batch, length, num_heads, d_state)
  Returns:
    Y: (batch, length, num_heads, d_head)
    new_states: (batch, length, num_heads, d_state)
  """
  batch, length, num_heads, d_head = X.shape
  d_state = B.shape[-1]

  if has_cls_token and length > 1:
    X = rearrange_cls_token(X, block_len)
    A = rearrange_cls_token(A, block_len)
    B = rearrange_cls_token(B, block_len)
    C = rearrange_cls_token(C, block_len)
  else:
    # rearrange into blocks/chunks
    num_chunks = length // block_len
    X = jnp.reshape(X, (batch, num_chunks, block_len, num_heads, d_head))
    A = jnp.reshape(A, (batch, num_chunks, block_len, num_heads))
    B = jnp.reshape(B, (batch, num_chunks, block_len, num_heads, d_state))
    C = jnp.reshape(C, (batch, num_chunks, block_len, num_heads, d_state))

  A = jnp.einsum('bclh->bhcl', A)
  A_cumsum = jnp.cumsum(A, axis=-1)

  # 1. compute the output for each intra-chunk (diagonal blocks)
  L = jnp.exp(segsum(A))
  Y_diag = jnp.einsum('bclhn,bcshn,bhcls,bcshp->bclhp', C, B, L, X)

  # 2. compute the state for each intra-chunk
  decay_states = jnp.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
  states = jnp.einsum('bclhn,bhcl,bclhp->bchpn', B, decay_states, X)

  # 3. compute the inter-chunk SSM recurrence; produces correct SSM states at
  #    chunk boundaries (middle term of factorization of off-diag blocks;
  #    A terms).
  if initial_states is None:
    initial_states = jnp.zeros_like(states[:, :1])
  states = jnp.concatenate([initial_states, states], axis=1)
  decay_chunk = segsum(jnp.pad(A_cumsum[:, :, :, -1], ((0, 0), (0, 0), (1, 0))))
  decay_chunk = jnp.exp(decay_chunk)
  new_states = jnp.einsum('bhzc,bchpn->bzhpn', decay_chunk, states)
  states, final_states = new_states[:, :-1], new_states[:, -1:]

  # 4. compute state -> output conversion per chunk
  state_decay_out = jnp.exp(A_cumsum)
  Y_off = jnp.einsum('bclhn,bchpn,bhcl->bclhp', C, states, state_decay_out)

  # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal
  # blocks)
  Y = Y_diag + Y_off
  if has_cls_token:
    cls_token, pos_token = Y[:, :1], Y[:, 1:]
    cls_token = cls_token.reshape((batch, -1, num_heads, d_head))
    pos_token = pos_token.reshape((batch, -1, num_heads, d_head))
    cls_token = jnp.mean(cls_token, axis=1, keepdims=True)
    Y = jnp.concatenate([cls_token, pos_token], axis=1)
  else:
    Y = jnp.reshape(Y, (batch, length, num_heads, d_head))
  return Y, final_states
