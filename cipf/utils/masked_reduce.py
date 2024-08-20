import jax.numpy as jnp


def reduce_mean(func, mask, *args, **kwargs):
  """Reduce mean with mask

  Args:
    func: The function to be reduced.
    mask: The mask to be applied to the function.

  Returns:
    The reduced mean value.
  """
  if mask is None:
    return jnp.mean(func(*args, **kwargs))
  losses = func(*args, **kwargs)
  return jnp.sum(losses * mask) / jnp.sum(mask)
