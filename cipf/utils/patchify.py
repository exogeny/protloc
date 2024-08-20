import jax
import jax.numpy as jnp
from functools import partial


@partial(jax.jit, static_argnames=['patch_size'])
def patchify(images, patch_size):
  """
  images: (N, H, W, C)
  x: (N, L, patch_size**2*C)
  """
  h = w = images.shape[1] // patch_size
  x = images.reshape(images.shape[0], h, patch_size, w, patch_size, images.shape[3])
  x = jnp.einsum('nhpwqc->nhwpqc', x)
  x = x.reshape(images.shape[0], h * w, patch_size**2 * images.shape[3])
  return x


@partial(jax.jit, static_argnames=['patch_size'])
def unpatchify(x, patch_size):
  """
  x: (N, L, patch_size**2*C)
  images: (N, H, W, C)
  """
  p = patch_size
  h = w = int(x.shape[1]**.5)
  x = x.reshape(x.shape[0], h, w, p, p, x.shape[2] // (p**2))
  x = jnp.einsum('nhwpqc->nhpwqc', x)
  images = x.reshape(x.shape[0], h * p, w * p, x.shape[5])
  return images
