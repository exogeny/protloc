import math
from functools import partial

import jax
import jax.image
from jax import lax
from jax import scipy
from jax import random
import jax.numpy as jnp

import numpy as np
from PIL import Image


def display_image(proteins,
                  cells,
                  nucleus,
                  protein_recons,
                  cell_recons = None,
                  nucleus_recons = None,
                  ncols: int = 8,
                  padding: int = 16):
  images = [proteins, protein_recons, cells, nucleus]
  if cell_recons is not None:
    images.insert(3, cell_recons)
  if nucleus_recons is not None:
    images.insert(5, nucleus_recons)
  nrows = len(images)

  images = jnp.stack(images, axis=-1)

  ncols = min(images.shape[0], ncols)
  height, width = (
      int(images.shape[1] + padding),
      int(images.shape[2] + padding),
  )
  grid = jnp.full(
      (height * nrows + padding, width * ncols + padding, 1),
      0, dtype=jnp.float32)

  for y in range(nrows):
    for x in range(ncols):
      grid = grid.at[
          y * height + padding : (y + 1) * height,
          x * width + padding : (x + 1) * width,
      ].set(images[x, ..., y])

  ndarr = np.array(jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8))
  return ndarr


def save_image(ndarray, fp, nrow=8, padding=2, pad_value=0.0, format_img=None):
  """Make a grid of images and Save it into an image file.

  Args:
    ndarray (array_like): 4D mini-batch images of shape (B x H x W x C)
    fp:  A filename(string) or file object
    nrow (int, optional): Number of images displayed in each row of the grid.
      The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
    padding (int, optional): amount of padding. Default: ``2``.
    pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    format_img(Optional):  If omitted, the format to use is determined from the
      filename extension. If a file object was used instead of a filename,
      this parameter should always be used.
  """

  if not (
      isinstance(ndarray, jnp.ndarray)
      or (
          isinstance(ndarray, list)
          and all(isinstance(t, jnp.ndarray) for t in ndarray)
      )
  ):
    raise TypeError(f'array_like of tensors expected, got {type(ndarray)}')

  ndarray = jnp.asarray(ndarray)

  if ndarray.ndim == 4 and ndarray.shape[-1] == 1:  # single-channel images
    ndarray = jnp.concatenate((ndarray, ndarray, ndarray), -1)

  # make the mini-batch of images into a grid
  nmaps = ndarray.shape[0]
  xmaps = min(nrow, nmaps)
  ymaps = int(math.ceil(float(nmaps) / xmaps))
  height, width = (
      int(ndarray.shape[1] + padding),
      int(ndarray.shape[2] + padding),
  )
  num_channels = ndarray.shape[3]
  grid = jnp.full(
      (height * ymaps + padding, width * xmaps + padding, num_channels),
      pad_value,
  ).astype(jnp.float32)
  k = 0
  for y in range(ymaps):
    for x in range(xmaps):
      if k >= nmaps:
        break
      grid = grid.at[
          y * height + padding : (y + 1) * height,
          x * width + padding : (x + 1) * width,
      ].set(ndarray[k])
      k = k + 1

  # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
  ndarr = np.array(jnp.clip(grid * 255.0 + 0.5, 0, 255).astype(jnp.uint8))
  im = Image.fromarray(ndarr.copy())
  im.save(fp, format=format_img)


@jax.jit
def get_affine_matrix(theta, scale, translate):
  matrix = theta * scale
  matrix = jnp.concatenate([matrix, translate], axis=1)
  matrix = matrix.transpose(0, 2, 1)
  return matrix


def meshgrid(height, width):
  x_t = jnp.dot(jnp.ones((height, 1)), jnp.linspace(-1.0, 1.0, width)[None])
  y_t = jnp.dot(jnp.linspace(-1.0, 1.0, height)[:, None], jnp.ones((1, width)))

  x_t_flat = x_t.reshape((1, -1))
  y_t_flat = y_t.reshape((1, -1))
  ones =  jnp.ones_like(x_t_flat)
  grid = jnp.concatenate([x_t_flat, y_t_flat, ones], axis=0)
  return grid


@jax.jit
def affine_transform(images, matrix):
  batch_size, height, width, num_channels = images.shape
  grid = meshgrid(height, width)
  transformed_points = jnp.dot(matrix, grid)
  transformed_points = (transformed_points + 1) / 2
  transformed_points = transformed_points * jnp.array([[width], [height]])
  output = lax.map(
      lambda x: jnp.stack(
        [
          scipy.ndimage.map_coordinates(
              x[0][..., channel], x[1], order=1, mode='constant'
          ) for channel in range(num_channels)
        ], axis=-1
      ),
      [images, transformed_points]
  )
  return output.reshape((batch_size, height, width, num_channels))


@partial(jax.jit, static_argnums=(1, 2))
def random_affine(image, rng, prob: float = 0.5):
  B = image.shape[0]
  theta = random.bernoulli(rng, prob, (image.shape[0], 1, 1, 1))
  r1, r2, r3 = random.split(rng, 3)
  r_theta = random.uniform(r1, (B, 2, 2), minval=-1.0, maxval=1.0)
  r_scale = random.uniform(r2, (B, 2, 2), minval=0.7, maxval=1.3)
  r_trans = random.uniform(r3, (B, 1, 2), minval=-0.3, maxval=0.3)
  affine_matrix = get_affine_matrix(r_theta, r_scale, r_trans)
  image = jnp.where(theta, affine_transform(image, affine_matrix), image)
  return image
