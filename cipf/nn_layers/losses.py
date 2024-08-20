import jax
import jax.numpy as jnp
from functools import partial

from flax import linen as nn

@jax.vmap
def kl_divergence(mean, logvar):
  return -0.5 * jnp.sum(1 + logvar - jnp.square(mean) - jnp.exp(logvar))

@jax.vmap
def l1_error(logits, labels):
  logits = nn.sigmoid(logits)
  return jnp.sum(jnp.abs(logits - labels))

@jax.vmap
def mean_squared_error(logits, labels):
  return jnp.mean(jnp.square(logits - labels))

@partial(jax.vmap, in_axes=(0, 0, None))
def binary_cross_entropy_with_logits(logits, labels, class_weight=None):
  # https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits # pylint: disable=line-too-long
  max_value = jnp.maximum(logits, 0)
  loss = max_value - logits * labels + jnp.log(1 + jnp.exp(-jnp.abs(logits)))
  if class_weight is not None:
    loss = loss * class_weight
  return jnp.mean(loss)

def focal_loss(logits, labels, alpha: float = 0.25, gamma: float = 2.0):
  # https://arxiv.org/abs/1708.02002

  @jax.vmap
  def _focal_loss(logits, labels):
    y_pred = jnp.clip(jax.nn.sigmoid(logits), 1e-8, 1 - 1e-8)
    alpha_t = labels * alpha + (1. - labels) * (1. - alpha)
    y_t = jnp.multiply(labels, y_pred) + jnp.multiply(1 - labels, 1 - y_pred)
    weight = jnp.power(jnp.subtract(1., y_t), gamma)
    fl = jnp.multiply(jnp.multiply(alpha_t, jnp.log(y_t)), weight)
    return -jnp.sum(fl)
  return _focal_loss(logits, labels)

@jax.vmap
def softmax_cross_entropy_with_logits(logits, labels):
  return -jnp.sum(labels * nn.log_softmax(logits))


@jax.vmap
def focal_binary_cross_entropy(logits, labels):
  p = jnp.clip(jax.nn.sigmoid(logits), 1e-8, 1 - 1e-8)
  p = jnp.where(labels >= 0.5, p, 1 - p)
  logp = - jnp.log(p)
  loss = logp * ((1 - p) ** 2)
  return loss.sum()


@jax.vmap
def fid(logits, labels):
  pred = nn.sigmoid(logits)
  mu1, sigma1 = jnp.mean(pred), jnp.cov(pred, rowvar=False)
  mu2, sigma2 = jnp.mean(labels), jnp.cov(labels, rowvar=False)
  ssdiff = jnp.sum((mu1 - mu2)**2.0)
  covmean = jnp.linalg.sqrtm(sigma1.dot(sigma2))
  if jnp.iscomplexobj(covmean):
    covmean = covmean.real
  fid = ssdiff + jnp.trace(sigma1 + sigma2 - 2.0 * covmean)
  return fid

def cosine_similarity(e1,
                      e2,
                      margin: float = 0.0,
                      similary: bool = True):
  @jax.vmap
  def compute(e1, e2):
    ang = jnp.dot(e1, e2) / (jnp.linalg.norm(e1) * jnp.linalg.norm(e2))
    return 1 - ang if similary else jnp.maximum(0, ang - margin)
  return jnp.sum(compute(e1, e2))

@jax.jit
def compute_multilabel_confusion_matrix(y_pred, y_true, mask = None):
  if mask is None:
    mask = jnp.ones(y_pred.shape[0], dtype=y_pred.dtype)
  mask = jnp.expand_dims(mask, axis=-1)
  tp = jnp.count_nonzero(jnp.logical_and(y_pred == 1, y_true == 1) * mask, axis=0)
  fp = jnp.count_nonzero(jnp.logical_and(y_pred == 1, y_true == 0) * mask, axis=0)
  fn = jnp.count_nonzero(jnp.logical_and(y_pred == 0, y_true == 1) * mask, axis=0)
  tn = jnp.count_nonzero(jnp.logical_and(y_pred == 0, y_true == 0) * mask, axis=0)
  matrix = jnp.stack([tp, fp, fn, tn], axis=0).T.reshape((-1, 2, 2))
  return matrix


def ssim(a: jax.Array,
         b: jax.Array,
         *,
         max_val: float = 1.0,
         filter_size: int = 11,
         filter_sigma: float = 1.5,
         k1: float = 0.01,
         k2: float = 0.03,
         return_map: bool = False,
         precision=jax.lax.Precision.HIGHEST,
         filter_fn=None):
  """Computes the Structural Similarity Index (SSIM) between two images.
  https://github.com/google-deepmind/dm_pix/blob/master/dm_pix/_src/metrics.py
  """
  if filter_fn is None:
    # Construct a 1D Gaussian blur filter.
    hw = filter_size // 2
    shift = (2 * hw - filter_size + 1) / 2
    f_i = ((jnp.arange(filter_size) - hw + shift) / filter_sigma) ** 2
    filt = jnp.exp(-0.5 * f_i)
    filt /= jnp.sum(filt)

    # Construct a 1D convolution.
    def filter_fn_1(z):
      return jnp.convolve(z, filt, mode="valid", precision=precision)

    filter_fn_vmap = jax.vmap(filter_fn_1)

    # Apply the vectorized filter along the y axis.
    def filter_fn_y(z):
      z_flat = jnp.moveaxis(z, -3, -1).reshape((-1, z.shape[-3]))
      z_filtered_shape = ((z.shape[-4],) if z.ndim == 4 else ()) + (
          z.shape[-2],
          z.shape[-1],
          -1,
      )
      z_filtered = jnp.moveaxis(
          filter_fn_vmap(z_flat).reshape(z_filtered_shape), -1, -3
      )
      return z_filtered

    # Apply the vectorized filter along the x axis.
    def filter_fn_x(z):
      z_flat = jnp.moveaxis(z, -2, -1).reshape((-1, z.shape[-2]))
      z_filtered_shape = ((z.shape[-4],) if z.ndim == 4 else ()) + (
          z.shape[-3],
          z.shape[-1],
          -1,
      )
      z_filtered = jnp.moveaxis(
          filter_fn_vmap(z_flat).reshape(z_filtered_shape), -1, -2
      )
      return z_filtered

    # Apply the blur in both x and y.
    filter_fn = lambda z: filter_fn_y(filter_fn_x(z))

  mu0 = filter_fn(a)
  mu1 = filter_fn(b)
  mu00 = mu0 * mu0
  mu11 = mu1 * mu1
  mu01 = mu0 * mu1
  sigma00 = filter_fn(a**2) - mu00
  sigma11 = filter_fn(b**2) - mu11
  sigma01 = filter_fn(a * b) - mu01

  # Clip the variances and covariances to valid values.
  # Variance must be non-negative:
  epsilon = jnp.finfo(jnp.float32).eps ** 2
  sigma00 = jnp.maximum(epsilon, sigma00)
  sigma11 = jnp.maximum(epsilon, sigma11)
  sigma01 = jnp.sign(sigma01) * jnp.minimum(
      jnp.sqrt(sigma00 * sigma11), jnp.abs(sigma01)
  )

  c1 = (k1 * max_val) ** 2
  c2 = (k2 * max_val) ** 2
  numer = (2 * mu01 + c1) * (2 * sigma01 + c2)
  denom = (mu00 + mu11 + c1) * (sigma00 + sigma11 + c2)
  ssim_map = numer / denom
  ssim_value = jnp.mean(ssim_map, list(range(-3, 0)))
  return ssim_map if return_map else ssim_value
