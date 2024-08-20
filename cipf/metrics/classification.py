import jax
import jax.numpy as jnp

from cipf.metrics import plot_utils


def generate_matrix_from_mcm(mcm, reduced_metrics = {}, name: str = 'mcm'):
  name = name or 'mcm'
  mcm = jnp.astype(mcm, jnp.float32)
  tp = mcm[:, 0, 0]
  fp = mcm[:, 0, 1]
  fn = mcm[:, 1, 0]
  p = tp / (tp + fp + 1e-8)
  r = tp / (tp + fn + 1e-8)
  f1 = 2 * (p * r) / (p + r + 1e-8)
  support = tp + fn
  support_prob = support / jnp.sum(support)

  scalars = reduced_metrics.get('scalar', {})

  scalars.update({
    f'f1-score@macro': jnp.mean(f1),
    f'f1-score@micro': jnp.sum(tp) / (jnp.sum(tp) + jnp.sum(fp) + 1e-8),
    f'f1-score@weighted': jnp.sum(f1 * support_prob),
    f'precision@macro': jnp.mean(p),
    f'precision@weighted': jnp.sum(p * support_prob),
    f'recall@macro': jnp.mean(r),
    f'recall@weighted': jnp.sum(r * support_prob),
  })

  num_classes = mcm.shape[0]
  if num_classes < 50:
    # metrics for each class
    for ncls in range(mcm.shape[0]):
      scalars.update({
          f'precision@{name}/{ncls}': p[ncls],
          f'recall@{name}/{ncls}': r[ncls],
          f'f1-score@{name}/{ncls}': f1[ncls],
      })
    # plot confusion matrix
    mcm_sum = jnp.sum(mcm, axis=(1, 2), keepdims=True)
    mcm = jnp.divide(mcm, mcm_sum + 1e-8)
    image = plot_utils.plot_multilabel_confusion_matrix(mcm)
    if 'image' not in reduced_metrics:
      reduced_metrics['image'] = {}
    reduced_metrics['image'][name] = image

  reduced_metrics['scalar'].update(scalars)
  return reduced_metrics
