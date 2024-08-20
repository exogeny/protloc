from typing import Any, Dict, Text, Tuple
import os
import time

import jax
from jax import numpy as jnp
from clu import metric_writers

from cipf.metrics import classification


def _reduce_fn(typed_metrics, count):
  count = max(count, 1)
  reduced_metrics = {k: {} for k in typed_metrics.keys()}
  for type, metrics in typed_metrics.items():
    if type == 'scalar':
      reduced_metrics[type].update(jax.tree_util.tree_map(
          lambda x: x.mean() / count, metrics))
    elif type == 'mcm':
      metrics = jax.tree_util.tree_map(lambda x: x.sum(axis=0), metrics)
      for k, metric in metrics.items():
        reduced_metrics = classification.generate_matrix_from_mcm(
            metric, reduced_metrics, k)
    else:
      reduced_metrics[type].update(jax.tree_util.tree_map(
          lambda x: x[0], metrics))
  return reduced_metrics


class MetricsManager:

  def __init__(self, logdir: str, flush_interval: int = 100, logging: bool = False):
    if not os.path.exists(logdir):
      os.makedirs(logdir, exist_ok=True)

    self._metrics_type_dict = {}
    self._typed_metrics = {
      'scalar': {},
      'image': {},
      'histogram': {},
      'mcm': {}, # multilabel confusion matrix
    }
    writers = [metric_writers.SummaryWriter(logdir)]
    if logging:
      writers.append(metric_writers.LoggingWriter())
    self._writer = metric_writers.AsyncMultiWriter(writers)

    self._write_fn = {
      'scalar': self._writer.write_scalars,
      'image': self._writer.write_images,
      'histogram': self._writer.write_histograms,
    }
    self._steps = 0
    self._last_step = 0
    self._flush_interval = flush_interval
    self._unflushed_metrics = []
    self._last_time = None

  def _flush(self):
    def get_metric_type(name: Text) -> Tuple[str, str]:
      if name in self._metrics_type_dict:
        return self._metrics_type_dict[name]

      newname, type = name, 'scalar'
      if '@' in name:
        namesplited = name.split('@')
        newname = '@'.join(namesplited[:-1])
        type = namesplited[-1]
        if type not in self._typed_metrics.keys():
          raise ValueError(f'Unknown metric type: {type}')
      self._metrics_type_dict[name] = (newname, type)
      return newname, type

    def update_value(k, v):
      nk, t = get_metric_type(k)
      self._typed_metrics[t][nk] = self._typed_metrics[t].get(nk, 0) + v

    def reduce(vlist, k):
      t = get_metric_type(k)[1]
      if t == 'image' or t == 'histogram':
        return vlist[0]
      return jnp.sum(jnp.asarray(vlist), axis=0)

    if self._last_step >= self._steps or len(self._unflushed_metrics) == 0:
      return

    # Clear old metrics
    self._typed_metrics['image'].clear()
    self._typed_metrics['histogram'].clear()

    # Split metrics by type
    # e.g. { 'scalar': { 's1': 1.0, 's2': 0.7 }, 'image': { 'i1': 1.0 } }
    # convert list of dict to dict of list
    metrics = {
        k: reduce([dic[k] for dic in self._unflushed_metrics], k)
        for k in self._unflushed_metrics[0].keys()
    }
    for k, v in metrics.items():
      update_value(k, v)
    self._unflushed_metrics.clear()
    self._last_step = self._steps

  def _check_nokeep(self, k, v):
    if any([nokeep in str(k) for nokeep in ['image', 'histogram']]):
      if self._steps - self._last_step != 1:
        return None
    return v

  def append(self, metrics: Dict[Text, Any]):
    if self._last_time is None and self._steps == 0:
      self._last_time = time.time()

    self._steps += 1
    self._unflushed_metrics.append(jax.tree_util.tree_map_with_path(
        self._check_nokeep, metrics))
    if self._steps - self._last_step >= self._flush_interval:
      self._flush()

  def write(self, step):
    self._flush()
    metrics = _reduce_fn(self._typed_metrics, self._steps)
    if self._steps > 0 and self._last_time is not None:
      cost_time = time.time() - self._last_time
      metrics['scalar'].update({
          'steps_per_second': self._steps / cost_time
      })

    for k, m in metrics.items():
      if len(m) > 0:
        self._write_fn[k](step, m)

    for k in self._typed_metrics.keys():
        self._typed_metrics[k].clear()

    self._writer.flush()
    self._steps = 0
    self._last_step = 0
    self._last_time = None
