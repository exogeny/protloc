import abc
import math
import functools
import importlib
from absl import logging

import jax
import jax.numpy as jnp
import flax.jax_utils
import tensorflow as tf
import tensorflow_datasets as tfds

from cipf.dataset.auto_augment import AutoAugment


class BaseDataset:

  def __init__(self, **config):
    self._datasets = {}
    self._config = config
    self._augment = None
    augment_config = config.get('augment_config', None)
    if augment_config is not None and augment_config.name:
      self._augment = AutoAugment(augment_config.name,
                                  augment_config.cutout_const,
                                  augment_config.translate_const)
      logging.info('Using AutoAugment policy: %s', augment_config.name)

  def prefetch_to_device(self, dataset, batch_size = None):
    local_device_count = jax.local_device_count()

    def _prepare(x):
      x = x._numpy() # pylint: disable=protected-access

      if batch_size is not None and x.shape[0] != batch_size:
        padded = jnp.repeat(x[0:1], repeats=batch_size - x.shape[0], axis=0)
        x = jnp.concatenate([x, padded], axis=0)
      # reshape (host_batch_size, height, width, channel) to
      # (local_devices, device_batch_size, height, width, channel)
      return x.reshape((local_device_count, -1) + x.shape[1:])
    iterator = map(lambda batch: jax.tree_map(_prepare, batch), dataset)
    iterator = flax.jax_utils.prefetch_to_device(iterator, 2)
    return iterator

  def distort_image(self, image):
    if self._augment is not None:
      image = self._augment.distort(image)
    return image

  @property
  def config(self):
    return self._config

  @abc.abstractmethod
  def initialize(self, split):
    pass

  def preprocess(self, example, split: str = tfds.Split.TRAIN):
    del split
    return example

  def _preprocess(self, example, split: str = tfds.Split.TRAIN):
    example = self.preprocess(example, split)
    example['__mask'] = jnp.ones((), jnp.bool_) # for pad_shard_unpad
    return example

  @property
  @abc.abstractmethod
  def info(self):
    pass

  def _initialize(self, split):
    split_name = f'{split}:{jax.process_index()}'
    if split_name not in self._datasets:
      logging.info(f'initializing split: {split_name}')
      batch_size = self.config.get('batch_size', 32)
      num_examples = self.info.splits[split].num_examples
      split_size = math.ceil(num_examples / jax.process_count())
      start = jax.process_index() * split_size
      dataset = self.initialize(f'{split}[{start}:{start + split_size}]')
      if self.config.get('cache', False):
        dataset = dataset.cache()

      if split in [tfds.Split.TRAIN]:
        dataset = dataset.repeat()
        shuffle_buffer_size = self.config.get('shuffle_buffer_size', batch_size)
        dataset = dataset.shuffle(shuffle_buffer_size, seed=42)
      dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
      dataset = dataset.map(functools.partial(self._preprocess, split=split),
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)
      dataset = dataset.batch(batch_size,
                              drop_remainder=split in [tfds.Split.TRAIN])
      dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
      self._datasets[split_name] = dataset
    return self._datasets[split_name]

  def train_iter(self):
    dataset = self._initialize(tfds.Split.TRAIN)
    return self.prefetch_to_device(dataset)

  def valid_iter(self):
    batch_size = self.config.get('batch_size', 32)
    dataset = self._initialize(tfds.Split.VALIDATION)
    dataset_iter = tf.nest.map_structure(iter, dataset)
    return self.prefetch_to_device(dataset_iter, batch_size)

  def test_iter(self):
    batch_size = self.config.get('batch_size', 32)
    dataset = self._initialize(tfds.Split.TEST)
    dataset_iter = tf.nest.map_structure(iter, dataset)
    return self.prefetch_to_device(dataset_iter, batch_size)

  @staticmethod
  def init_dataset_from_config(config):
    target = config.get('target', None)
    if target is None:
      raise ValueError('The parameter of `target` is not specified in dataset '
                      'config')

    logging.info('Creating dataset with target: %s', target)
    module, clsname = target.rsplit('.', 1)
    dataset_module = importlib.import_module(module)
    dataset_cls = getattr(dataset_module, clsname)
    parameters = config.get('parameters', {})
    return dataset_cls(**parameters)
