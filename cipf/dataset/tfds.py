import tensorflow as tf
import tensorflow_datasets as tfds
from absl import logging

from cipf.dataset import BaseDataset


SUPPORTED_DATASETS = [
  'imagenet2012'
]


class DatasetFromConfig(BaseDataset):
  def __init__(self,
               name: str,
               **kwargs):
    super().__init__(**kwargs)
    logging.info('Creating dataset with target: %s', name)
    if not any(name.startswith(x) for x in SUPPORTED_DATASETS):
      raise ValueError('Only support imagenet dataset for now, supporting '
                       f'datasets are: {SUPPORTED_DATASETS}')
    self._dataset_builder = tfds.builder(name)
    self._dataset_builder.download_and_prepare()

  def preprocess(self, example, split: str = tfds.Split.TRAIN):
    image_size = self.config.get('image_size', 32)
    image = example['image']
    label = example['label']

    if split in [tfds.Split.TRAIN] and self._augment is not None:
      image = self.distort_image(image)
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return (image, label)

  def initialize(self, split):
    return self._dataset_builder.as_dataset(split=split)

  @property
  def info(self):
    return self._dataset_builder.info
