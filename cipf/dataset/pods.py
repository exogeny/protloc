from absl import logging
import os
import functools
import jax.numpy as jnp
import tensorflow as tf
import tensorflow_datasets as tfds
from cipf.dataset import BaseDataset

from cipf.dataset.human_protein_atlas import HumanProteinAtlas


def decode_example(example,
                   ensg_id_map,
                   image_size,
                   classifer_names=[],
                   protein_only=False,
):
  data_dict = dict()
  image = tf.image.decode_png(example['image'], channels=4)
  mask = tf.image.decode_png(example['image/mask'], channels=1)

  for classifier_name in classifer_names:
    if classifier_name in ['ensg/id', 'ensg/name']:
      ensg = tf.py_function(
          ensg_id_map,
          inp=[example[classifier_name]],
          Tout=jnp.float32,
      )
      print(ensg)
      data_dict[classifier_name] = ensg
    else:
      data_dict[classifier_name] = example[classifier_name]

  # Using the protein channel, [microtuble, protein, nucleus, er]
  _, protein, nucleus, _ = tf.split(image, 4, axis=-1)
  protein = protein * mask

  if protein_only:
    image = tf.repeat(protein, 3, axis=-1)
  else:
    nucleus = nucleus * mask
    mask = mask * 255
    image = tf.concat([protein, nucleus, mask], axis=-1)
  # The method must be NEAREST_NEIGHBOR because we need uint8 output.
  image = tf.image.resize(image, image_size,
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  data_dict['image'] = image
  return data_dict


class PodsDataset(BaseDataset):
  """Proteomics Datasets

  This class is used to load proteomics datasets from the config file.
  """
  SUPPORTED_CELL_LINES = ['U2OS', 'HEK293', 'A-431', 'U-251MG', 'All']

  def __init__(self, **config):
    super().__init__(**config)
    cell_line = config.get('cell_line', 'U2OS')
    if cell_line not in self.SUPPORTED_CELL_LINES:
      raise ValueError(f'Only support {", ".join(self.SUPPORTED_CELL_LINES)}')
    ensg_ids_path = os.path.abspath(__file__).replace('pods.py', 'ensgs')
    ensg_ids_path = os.path.join(ensg_ids_path, f'{cell_line}.txt')
    ensg_ids = list(set([
        e for e in open(ensg_ids_path).read().splitlines() if e.startswith('ENSG')
    ]))
    self._ensg_ids_map = {e: i for i, e in enumerate(ensg_ids)}

    classifier_name = config.get('classifier_name', None)
    classifier_names = classifier_name or []
    if not isinstance(classifier_names, (list, tuple)):
      classifier_names = [classifier_names]
    self._classifier_names = classifier_names

    logging.info(f'Loading dataset: human_protein_atlas/{cell_line}')
    self._dataset_builder = tfds.builder(f'human_protein_atlas/{cell_line}')
    self._dataset_builder.download_and_prepare()
    self._info = self._dataset_builder.info
    self._info.splits[tfds.Split.VALIDATION] = self._info.splits[tfds.Split.TEST]

  @property
  def info(self):
    return self._info

  def initialize(self, split):
    split = split.replace(tfds.Split.VALIDATION, tfds.Split.TEST)

    features = {
      'image': True,
      'image/mask': True,
    }
    decoders={
      'image': tfds.decode.SkipDecoding(),
      'image/mask': tfds.decode.SkipDecoding(),
    }
    for classifier_name in self._classifier_names:
      features[classifier_name] = True
      decoders[classifier_name] = tfds.decode.SkipDecoding()

    dataset = self._dataset_builder.as_dataset(
        split=split,
        decoders= tfds.decode.PartialDecoding(
            features=features,
            decoders=decoders,
        )
    )
    options = tf.data.Options()
    options.threading.private_threadpool_size = 48
    dataset = dataset.with_options(options)

    def _ensg_id_map(ensg_name):
      # if ensg_name not in self._ensg_ids_map:
      #   return tf.zeros(len(self._ensg_ids_map))
      ensg = self._ensg_ids_map[ensg_name.numpy().decode('utf-8')]
      return tf.one_hot(ensg, len(self._ensg_ids_map))

    image_size = self.config.get('image_size', 32)
    return dataset.map(
        functools.partial(
            decode_example,
            ensg_id_map=_ensg_id_map,
            image_size=(image_size, image_size),
            classifer_names=self._classifier_names,
            protein_only=self.config.get('protein_only', False)
        ),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

  def preprocess(self, example, split: str = tfds.Split.TRAIN):
    image_size = self.config.get('image_size', 32)
    image = example['image']

    if split in [tfds.Split.TRAIN] and self._augment is not None:
      image = self.distort_image(image)
    image = tf.image.resize(
        image, (image_size, image_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    image = tf.cast(image, tf.uint8)
    image = tf.image.convert_image_dtype(image, tf.float32)

    example['image'] = image # replace the image
    return example
