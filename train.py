from absl import app, flags, logging
import os
import jax
from clu import platform

from cipf import core
from cipf import trainlib
from cipf import define_flags

# disable tensorflow logging config XLA Flags for GPU performance
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.00'


def main(argv):
  FLAGS = flags.FLAGS
  configuration = core.parse_configuration(FLAGS)
  core.serialize_configuration(configuration)

  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  import tensorflow as tf
  tf.config.experimental.set_visible_devices([], 'GPU')
  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  platform.work_unit().set_task_status(
      f'process_index: {jax.process_index()}, '
      f'process_count: {jax.process_count()}'
  )
  trainlib.run_experiment(configuration)


if __name__ == '__main__':
  define_flags()
  jax.config.parse_flags_with_absl()
  flags.mark_flags_as_required(['workdir', 'config_file'])
  app.run(main)
