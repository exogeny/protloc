import os
import tensorflow as tf
from absl import logging
from omegaconf import OmegaConf


def parse_configuration(flags_obj, config_file=None):
  config_file = config_file or flags_obj.config_file
  if not isinstance(config_file, (list, tuple)):
    config_file = [config_file]
  configs = [OmegaConf.load(cfg) for cfg in config_file]
  if getattr(flags_obj, 'params_override', None):
    print('flags_obj.params_override:', flags_obj.params_override)
    configs.append(OmegaConf.from_dotlist(flags_obj.params_override))
  params_additional = {
    'workdir': flags_obj.workdir,
  }
  config = OmegaConf.merge(*configs, params_additional)
  # validate dataset
  for required_parameter in ['dataset']:
    if required_parameter not in config:
      raise ValueError('Missing required parameter: %s' % required_parameter)
  return config

def serialize_configuration(config: OmegaConf):
  workdir = config.workdir
  tf.io.gfile.makedirs(workdir)
  params_save_path = os.path.join(workdir, 'params.yaml')
  OmegaConf.save(config, params_save_path)
  logging.info('Configuration is saved to %s', params_save_path)
