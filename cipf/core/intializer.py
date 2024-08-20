from typing import Any, Callable, Mapping, Optional
import functools
import importlib
from absl import logging

import optax
import jax
import jax.numpy as jnp
from flax import struct
from flax import traverse_util
from flax.core import unfreeze
from flax.linen import Module
from flax.training import train_state

class TrainState(train_state.TrainState):
  batch_stats: Any
  learning_rate_fn: Callable = struct.field(pytree_node=False)
  compute_metrics_fn: Callable = struct.field(pytree_node=False)


class dictproxy(dict):
  __getattr__ = dict.get
  __setattr__ = dict.__setitem__
  __delattr__ = dict.__delitem__

  def __hash__(self):
    return hash(frozenset(self.items()))

  def __eq__(self, other):
    return (isinstance(other, dictproxy) and self.items() == other.items())

def get_cls_from_target(target):
  module, clsname = target.rsplit('.', 1)
  module = importlib.import_module(module)
  cls = getattr(module, clsname)
  return cls


def init_module(target, parameters: Optional[Mapping] = None):
  cls = get_cls_from_target(target)
  parameters = parameters or {}
  return cls(**parameters)


def init_model_from_config(config, **kwargs) -> Module:
  target = config.get('target', None)
  if target is None:
    raise ValueError('The parameter of `target` is not specified in model '
                     'config')

  model_cls = get_cls_from_target(target)
  parameters = config.get('parameters', {})
  parameters.update(kwargs)
  model = model_cls(**parameters)
  model_cls = model.__class__ # model_cls may be the function, so reassign it
  # check if the model has a metrics_fn
  if not hasattr(model_cls, 'metrics_fn'):
    raise ValueError('Model must have a metrics_fn method.')
  # copy all attributes from the model to the model_cls for metrics to work.
  attributes = {}
  for key, value in model.__dict__.items():
    if key.startswith('_'): # ignore private attributes
      continue
    if isinstance(value, dict):
      value = dictproxy(value)
    attributes[key] = value
  setattr(model_cls.metrics_fn, '_attributes', dictproxy(attributes))
  return model


def init_scheduler_from_config(config, steps_per_epoch):
  if 'target' in config:
    target = config.target
    parameters = config.get('parameters', {})
    logging.info(f'Creating schedule: {target}')

    # replace epoch with steps
    replaced_parameters = {}
    for key, value in parameters.items():
      if 'epoch' in key:
        replaced_parameters[key.replace('epochs', 'steps')] = (
            value * steps_per_epoch)
      else:
        replaced_parameters[key] = value
    parameters = replaced_parameters

    if 'join_schedules' in target:
      schedules_config = parameters['schedules']
      schedules = [
          init_scheduler_from_config(subconfig)
          for subconfig in schedules_config
      ]
      return optax.join_schedules(
          schedules=schedules,
          boundaries=parameters['boundaries'])
    return init_module(target, parameters)
  raise ValueError('Invalid schedule config because `target` is missing.')  


def init_optax_from_config(config, learning_rate_fn):
  target = config.get('target', None)
  if target is None:
    raise ValueError('The parameter of `target` is not specified in model '
                     'config')

  logging.info(f'Creating optimizer: {target}')
  parameters = dict(config.get('parameters', {}))
  parameters.update({
      'learning_rate': learning_rate_fn,
  })
  return init_module(target, parameters)


def zero_grads():
  # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
  def init_fn(_):
    return ()
  def update_fn(updates, state, params=None):
    return jax.tree_map(jnp.zeros_like, updates), ()
  return optax.GradientTransformation(init_fn, update_fn)


def init_batched_data(init_data, batch_size):
  init_data['__mask'] = jnp.ones((), jnp.bool_)  # for pad_shard_unpad
  return jax.tree_util.tree_map(
      lambda x: jnp.repeat(x, batch_size, axis=0), init_data)

def init_initial_state(config, model, rng_key, init_data, training: bool = True):
  tx = None
  learning_rate_fn = None
  batch_size = config.dataset.parameters.batch_size
  init_data = init_batched_data(init_data, batch_size)

  def initialize(*arg, **kwargs):
    return model.init(*arg, **kwargs)

  jitted_initialize_func = jax.jit(functools.partial(
      initialize,
      training=True,
  ))
  variables = jitted_initialize_func(
    {'params': rng_key, 'dropout': rng_key},
    init_data,
  )
  params, batch_stats = variables['params'], variables.get('batch_stats', dict())
  print(model.tabulate(rng_key, init_data))

  if training:
    trainer_config = config.trainer
    learning_rate_fn = init_scheduler_from_config(trainer_config.scheduler,
                                                  trainer_config.steps_per_epoch)
    tx = init_optax_from_config(trainer_config.optimizer, learning_rate_fn)
    freezed_params = config.get('freezed_params', None)
    if freezed_params is not None:
      flatten_params = traverse_util.flatten_dict(params)
      flatten_params_keys = ['/'.join(k) for k in flatten_params.keys()]
      param_labels = {
        key:
          'zero'
            if any([key.startswith(prefix) for prefix in freezed_params])
            else 'optax'
        for key in flatten_params_keys
      }
      # label_fn is used to label the parameters for optax
      def label_fn(nested_dict, parent_key=None):
        parent_key = f'{parent_key}/' if parent_key is not None else ''
        return {
          k: (
            label_fn(v, parent_key=f'{parent_key}{k}')
              if isinstance(v, dict)
              else
            param_labels[f'{parent_key}{k}']
          ) for k, v in nested_dict.items()
        }
      tx = optax.multi_transform({
        'optax': tx,
        'zero': zero_grads(),
      }, label_fn)
      logging.info(
          'Freezed parameters: %s[%d/%d]',
          ','.join(freezed_params),
          sum([v == "zero" for v in param_labels.values()]),
          len(param_labels),
      )

  attributes = model.metrics_fn._attributes
  metrics_fn = jax.jit(functools.partial(model.metrics_fn, attributes))

  state = TrainState.create(
    apply_fn=model.apply,
    params=unfreeze(params),
    batch_stats=batch_stats,
    tx=tx,
    learning_rate_fn=learning_rate_fn if training else None,
    compute_metrics_fn=lambda outputs, batch: metrics_fn(outputs, batch),
  )
  return state
