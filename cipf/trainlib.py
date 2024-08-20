import os
import math
import functools

import tqdm
from time import time
from absl import logging

import jax
from jax import lax
from jax import random

import orbax.checkpoint
from flax import jax_utils
from flax import traverse_util
from clu import periodic_actions
from cipf import core
from cipf.dataset import init_dataset_from_config
from cipf.metrics import MetricsManager

# pmean only works inside pmap because it needs an axis name.
# This function will average the inputs across all devices.
cross_replica_mean = jax.pmap(lambda x: lax.pmean(x, 'x'), 'x')


def sync_batch_stats(state):
  if len(state.batch_stats) == 0:
    return state
  return state.replace(batch_stats=cross_replica_mean(state.batch_stats))


def replace_pretrain_weights(value, flatten_restored, scope):
  num_paired_keys = 0
  flatten_restored = {
      '/'.join(k[1:]): v
      for k, v in flatten_restored.items()
      if scope in k
  }

  flatten_value = traverse_util.flatten_dict(value)
  # pair the parameters
  for key in flatten_value.keys():
    possible_keys = ['/'.join(key[i:]) for i in range(len(key))]
    paired = [k in flatten_restored for k in possible_keys]
    if any(paired):
      num_paired_keys += 1
      index = paired.index(True)
      flatten_value[key] = flatten_restored[possible_keys[index]]
  return flatten_value, num_paired_keys


def restore_pretrain_weights(init_checkpoint, state):
  if not init_checkpoint:
    return state

  init_checkpoint = os.path.abspath(init_checkpoint)
  logging.info('Restoring weights from %s', init_checkpoint)
  orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
  restored = orbax_checkpointer.restore(init_checkpoint)
  flatten_restored = traverse_util.flatten_dict(restored)

  flatten_params, num_paired_params = replace_pretrain_weights(
      state.params, flatten_restored, 'params')
  flatten_batch_stats, num_paired_batch_stats = replace_pretrain_weights(
      state.batch_stats, flatten_restored, 'batch_stats')

  if num_paired_params + num_paired_batch_stats > 0:
    state = state.replace(
        params=traverse_util.unflatten_dict(flatten_params),
        batch_stats=traverse_util.unflatten_dict(flatten_batch_stats),
    )
    logging.info(f'Restored weights completed: {num_paired_params} parameters '
                 f'and {num_paired_batch_stats} batch stats are restored.')
  else:
    logging.info('No paired keys found, skipping weights restoration.')
  return state


def train_step(state, batch, rng_key):
  def loss_fn(params):
    variables = {'params': params, 'batch_stats': state.batch_stats}
    rng1, rng2, rng3 = random.split(rng_key, num=3)
    outputs, updates = state.apply_fn(
        variables,
        batch,
        training=True,
        rngs={
            'default': rng1,
            'dropout': rng2,
        },
        mrng = rng3, # model rng
        mutable=['batch_stats'],
    )
    metrics = state.compute_metrics_fn(outputs, batch)
    loss = metrics['losses/total_loss']
    return loss, (metrics, updates)

  step = state.step
  lr = state.learning_rate_fn(step)

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  aux, grads = grad_fn(state.params)
  metrics, updates = aux[1]
  metrics['learning_rate'] = lr
  grads = lax.pmean(grads, axis_name='batch')
  batch_stats = lax.pmean(updates['batch_stats'], axis_name='batch')
  state = state.apply_gradients(grads=grads, batch_stats=batch_stats)
  return state, metrics


def test_step(state, batch, rng_key):
  variables = {'params': state.params, 'batch_stats': state.batch_stats}
  rng1, rng2, rng3 = random.split(rng_key, num=3)
  outputs = state.apply_fn(
      variables,
      batch,
      training=False,
      rngs={
            'default': rng1,
            'dropout': rng2,
      },
      mrng = rng3, # model rng
      mutable=False,
  )
  metrics = state.compute_metrics_fn(outputs, batch)
  return metrics


def should_update_bar(bar: tqdm.tqdm, info, force: bool = False):
  if info is None:
    bar.update()
    return (1, bar.last_print_n, bar.last_print_t)

  count, last_print_n, last_print_t = info
  if force:
    bar.update(count - last_print_n)
    return (count, bar.last_print_n, bar.last_print_t)

  count += 1
  min_start_t = bar.start_t + bar.delay
  if count - last_print_n >= bar.miniters:
    cur_t = time()
    dt = cur_t - bar.last_print_t
    if dt >= bar.mininterval and cur_t >= min_start_t:
      bar.update(count - last_print_n)
      last_print_n = bar.last_print_n
      last_print_t = bar.last_print_t
  return (count, last_print_n, last_print_t)


def run_experiment(config, mode: str = 'train_and_test'):
  workdir = os.path.abspath(config.workdir)
  writer_flush_interval = config.get('writer_flush_interval', 100)
  train_writer = MetricsManager(f'{workdir}/train', writer_flush_interval)
  if mode == 'train_and_test':
    test_writer = MetricsManager(f'{workdir}/test', writer_flush_interval)
  elif mode == 'test':
    test_writer = MetricsManager(
        f'{workdir}/test_on_one_epoch', writer_flush_interval, logging=True)

  logging.info("Loading dataset...")
  dataset = init_dataset_from_config(config.dataset)
  logging.info(dataset.info)

  num_epochs = config.trainer.num_epochs
  batch_size = config.dataset.parameters.batch_size * jax.process_count()
  num_train_examples = dataset.info.splits['train'].num_examples
  steps_per_epoch = math.ceil(num_train_examples / batch_size)
  num_steps = int(steps_per_epoch * num_epochs)
  num_test_examples = dataset.info.splits['validation'].num_examples
  steps_per_test = math.ceil(num_test_examples / batch_size)
  # update the steps_per_epoch in the config
  config.trainer.steps_per_epoch = steps_per_epoch

  logging.info("Building model...")
  init_rng = random.PRNGKey(config.get('seed', 42))
  model = core.init_model_from_config(config.model)
  logging.info("Creating training state...")
  image_size = dataset.config.get('image_size', 224)
  state = core.init_initial_state(config, model,
                                  rng_key=init_rng,
                                  init_data=model.empty_data(image_size))
  checkpoint_manager = orbax.checkpoint.CheckpointManager(
      workdir,
      options=orbax.checkpoint.CheckpointManagerOptions(
          max_to_keep=10,
          create=False,
      ),
  )
  step_offset = checkpoint_manager.latest_step() or 0
  if step_offset > 0:
    # Restore the training state from checkpoints
    state = checkpoint_manager.restore(
        step_offset,
        args=orbax.checkpoint.args.PyTreeRestore(state)
    )
    step_offset += 1 # start from the next step.
  else:
    # Restore the weights from pretrained model
    state = restore_pretrain_weights(config.get('init_checkpoint'), state)
  state = jax_utils.replicate(state)

  rng, rng_key = random.split(init_rng)
  p_train_step = jax.pmap(train_step, axis_name='batch')
  p_test_step = jax.pmap(functools.partial(
      test_step,
      rng_key=rng_key
  ), axis_name='batch')

  if mode == 'train_and_test':
    logging.info('Initial compilation, this might take some minutes...')
    # Disable logging from absl
    training_log = os.path.join(workdir, 'training.log')
    logging.get_absl_handler().python_handler.stream = open(training_log, 'a')

  if mode == 'train_and_test':
    with tqdm.trange(num_epochs, position=0, desc=f'Epoch')       as epbar, \
        tqdm.trange(steps_per_epoch, position=1, desc='Training') as tpbar, \
        tqdm.trange(steps_per_test, position=2, desc='Testing')   as vpbar:
      epbar.update(int(step_offset / steps_per_epoch))
      tpbar.update(int(step_offset % steps_per_epoch))
      epbar.refresh()
      tpbar.refresh()

      tpbar_info, vpbar_info = None, None
      for step, batch in zip(range(step_offset, num_steps), dataset.train_iter()):
        with jax.profiler.StepTraceAnnotation('train', step_num=step):
          rng, key = random.split(rng)
          rng_key = jax_utils.replicate(key[...])
          state, metrics = p_train_step(state, batch, rng_key)
          train_writer.append(metrics)
        tpbar_info = should_update_bar(tpbar, tpbar_info)

        if (step + 1) % steps_per_epoch == 0:
          should_update_bar(tpbar, tpbar_info, force=True)
          vpbar.reset()

          # sync batch statistics across replicas
          state = sync_batch_stats(state)

          # testing loop
          for test_batch in dataset.valid_iter():
            metrics = p_test_step(state, test_batch)
            test_writer.append(metrics)
            vpbar_info = should_update_bar(vpbar, vpbar_info)
          should_update_bar(vpbar, vpbar_info, force=True)

          # save checkpoint
          saved_state = jax.device_get(
              jax.tree_util.tree_map(lambda x: x[0], state))
          checkpoint_manager.save(step,
                                  args=orbax.checkpoint.args.PyTreeSave(saved_state))

          # logging metrics
          train_writer.write(epbar.n + 1)
          test_writer.write(epbar.n + 1)

          # update progress bars
          tpbar_info, vpbar_info = None, None
          tpbar.reset()
          epbar.update(1)
  elif mode == 'test':
    # testing loop
    for test_batch in tqdm.tqdm(dataset.valid_iter(), total=steps_per_test):
      metrics = p_test_step(state, test_batch)
      test_writer.append(metrics)
    test_writer.write(1)
  else:
    raise ValueError(f'Unknown mode: {mode}')

  # Wait until computations are done before exiting
  jax.random.normal(random.PRNGKey(0), ()).block_until_ready()
  return state
