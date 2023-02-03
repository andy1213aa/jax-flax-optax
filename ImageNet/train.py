import functools
import time
from typing import Any

from absl import logging
from clu import metric_writers
from clu import periodic_actions
import flax
from flax import jax_utils
from flax.training import checkpoints
from flax.training import common_utils
from flax.training import dynamic_scale as dynamic_scale_lib
from flax.training import train_state
import jax
from jax import lax
import jax.numpy as jnp
from jax import random as jrand
import ml_collections
import optax
import tensorflow as tf
import tensorflow_datasets as tfds

import input_pipeline
import models

NUM_CLASSES = 10


class TrainState(train_state.TrainState):
    batch_stats: Any
    dynamic_scale: dynamic_scale_lib.DynamicScale

def restore_checkpoint(state, workdir):
    return checkpoints.restore_checkpoint(workdir, state)

def create_model(*, model_cls, half_precision, **kwargs):
    platform = jax.local_devices()[0].platform  # tpu or gpu
    if half_precision:
        if platform == 'tpu':
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float16
    else:
        model_dtype = jnp.float32

    return model_cls(num_classes=NUM_CLASSES, dtype=model_dtype, **kwargs)


def create_train_state(config, model, image_size, learning_rate_fn, rng):
    """Create initial training state."""

    platform = jax.local_devices()[0].platform

    if config.half_precision and platform == 'gpu':
        dynamic_scale = dynamic_scale_lib.DynamicScale()
    else:
        dynamic_scale = None
    params, batch_stats = initialized(rng, image_size, model)

    tx = optax.sgd(learning_rate=learning_rate_fn,
                   momentum=config.momentum,
                   nesterov=True)

    state = TrainState.create(apply_fn=model.apply,
                              params=params,
                              tx=tx,
                              batch_stats=batch_stats,
                              dynamic_scale=dynamic_scale)
    return state


def create_learning_rate_fn(config, base_learning_rate: float,
                            steps_per_epoch: int):
    """Create learning rate schedule."""
    warmup_fn = optax.linear_schedule(init_value=0.,
                                      end_value=base_learning_rate,
                                      transition_steps=config.warmup_epochs *
                                      steps_per_epoch)

    cosine_epochs = max(config.num_epochs - config.warmup_epochs, 1)
    cosine_fn = optax.cosine_decay_schedule(init_value=base_learning_rate,
                                            decay_steps=cosine_epochs *
                                            steps_per_epoch)
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[config.warmup_epochs * steps_per_epoch])

    return schedule_fn


def create_input_iter(dataset_builder, batch_size, image_size, dtype, train,
                      cache):
    ds = input_pipeline.create_split(dataset_builder,
                                     batch_size,
                                     image_size=image_size,
                                     dtype=dtype,
                                     train=train,
                                     cache=cache)
    it = map(prepare_tf_data, ds)
    it = jax_utils.prefetch_to_device(it, 2)
    return it


def initialized(key, image_size, model):
    input_shape = (1, image_size, image_size, 1)

    @jax.jit
    def init(*args):
        return model.init(*args)

    variables = init({'params': key}, jnp.ones(input_shape, model.dtype))

    return variables['params'], variables['batch_stats']


def prepare_tf_data(xs):
    """Convert a input batch from tf Tensors to numpy arrays."""
    local_device_count = jax.local_device_count()

    def _prepare(x):
        # Use _numpy() for zero-copy conversion between TF and NumPy.
        x = x._numpy()  # pylint: disable=protected-access

        # reshape (host_batch_size, height, width, 3) to
        # (local_devices, device_batch_size, height, width, 3)
        return x.reshape((local_device_count, -1) + x.shape[1:])

    return jax.tree_util.tree_map(_prepare, xs)


def train_and_evaluate(config, workdir: str):
    """Execute model training and evaluation loop.
        Args:
            config: Hyperparameter configuration for training and evaluation.
            workdir: Directory where the tensorboard summaries are written to.
        Returns:
            Final TrainState.
    """

    writer = metric_writers.create_default_writer(
        logdir=workdir, just_logging=jax.process_index() != 0)
    rng = jrand.PRNGKey(0)

    image_size = 224

    if config.batch_size % jax.device_count() > 0:
        raise ValueError(
            'Batch size must be divisible by the number of devices')

    local_batch_size = config.batch_size // jax.process_count()

    platform = jax.local_devices()[0].platform

    if config.half_precision:
        if platform == 'tpu':
            inputdtype = tf.bfloat16
        else:
            input_dtype = tf.float16
    else:
        input_dtype = tf.float32

    dataset_builder = tfds.builder(config.dataset)

    train_iter = create_input_iter(dataset_builder,
                                   local_batch_size,
                                   image_size,
                                   input_dtype,
                                   train=True,
                                   cache=config.cache)

    eval_iter = create_input_iter(dataset_builder,
                                  local_batch_size,
                                  image_size,
                                  input_dtype,
                                  train=False,
                                  cache=config.cache)

    steps_per_epoch = (dataset_builder.info.splits['train'].num_examples //
                       config.batch_size)

    if config.num_train_steps == -1:
        num_steps = int(steps_per_epoch * config.num_epochs)
    else:
        num_steps = config.num_train_steps

    if config.steps_per_eval == -1:
        num_validation_examples = dataset_builder.info.splits[
            'test'].num_examples
        steps_per_eval = num_validation_examples // config.batch_size
    else:
        steps_per_eval = config.steps_per_eval

    steps_per_checkpoint = steps_per_epoch * 10

    base_learning_rate = config.learning_rate * config.batch_size / 256.

    model_cls = getattr(models, config.model)
    model = create_model(model_cls=model_cls,
                         half_precision=config.half_precision)

    learning_rate_fn = create_learning_rate_fn(config, base_learning_rate,
                                               steps_per_epoch)

    state = create_train_state(config, model, image_size,
                               learning_rate_fn, rng)

    state = restore_checkpoint(state, workdir)
    # step_offset > 0 if restarting from checkpoint
    step_offset = int(state.step)
    state = jax_utils.replicate(state)

    

    return 0
