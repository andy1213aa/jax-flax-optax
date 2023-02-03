from absl import app
from absl import logging
from absl import flags
from clu import platform
import jax
from ml_collections import config_flags
import tensorflow as tf
import train

FLAGS = flags.FLAGS

flags.DEFINE_string('workdir', None, 'Directory to store model data.')

config_flags.DEFINE_config_file(
    'config',
    None,
    'File path to the training hyperparameter configuration.',
)

def main(argv):

    tf.config.experimental.set_visible_devices([], 'GPU')

    logging.info(f'JAX process: {jax.process_index()} / {jax.process_count()}')
    logging.info(f'JAX local devices: {jax.local_devices()}')
    
    platform.work_unit().set_task_status(f'process_index: {jax.process_index()}, '
                                       f'process_count: {jax.process_count()}')
    platform.work_unit().create_artifact(platform.ArtifactType.DIRECTORY,
                                        FLAGS.workdir, 'workdir')

    train.train_and_evaluate(FLAGS.config, FLAGS.workdir)   
    
if __name__ == '__main__':
    flags.mark_flags_as_required(['workdir', 'config'])
    app.run(main)
    