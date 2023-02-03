"""Default Hyperparameter configuration."""

import ml_collections


def get_config():
    """Get the default hyperparameter configuration."""

    config = ml_collections.ConfigDict()
    config.model = 'ResNet18'
    config.dataset = 'mnist'
    config.learning_rate = 0.1
    config.warmup_epochs = 5.0
    config.momentum = 0.9
    config.num_epochs = 100
    config.batch_size = 32
    config.half_precision = False
    config.cache = False
    # If num_train_steps==-1 then the number of training steps is calculated from
    # num_epochs using the entire dataset. Similarly for steps_per_eval.
    config.num_train_steps = -1
    config.steps_per_eval = -1

    return config