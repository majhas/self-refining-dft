import optax


def adamw(*args, **kwargs):
    return optax.inject_hyperparams(optax.adamw)(*args, **kwargs)
