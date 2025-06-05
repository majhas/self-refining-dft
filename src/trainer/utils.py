import os.path as osp
from typing import Any

import jax
from flax.training.train_state import TrainState as FlaxTrainState
from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)


class TrainState(FlaxTrainState):
    key: Any
    step_size: float = 0.0
    num_generated: int = 0


def shard(xs, device_count=None):
    if device_count is None:
        device_count = jax.local_device_count()
    return jax.tree_util.tree_map(
        lambda x: x.reshape((device_count, -1) + x.shape[1:]), xs
    )


def unreplicate(x):
    return jax.tree_util.tree_map(lambda y: y[0], x)


def count_parameters(params):
    return sum(x.size for x in jax.tree.leaves(params))


def get_lr_from_opt(state):
    if isinstance(state.opt_state, tuple):
        return state.opt_state[-1].hyperparams["learning_rate"]
    else:
        return state.opt_state.hyperparams["learning_rate"]


def init_checkpoint_manager(
    ckpt_dir,
    config={},
    max_to_keep=3,
    create=True,
    best_mode="min",
    keep_latest: bool = False,
):
    if keep_latest:
        ckpt_dir = osp.join(ckpt_dir, "latest")
        options = CheckpointManagerOptions(
            max_to_keep=1,
            step_prefix="latest",
            create=create,
        )
    else:
        ckpt_dir = osp.join(ckpt_dir, "best")
        options = CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            create=create,
            best_mode=best_mode,
            best_fn=lambda metrics: metrics["valid_loss"],
            keep_checkpoints_without_metrics=True,
        )

    checkpoint_manager = CheckpointManager(
        ckpt_dir, PyTreeCheckpointer(), options, metadata={"config": config}
    )
    checkpoint_manager.reload()
    return checkpoint_manager
