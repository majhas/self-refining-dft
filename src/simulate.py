import argparse
import os
import os.path as osp
from copy import deepcopy

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import orbax_utils
from loguru import logger as log
from tqdm import tqdm

from src.commons import (
    create_graph_from_mol,
    hamiltonian_matrix,
    homo_lumo_gap,
    orbital_energy,
    total_energy,
)
from src.instantiate import (
    instantiate_datamodule,
    instantiate_lr_scheduler,
    instantiate_model,
    instantiate_optimizer,
)
from src.sampling import run_mcmc
from src.trainer.trainer import batch_data
from src.trainer.utils import TrainState, init_checkpoint_manager


def compute_metrics(hamiltonian, pred_coef, gt_coef):
    # Total energy error
    pred_energy = jax.vmap(total_energy)(H=hamiltonian, C=pred_coef)
    gt_energy = jax.vmap(total_energy)(H=hamiltonian, C=gt_coef)

    energy_mae = jnp.mean(abs(pred_energy - gt_energy))

    # Orbital Energy
    pred_orb_energy = jax.vmap(orbital_energy)(H=hamiltonian, C=pred_coef)
    gt_orb_energy = jax.vmap(orbital_energy)(H=hamiltonian, C=gt_coef)
    orb_energy_mae = jnp.mean(abs(pred_orb_energy - gt_orb_energy))

    # hamiltonian matrix
    pred_H = jax.vmap(hamiltonian_matrix)(H=hamiltonian, C=pred_coef)
    gt_H = jax.vmap(hamiltonian_matrix)(H=hamiltonian, C=gt_coef)
    H_mae = jnp.mean(abs(pred_H - gt_H))

    pred_homo, pred_lumo, pred_gap = jax.vmap(homo_lumo_gap)(H=hamiltonian, C=pred_coef)
    gt_homo, gt_lumo, gt_gap = jax.vmap(homo_lumo_gap)(H=hamiltonian, C=gt_coef)
    homo_mae = jnp.mean(abs(pred_homo - gt_homo))
    lumo_mae = jnp.mean(abs(pred_lumo - gt_lumo))
    gap_mae = jnp.mean(abs(pred_gap - gt_gap))

    return {
        "pred_energy": jnp.mean(pred_energy),
        "gt_energy": jnp.mean(gt_energy),
        "energy_mae": energy_mae,
        "orb_energy_mae": orb_energy_mae,
        "hamiltonian_matrix_mae": H_mae,
        "homo_mae": homo_mae,
        "lumo_mae": lumo_mae,
        "gap_mae": gap_mae,
    }


def init_model(
    key,
    model,
    optimizer,
    atomic_number,
    basis_name,
    xc_method,
    checkpoint_dir=None,
    checkpoint_manager=None,
):
    num_atoms = len(atomic_number)
    position = jnp.zeros((num_atoms, 3))

    sample = create_graph_from_mol(
        atomic_number, position, basis_name=basis_name, xc_method=xc_method
    )
    params = model.init(
        key,
        atomic_number=sample.atomic_number,
        position=sample.position,
        orbital_tokens=sample.orbital_tokens,
        orbital_index=sample.orbital_index,
        hamiltonian=sample.hamiltonian,
        senders=sample.senders,
        receivers=sample.receivers,
    )

    state = TrainState.create(
        key=key, apply_fn=model.apply, params=params, tx=optimizer, step_size=0
    )

    if (checkpoint_dir is not None) or (checkpoint_manager is not None):
        if checkpoint_manager is None:
            checkpoint_manager = init_checkpoint_manager(checkpoint_dir)

        restore_args = orbax_utils.restore_args_from_target(state)
        state = checkpoint_manager.restore(
            checkpoint_manager.best_step(),
            items=state,
            restore_kwargs={"restore_args": restore_args},
        )

    return state


def run(config: dict, checkpoint_manager):
    # check if debug mode
    debug = config.get("debug", False)

    seed = config.get("seed", 42)
    basis_name = config.get("basis_name", "sto-3g")
    xc_method = config.get("xc_method", "lda")
    grid_level = config.get("grid_level", 3)

    task_name = config.get("task_name", None)
    if debug:
        task_name = "debug_run"

    assert task_name is not None, "Task name not provided"

    key = jax.random.PRNGKey(seed)

    datamodule_args = config["datamodule_args"]
    datamodule_args["batch_size"] = 8

    datamodule = instantiate_datamodule(
        datamodule_type=config["datamodule_type"],
        datamodule_args=datamodule_args,
        basis_name=basis_name,
        xc_method=xc_method,
        grid_level=grid_level,
        seed=seed,
    )

    atomic_number = datamodule.dataset[0].atomic_number

    lr_scheduler_fn = instantiate_lr_scheduler(
        lr_scheduler_type=config.get("lr_scheduler_type", None),
        lr_scheduler_args=config.get("lr_scheduler_args", None),
    )

    optimizer_args = deepcopy(config["optimizer_args"])
    if lr_scheduler_fn is not None:
        optimizer_args["learning_rate"] = lr_scheduler_fn

    optimizer = instantiate_optimizer(
        optimizer_type=config["optimizer_type"], optimizer_args=optimizer_args
    )
    model = instantiate_model(
        model_type=config["model_type"], model_args=config["model_args"]
    )

    state = init_model(
        key,
        model,
        optimizer,
        atomic_number,
        basis_name,
        xc_method,
        checkpoint_manager=checkpoint_manager,
    )

    test_dataloader = datamodule.test_dataloader(size=args.num_eval)
    step_size = args.step_size
    M = int(args.num_steps / args.save_every)
    positions = np.zeros(shape=(args.num_eval, M, atomic_number.shape[0], 3))
    save_indices = np.array(
        [i for i in range(args.save_every, args.num_steps + 1, args.save_every)]
    )
    start = 0
    for data in tqdm(test_dataloader):
        bs = len(data)
        batch = batch_data(data)
        pos = batch.position

        key, loc_key = jax.random.split(key)
        _, step_size, _, batch_pos = run_mcmc(
            key=loc_key,
            params=state.params,
            apply_fn=state.apply_fn,
            atomic_number=batch.atomic_number,
            x0=pos,
            step_size=step_size,
            n_iter=args.num_steps,  # iterate n steps starting from last pos
            basis_name=basis_name,
            xc_method=xc_method,
            grid_level=grid_level,
            kernel_method=args.kernel_method,
            return_chain=True,
        )

        positions[start : start + bs, :] = batch_pos[:, save_indices]

        start += bs

    nsteps = args.save_every
    for i in range(0, M):
        dirpath = osp.join(args.result_dir, task_name)
        if not osp.exists(dirpath):
            os.makedirs(dirpath)

        path = osp.join(
            dirpath, f"simulated_pos_nsteps_{nsteps}_se_{args.save_every}.npy"
        )
        np.save(path, positions[:, i].reshape(args.num_eval, atomic_number.shape[0], 3))

        nsteps += args.save_every


if __name__ == "__main__":
    # read config path
    parser = argparse.ArgumentParser()
    # parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--result_dir", "-r", default="./results", type=str, required=False
    )
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--no_logger", "-l", action="store_true")
    parser.add_argument("--num_eval", "-e", type=int, default=500)
    parser.add_argument("--num_steps", "-n", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--step_size", "-s", type=float, default=2e-3)
    parser.add_argument("--kernel_method", "-k", type=str, default="mala")
    args = parser.parse_args()

    # read config
    osp.exists(args.checkpoint_dir), f"Checkpoint path {args.checkpoint_dir} not found"

    # load state
    log.info(f"Loading state from {args.checkpoint_dir}")
    checkpoint_manager = init_checkpoint_manager(args.checkpoint_dir)
    config = checkpoint_manager.metadata()["config"]

    # update config with debug mode
    config["debug"] = args.debug
    config["no_logger"] = args.no_logger

    run(
        config=config,
        checkpoint_manager=checkpoint_manager,
    )
