import argparse
import os.path as osp
from collections import defaultdict
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from flax.training import orbax_utils
from loguru import logger as log
from tqdm import tqdm

import wandb
from src.commons import (
    create_graph_from_mol,
    hamiltonian_matrix,
    homo_lumo_gap,
    orbital_energy,
    run_pyscf_solver,
    total_energy,
)
from src.instantiate import (
    instantiate_datamodule,
    instantiate_logger,
    instantiate_lr_scheduler,
    instantiate_model,
    instantiate_optimizer,
)
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


def run(
    config: dict,
    checkpoint_manager,
):
    # check if debug mode
    debug = config.get("debug", False)
    no_logger = config.get("no_logger", False)

    seed = config.get("seed", 42)
    basis_name = config.get("basis_name", "sto-3g")
    xc_method = config.get("xc_method", "lda")
    grid_level = config.get("grid_level", 3)

    task_name = config.get("task_name", None)
    if debug:
        task_name = "debug_run"

    assert task_name is not None, "Task name not provided"

    key = jax.random.PRNGKey(seed)

    simulated_by = "-".join(args.result_dir.split(osp.sep)[-2:])
    logger = instantiate_logger(
        task_name=f"Sim Evaluation ({simulated_by}): {task_name}",
        logger_args=config["logger_args"],
        debug_mode=debug,
        no_logger=no_logger,
    )
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

    atomic_number = np.array(datamodule.dataset[0].atomic_number, dtype=int)

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

    _apply_fn = partial(
        state.apply_fn, variables=state.params, output_coefficients=True
    )

    batch_size = 8

    metrics_dict = {}
    for step in range(args.eval_every, args.num_steps + 1, args.eval_every):
        metrics_dict[step] = defaultdict(list)

        positions = np.load(
            osp.join(
                args.result_dir, f"simulated_pos_nsteps_{step}_se_{args.eval_every}.npy"
            )
        )

        for i in (pbar := tqdm(range(0, len(positions), batch_size), leave=False)):
            batch_pos = positions[i : i + batch_size]

            graph = [
                create_graph_from_mol(
                    atomic_number=atomic_number,
                    position=batch_pos[j],
                    basis_name=basis_name,
                    xc_method=xc_method,
                    grid_level=grid_level,
                )
                for j in range(len(batch_pos))
            ]

            graph = batch_data(graph)

            _, pred_coef = jax.lax.stop_gradient(
                jax.jit(jax.vmap(_apply_fn))(
                    atomic_number=graph.atomic_number,
                    position=graph.position,
                    senders=graph.senders,
                    receivers=graph.receivers,
                    hamiltonian=graph.hamiltonian,
                    orbital_tokens=graph.orbital_tokens,
                    orbital_index=graph.orbital_index,
                )
            )

            # skip if the gt result does not converge
            try:
                gt_coef = []
                for g in graph:
                    _, pyscf_coeff = run_pyscf_solver(
                        g.atomic_number,
                        g.position,
                        basis_name=basis_name,
                        xc_method=xc_method,
                        max_cycle=args.pyscf_max_steps,
                    )
                    gt_coef.append(pyscf_coeff)

                gt_coef = jnp.stack(gt_coef)
            except:
                continue

            metrics_dict_step = jax.jit(compute_metrics)(
                graph.hamiltonian, pred_coef, gt_coef
            )

            for key, value in metrics_dict_step.items():
                metrics_dict[step][key].append(value.item())

            mae = metrics_dict_step["energy_mae"]
            gt_energy = metrics_dict_step["gt_energy"]
            pred_energy = metrics_dict_step["pred_energy"]
            pbar.set_description(
                f"Pred Energy: {pred_energy} | GT Energy: {gt_energy} |Energy MAE: {mae}"
            )

            if debug:
                break

        for key, value in metrics_dict[step].items():
            metrics_dict[step][key] = jnp.mean(jnp.array(value)).item()

        print(metrics_dict[step])

    example = metrics_dict[list(metrics_dict.keys())[0]]
    table_from_dict = [
        [metrics_dict[row][col] for col in metrics_dict[row].keys()]
        for row in metrics_dict.keys()
    ]

    nsteps = list(metrics_dict.keys())
    table_from_dict = [
        [nsteps[i]] + table_from_dict[i] for i in range(len(table_from_dict))
    ]
    if logger:
        cols = ["num_sim_steps"] + list(example.keys())
        table = wandb.Table(
            rows=list(metrics_dict.keys()), columns=cols, data=table_from_dict
        )
        logger.run.log({"Metrics": table})

    description = ""
    for key, value in metrics_dict.items():
        description += f"{key} : {value} | "

    log.info(description)


if __name__ == "__main__":
    # read config path
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--result_dir", type=str, default="./results")
    parser.add_argument("--use_labels", action="store_true")
    parser.add_argument("--pyscf_max_steps", type=int, default=20000)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--eval_every", type=int, default=10)
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--no_logger", "-l", action="store_true")
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
