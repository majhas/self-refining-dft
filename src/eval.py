import logging
import os

import rootutils
from dotenv import load_dotenv

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.95"
logging.getLogger("absl").setLevel(logging.ERROR)

rootutils.setup_root(__file__, pythonpath=True)
load_dotenv(override=True)

from collections import defaultdict
from functools import partial

import hydra
import jax
import jax.numpy as jnp
import numpy as np
from hydra.utils import instantiate, to_absolute_path
from loguru import logger as log
from omegaconf import DictConfig
from tqdm import tqdm

from src.commons.graph import batch_data
from src.commons.logger import log_hyperparameters
from src.dft.property import (
    fock_matrix,
    homo_lumo_gap,
    orbital_energy,
    run_pyscf_solver,
    total_energy,
)
from src.trainer.utils import init_checkpoint_manager


def compute_metrics(hamiltonian, pred_coef, gt_coef, z, pos):
    # Total energy error
    pred_energy = jax.vmap(total_energy)(
        H=hamiltonian, C=pred_coef, atomic_number=z, position=pos
    )
    gt_energy = jax.vmap(total_energy)(
        H=hamiltonian, C=gt_coef, atomic_number=z, position=pos
    )

    energy_mae = jnp.mean(abs(pred_energy - gt_energy))

    # Orbital Energy
    pred_orb_energy = jax.vmap(orbital_energy)(H=hamiltonian, C=pred_coef)
    gt_orb_energy = jax.vmap(orbital_energy)(H=hamiltonian, C=gt_coef)
    orb_energy_mae = jnp.mean(abs(pred_orb_energy - gt_orb_energy))

    # hamiltonian matrix
    pred_H = jax.vmap(fock_matrix)(H=hamiltonian, C=pred_coef)
    gt_H = jax.vmap(fock_matrix)(H=hamiltonian, C=gt_coef)
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


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval.yaml")
def main(cfg: DictConfig):
    jax.config.update("jax_enable_x64", True)
    jax.config.update("jax_default_matmul_precision", "high")

    debug = cfg.debug
    seed = cfg.seed if cfg.seed is not None else 42
    wandbid = cfg.logger.get("id", None)
    if wandbid is None:
        wandbid = str(np.random.randint(int(1e7), int(1e8)))
        cfg.logger.id = wandbid

    task_name = "debug" if debug else cfg.task_name
    assert task_name is not None, "Please set task_name in your config!"
    key = jax.random.PRNGKey(seed)

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)

    logger = instantiate(cfg.logger)

    if debug or cfg.no_logger:
        logger = None

    # log config and model to wandb
    log_hyperparameters(logger=logger, object_dict={"cfg": cfg, "model": model})

    ckpt_dir = cfg.get("ckpt_dir", None)
    assert ckpt_dir is not None, "Need ckpt_dir set to evaluate model"
    
    ckpt_dir = to_absolute_path(ckpt_dir)

    checkpoint_manager = init_checkpoint_manager(ckpt_dir)
    restored_state = checkpoint_manager.restore(
        checkpoint_manager.best_step(),
    )
    params = restored_state["params"]
    apply_fn = partial(model.apply, variables=params, output_coefficients=True)

    test_dataloader = datamodule.test_dataloader(size=cfg.get("num_eval"))

    metrics_dict = defaultdict(list)
    start = 0

    for graph in (pbar := tqdm(test_dataloader, leave=False)):
        bs = len(graph)
        end = start + bs
        batch = batch_data(graph)

        _, pred_coef = jax.lax.stop_gradient(
            jax.jit(jax.vmap(apply_fn))(
                atomic_number=batch.atomic_number,
                position=batch.position,
                senders=batch.senders,
                receivers=batch.receivers,
                hamiltonian=batch.hamiltonian,
                orbital_tokens=batch.orbital_tokens,
                orbital_index=batch.orbital_index,
            )
        )

        gt_coef = []
        for g in graph:
            _, pyscf_coeff = run_pyscf_solver(
                g.atomic_number,
                g.position,
                basis_name=cfg.basis_name,
                xc_method=cfg.xc_method,
                max_cycle=cfg.pyscf_max_steps,
            )
            gt_coef.append(pyscf_coeff)

        gt_coef = jnp.stack(gt_coef)

        metrics_dict_step = jax.jit(compute_metrics)(
            batch.hamiltonian, pred_coef, gt_coef, batch.atomic_number, batch.position
        )

        for key, value in metrics_dict_step.items():
            metrics_dict[key].append(value.item())

        mae = metrics_dict_step["energy_mae"]
        gt_energy = metrics_dict_step["gt_energy"]
        pred_energy = metrics_dict_step["pred_energy"]
        pbar.set_description(
            f"Pred Energy: {pred_energy} | GT Energy: {gt_energy} |Energy MAE: {mae}"
        )

        start = end

        if debug:
            break

        if cfg.num_eval is not None and start >= cfg.num_eval:
            break

    for key, value in metrics_dict.items():
        value = jnp.array(value)
        if cfg.num_eval is not None:
            value = value[: cfg.num_eval]
        metrics_dict[key] = jnp.mean(value).item()

    print(metrics_dict)
    table_from_dict = [[metrics_dict[col] for col in metrics_dict.keys()]]
    if logger:
        logger.log_table(columns=list(metrics_dict.keys()), data=table_from_dict)

    description = ""
    for key, value in metrics_dict.items():
        description += f"{key} : {value} | "

    log.info(description)


if __name__ == "__main__":
    main()
