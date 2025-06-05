import argparse
import os

import numpy as np
from loguru import logger as log
from tqdm import tqdm

from src.commons import read_yaml, run_pyscf_solver
from src.instantiate import instantiate_datamodule


def main(config):
    datamodule_args = config["datamodule_args"]

    # datamodule_args["dataset_type"] = "SupervisedGraphDataset"
    basis_name = config.get("basis_name", "sto-3g")
    xc_method = config.get("xc_method", "lda")
    grid_level = config.get("grid_level", 3)
    dataset_name = datamodule_args["dataset_name"]

    log.info(
        f"Processing coefficients for {dataset_name} dataset"
        + f" with {basis_name} basis and {xc_method}"
    )

    datamodule_args["batch_size"] = 4
    datamodule = instantiate_datamodule(
        datamodule_type="DataModule",
        datamodule_args=datamodule_args,
        basis_name=basis_name,
        xc_method=xc_method,
        grid_level=grid_level,
        seed=config.get("seed", 42),
    )

    test_dataloader = datamodule.test_dataloader()

    """ Preprocess the dataset according to basis"""
    p_join = os.path.join
    sample_mol = test_dataloader.dataset[0]
    basis = sample_mol.hamiltonian.basis

    ds_size = datamodule.test_size
    num_orbitals = basis.num_orbitals

    log.info(f"Dataset Size: {ds_size}")
    log.info(f"Num Orbitals: {num_orbitals}")

    save_dir = p_join(datamodule.dataset.processed_dir, basis_name)
    os.makedirs(save_dir, exist_ok=True)

    log.info(f"Saving at {save_dir}")

    coefficient_memmap = np.memmap(
        p_join(save_dir, "coefficient.memmap"),
        mode="w+",
        dtype=np.float32,
        shape=(ds_size, num_orbitals, num_orbitals),
    )

    # forces_memmap = np.memmap(
    #     p_join(save_dir, "forces.memmap"),
    #     mode="w+",
    #     dtype=np.float32,
    #     shape=(ds_size, num_atoms, 3),
    # )

    # energy_memmap = np.memmap(
    #     p_join(save_dir, "mess_energy.memmap"),
    #     mode="w+",
    #     dtype=np.float32,
    #     shape=(ds_size, 1),
    # )

    batch_size = test_dataloader.batch_size
    start = 0
    end = 0
    buffer_size = batch_size * 4  # Adjust based on available memory
    buffer = np.zeros((buffer_size, num_orbitals, num_orbitals), dtype=np.float32)
    buffer_count = 0

    for j, batch in enumerate(tqdm(test_dataloader)):
        bs = len(batch)
        C = []
        for g in batch:
            _, pyscf_coeff = run_pyscf_solver(
                g.atomic_number,
                g.position,
                basis_name=basis_name,
                xc_method=xc_method,
                max_cycle=args.pyscf_max_steps,
            )
            C.append(pyscf_coeff)

        C = np.stack(C)

        buffer[buffer_count : buffer_count + bs] = C.astype(np.float32)
        buffer_count += bs
        end += bs

        if buffer_count >= buffer_size:
            coefficient_memmap[start:end, :, :] = buffer[:buffer_count]
            coefficient_memmap.flush()

            buffer_count = 0
            start = end

    # Handle remaining data in the buffer
    if buffer_count > 0:
        coefficient_memmap[start:end, :, :] = buffer[:buffer_count]
        coefficient_memmap.flush()


if __name__ == "__main__":
    # read config path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    args = parser.parse_args()

    config = read_yaml(args.config)
    main(config)
