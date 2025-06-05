import argparse
import os

import numpy as np
from loguru import logger as log
from tqdm import trange

from src.commons import read_yaml
from src.data import SupervisedGraphDataset
from src.dft.molecule import (
    build_mesh,
    build_one_electron,
    build_pyscf_mol,
    build_two_electron,
    tokenize_orbitals,
)


def build_molecule(atomic_number, position, basis_name, xc_method, grid_level=3):
    mol = build_pyscf_mol(atomic_number, position, basis_name=basis_name)

    # Compute one-electron integrals
    one_electron = build_one_electron(mol)
    two_electron = build_two_electron(mol)
    mesh = build_mesh(mol, grid_level=grid_level)

    orbital_tokens, orbital_index = tokenize_orbitals(mol)
    return one_electron, two_electron, mesh, orbital_tokens, orbital_index


def main(config):
    dataset_name = config["datamodule_args"]["dataset_name"]
    dataset_args = config["datamodule_args"]["dataset_args"]
    basis_name = config.get("basis_name", "sto-3g")
    xc_method = config.get("xc_method", "lda")
    grid_level = config.get("grid_level", 3)

    log.info(f"Processing {dataset_name} dataset with {basis_name} basis")

    dataset = SupervisedGraphDataset(
        dataset_name=dataset_name, dataset_args=dataset_args
    )

    """ Preprocess the dataset according to basis"""
    p_join = os.path.join
    sample_mol = dataset[0]

    (one_electron, two_electron, mesh, orbital_tokens, orbital_index) = build_molecule(
        sample_mol.atomic_number, sample_mol.position, basis_name, xc_method, grid_level
    )
    num_orbitals = one_electron.overlap.shape[0]
    num_grid_points = mesh.points.shape[0]
    ds_size = len(dataset)

    log.info(f"Dataset Size: {ds_size}")
    log.info(f"Number of Orbitals: {num_orbitals}")
    log.info(f"Grid Size: {num_grid_points}")

    basis_dir = p_join(dataset.processed_dir, basis_name)
    os.makedirs(basis_dir, exist_ok=True)

    log.info(f"Saving at {basis_dir}")

    overlap_memmap = np.memmap(
        p_join(basis_dir, "overlap.memmap"),
        mode="w+",
        dtype=np.float32,
        shape=(ds_size, num_orbitals, num_orbitals),
    )
    kinetic_memmap = np.memmap(
        p_join(basis_dir, "kinetic.memmap"),
        mode="w+",
        dtype=np.float32,
        shape=(ds_size, num_orbitals, num_orbitals),
    )
    nuclear_memmap = np.memmap(
        p_join(basis_dir, "nuclear.memmap"),
        mode="w+",
        dtype=np.float32,
        shape=(ds_size, num_orbitals, num_orbitals),
    )

    eri_memmap = np.memmap(
        p_join(basis_dir, "eri.memmap"),
        mode="w+",
        dtype=np.float32,
        shape=(ds_size, num_orbitals, num_orbitals, num_orbitals, num_orbitals),
    )

    grid_points_memmap = np.memmap(
        p_join(basis_dir, "grid_points.memmap"),
        mode="w+",
        dtype=np.float32,
        shape=(ds_size, num_grid_points, 3),
    )
    grid_weights_memmap = np.memmap(
        p_join(basis_dir, "grid_weights.memmap"),
        mode="w+",
        dtype=np.float32,
        shape=(
            ds_size,
            num_grid_points,
        ),
    )

    orbital_tokens_memmap = np.memmap(
        p_join(basis_dir, "orbital_tokens.memmap"),
        mode="w+",
        dtype=int,
        shape=(ds_size, num_orbitals),
    )
    orbital_index_memmap = np.memmap(
        p_join(basis_dir, "orbital_index.memmap"),
        mode="w+",
        dtype=int,
        shape=(ds_size, num_orbitals),
    )

    for i in trange(ds_size):
        data = dataset[i]
        (
            one_electron,
            two_electron,
            mesh,
            orbital_tokens,
            orbital_index,
        )
        one_electron = build_molecule(
            data.atomic_number, data.position, basis_name, xc_method, grid_level
        )

        overlap = one_electron.overlap
        kinetic = one_electron.kinetic
        nuclear = one_electron.nuclear

        eri = two_electron.eri

        points = mesh.points
        weights = mesh.weights

        overlap_memmap[i, :, :] = overlap
        kinetic_memmap[i, :, :] = kinetic
        nuclear_memmap[i, :, :] = nuclear

        eri_memmap[i, :, :, :, :] = eri

        grid_points_memmap[i, :, :] = points
        grid_weights_memmap[i, :] = weights

        orbital_tokens_memmap[i, :] = orbital_tokens
        orbital_index_memmap[i, :] = orbital_index

    overlap_memmap.flush()
    kinetic_memmap.flush()
    nuclear_memmap.flush()
    eri_memmap.flush()
    grid_points_memmap.flush()
    grid_weights_memmap.flush()
    orbital_tokens_memmap.flush()
    orbital_index_memmap.flush()


if __name__ == "__main__":
    # read config path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    args = parser.parse_args()

    config = read_yaml(args.config)
    main(config)
