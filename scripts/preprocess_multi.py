import argparse
import os
from multiprocessing import Pool

import numpy as np
from loguru import logger as log
from pyscf import dft, gto
from tqdm import tqdm

from src.commons.io import read_yaml
from src.data.md17_dataset import MD17Dataset

ORBITAL_TOKENS = {
    "1s": 0,
    "2s": 1,
    "2px": 2,
    "2py": 3,
    "2pz": 4,
    "3s": 5,
    "3px": 6,
    "3py": 7,
    "3pz": 8,
    "3dxx": 9,
    "3dxy": 10,
    "3dxz": 11,
    "3dyy": 12,
    "3dyz": 13,
    "3dzz": 14,
    "4s": 15,
    "4px": 16,
    "4py": 17,
    "4pz": 18,
    "4dxx": 19,
    "4dxy": 20,
    "4dxz": 21,
    "4dyy": 22,
    "4dyz": 23,
    "4dzz": 24,
    "4fxxx": 25,
    "4fxxy": 26,
    "4fxxz": 27,
    "4fxyy": 28,
    "4fxyz": 29,
    "4fyyz": 30,
    "4fzzz": 31,
    "5s": 32,
    "5px": 33,
    "5py": 34,
    "5pz": 35,
    "5dxx": 36,
    "5dxy": 37,
    "5dxz": 38,
    "5dyy": 39,
    "5dyz": 40,
    "5dzz": 41,
    "5fxxx": 42,
    "5fxxy": 43,
    "5fxxz": 44,
    "5fxyy": 45,
    "5fxyz": 46,
    "5fyyz": 47,
    "5fzzz": 48,
}


def tokenize_orbitals(mol: gto.Mole):
    ao_labels = mol.ao_labels()
    orbital_tokens = []
    orbital_index = []

    for label in ao_labels:
        label = label.strip().split(" ")
        index = label[0]
        orbital = label[-1]
        orbital_index.append(int(index))
        orbital_tokens.append(ORBITAL_TOKENS[orbital])

    orbital_tokens = np.array(orbital_tokens)
    orbital_index = np.array(orbital_index)
    return orbital_tokens, orbital_index


def build_one_electron(mol: gto.Mole):
    # Compute one-electron integrals
    S = mol.intor("int1e_ovlp_cart")
    diag = np.diagonal(S)
    diag = np.where((diag > 0), diag, 1.0)
    N = 1 / (np.sqrt(diag))
    overlap = N[:, np.newaxis] * N[np.newaxis, :] * S
    kinetic = mol.intor("int1e_kin_cart")
    nuclear = mol.intor("int1e_nuc_cart")

    # Organize data into named tuples
    return overlap, kinetic, nuclear


def build_two_electron(mol: gto.Mole):
    # Compute two-electron integrals
    eri = mol.intor("int2e_cart", aosym="s1")
    return eri


def build_mesh(mol: gto.Mole, grid_level: int = 3):
    # Generate grid for XC functional
    grids = dft.gen_grid.Grids(mol)
    grids.level = grid_level
    grids.build()

    points = grids.coords
    weights = grids.weights

    return points, weights


def build_pyscf_mol(atomic_number, position, basis_name="sto-3g"):
    if not isinstance(atomic_number, np.ndarray):
        atomic_number = np.asarray(atomic_number)

    if not isinstance(position, np.ndarray):
        position = np.asarray(position)

    num_electrons = np.sum(atomic_number)

    mol = gto.Mole(unit="Bohr", spin=num_electrons % 2, cart=True)
    mol.atom = [(symbol, pos) for symbol, pos in zip(atomic_number, position)]
    mol.basis = basis_name
    mol.build(unit="Bohr")

    return mol


def build_molecule(atomic_number, position, basis_name, xc_method, grid_level=3):
    position = position - position.mean(axis=0)[None, ...]

    mol = build_pyscf_mol(atomic_number, position, basis_name=basis_name)
    # Compute one-electron integrals
    overlap, kinetic, nuclear = build_one_electron(mol)
    eri = build_two_electron(mol)
    points, weights = build_mesh(mol, grid_level=grid_level)

    orbital_tokens, orbital_index = tokenize_orbitals(mol)
    return (
        overlap,
        kinetic,
        nuclear,
        eri,
        points,
        weights,
        orbital_tokens,
        orbital_index,
    )


def build_molecule_sample(args):
    # Unpack arguments for a single sample.
    atomic_number, position, basis_name, xc_method, grid_level = args
    return build_molecule(atomic_number, position, basis_name, xc_method, grid_level)


def main(config):
    dataset_name = config["datamodule_args"]["dataset_name"]
    dataset_args = config["datamodule_args"]["dataset_args"]
    basis_name = config.get("basis_name", "sto-3g")
    xc_method = config.get("xc_method", "lda")
    grid_level = config.get("grid_level", 3)

    log.info(f"Processing {dataset_name} dataset with {basis_name} basis")
    root = dataset_args["root"]
    dataset = MD17Dataset(name=dataset_name, root=root)

    """ Preprocess the dataset according to basis"""
    sample_mol = dataset[0]

    (
        overlap,
        kinetic,
        nuclear,
        eri,
        points,
        weights,
        orbital_tokens,
        orbital_index,
    ) = build_molecule(
        sample_mol["atomic_number"],
        sample_mol["position"],
        basis_name,
        xc_method,
        grid_level,
    )
    num_orbitals = overlap.shape[0]
    num_grid_points = points.shape[0]
    ds_size = len(dataset)

    log.info(f"Dataset Size: {ds_size}")
    log.info(f"Number of Orbitals: {num_orbitals}")
    log.info(f"Grid Size: {num_grid_points}")

    basis_dir = os.path.join(dataset.processed_dir, basis_name)
    os.makedirs(basis_dir, exist_ok=True)
    log.info(f"Saving at {basis_dir}")

    # Create memmaps for all outputs
    overlap_memmap = np.memmap(
        os.path.join(basis_dir, "overlap.memmap"),
        mode="w+",
        dtype=np.float32,
        shape=(ds_size, num_orbitals, num_orbitals),
    )
    kinetic_memmap = np.memmap(
        os.path.join(basis_dir, "kinetic.memmap"),
        mode="w+",
        dtype=np.float32,
        shape=(ds_size, num_orbitals, num_orbitals),
    )
    nuclear_memmap = np.memmap(
        os.path.join(basis_dir, "nuclear.memmap"),
        mode="w+",
        dtype=np.float32,
        shape=(ds_size, num_orbitals, num_orbitals),
    )
    eri_memmap = np.memmap(
        os.path.join(basis_dir, "eri.memmap"),
        mode="w+",
        dtype=np.float32,
        shape=(ds_size, num_orbitals, num_orbitals, num_orbitals, num_orbitals),
    )
    grid_points_memmap = np.memmap(
        os.path.join(basis_dir, "grid_points.memmap"),
        mode="w+",
        dtype=np.float32,
        shape=(ds_size, num_grid_points, 3),
    )
    grid_weights_memmap = np.memmap(
        os.path.join(basis_dir, "grid_weights.memmap"),
        mode="w+",
        dtype=np.float32,
        shape=(ds_size, num_grid_points),
    )
    orbital_tokens_memmap = np.memmap(
        os.path.join(basis_dir, "orbital_tokens.memmap"),
        mode="w+",
        dtype=int,
        shape=(ds_size, num_orbitals),
    )
    orbital_index_memmap = np.memmap(
        os.path.join(basis_dir, "orbital_index.memmap"),
        mode="w+",
        dtype=int,
        shape=(ds_size, num_orbitals),
    )

    # Prepare arguments for parallel processing.
    args_list = [
        (
            dataset[i]["atomic_number"],
            dataset[i]["position"],
            basis_name,
            xc_method,
            grid_level,
        )
        for i in range(ds_size)
    ]

    # Parallelize the heavy computation.
    with Pool() as pool:
        results = list(tqdm(pool.imap(build_molecule_sample, args_list), total=ds_size))

    # Write results into memmaps in a single loop.
    for i, (
        overlap,
        kinetic,
        nuclear,
        eri,
        points,
        weights,
        orbital_tokens,
        orbital_index,
    ) in enumerate(results):
        overlap_memmap[i] = overlap
        kinetic_memmap[i] = kinetic
        nuclear_memmap[i] = nuclear
        eri_memmap[i] = eri
        grid_points_memmap[i] = points
        grid_weights_memmap[i] = weights
        orbital_tokens_memmap[i] = orbital_tokens
        orbital_index_memmap[i] = orbital_index

    # Flush memmaps to ensure data is written to disk.
    overlap_memmap.flush()
    kinetic_memmap.flush()
    nuclear_memmap.flush()
    eri_memmap.flush()
    grid_points_memmap.flush()
    grid_weights_memmap.flush()
    orbital_tokens_memmap.flush()
    orbital_index_memmap.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True)
    args = parser.parse_args()

    config = read_yaml(args.config)
    main(config)
