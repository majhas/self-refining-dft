import os.path as osp
from functools import partial

import numpy as np
from loguru import logger as log

from src.commons.graph import build_graph, build_graph_no_hamil
from src.commons.types import Data, Mesh, OneElectron, TwoElectron
from src.data.h2_dataset import H2Dataset
from src.data.md17_dataset import MD17Dataset
from src.data.utils import load_memmap
from src.dft.hamiltonian import Hamiltonian
from src.dft.molecule import build_molecule, build_pyscf_mol

DATASET_MAPPING = {
    "h2": H2Dataset,
    "water": partial(MD17Dataset, name="water"),
    "ethanol": partial(MD17Dataset, name="ethanol"),
    "malondialdehyde": partial(MD17Dataset, name="malondialdehyde"),
    "uracil": partial(MD17Dataset, name="uracil"),
    "aspirin": partial(MD17Dataset, name="aspirin"),
}


class BaseDataset:
    __datasets__ = ["h2", "water", "ethanol", "malondialdehyde", "uracil", "aspirin"]

    def __init__(
        self,
        dataset_name: str,
        dataset_args: dict = {},
        basis_name: str = "sto-3g",
        xc_method: str = "lda",
        grid_level: int = 3,
        load_preprocessed: bool = False,
    ):
        assert (
            dataset_name in self.__datasets__
        ), f"{dataset_name} not available in {self.__datasets__}"

        log.info(
            f"Loading dataset {dataset_name} with basis {basis_name} and XC {xc_method}"
        )

        self.dataset_name = dataset_name
        self.dataset = DATASET_MAPPING[dataset_name](**dataset_args)
        self.basis_name = basis_name
        self.xc_method = xc_method
        self.grid_level = grid_level
        self.orbital_tokens = {}
        self.load_preprocessed = load_preprocessed

        if self.load_preprocessed:
            self.load_preprocessed_dft()

    @property
    def data_dir(self):
        return self.dataset.data_dir

    @property
    def raw_dir(self):
        return self.dataset.raw_dir

    @property
    def processed_dir(self):
        return self.dataset.processed_dir

    @property
    def basis_dir(self):
        return osp.join(self.dataset.processed_dir, self.basis_name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        atomic_number = data["atomic_number"]
        position = data["position"]
        energy = data.get("energy", None)

        if self.load_preprocessed:
            return self.load_preprocessed_mol(idx, atomic_number, position, energy)

        return self.process_mol(atomic_number, position, energy)

    def load_preprocessed_dft(self):
        data = self.dataset[0]

        # build mol just to get num_orbitals and grid points to load preprocessed data correctly
        mol = build_molecule(
            data["atomic_number"],
            data["position"],
            basis_name=self.basis_name,
            grid_level=self.grid_level,
        )
        num_orbitals = mol.hamiltonian.H_core.shape[0]
        num_grid_points = mol.hamiltonian.mesh.points.shape[0]

        self.overlap = load_memmap(
            osp.join(self.basis_dir, "overlap.memmap"), dtype=np.float32
        ).reshape(len(self.dataset), num_orbitals, num_orbitals)
        self.kinetic = load_memmap(
            osp.join(self.basis_dir, "kinetic.memmap"), dtype=np.float32
        ).reshape(len(self.dataset), num_orbitals, num_orbitals)
        self.nuclear = load_memmap(
            osp.join(self.basis_dir, "nuclear.memmap"), dtype=np.float32
        ).reshape(len(self.dataset), num_orbitals, num_orbitals)

        self.eri = load_memmap(
            osp.join(self.basis_dir, "eri.memmap"), dtype=np.float32
        ).reshape(
            len(self.dataset), num_orbitals, num_orbitals, num_orbitals, num_orbitals
        )

        self.grid_points = load_memmap(
            osp.join(self.basis_dir, "grid_points.memmap"), dtype=np.float32
        ).reshape(len(self.dataset), num_grid_points, 3)
        self.grid_weights = load_memmap(
            osp.join(self.basis_dir, "grid_weights.memmap"), dtype=np.float32
        ).reshape(
            len(self.dataset),
            num_grid_points,
        )

        self.orbital_tokens = load_memmap(
            osp.join(self.basis_dir, "orbital_tokens.memmap"), dtype=int
        ).reshape(len(self.dataset), num_orbitals)
        self.orbital_index = load_memmap(
            osp.join(self.basis_dir, "orbital_index.memmap"), dtype=int
        ).reshape(len(self.dataset), num_orbitals)

    def process_mol(self, atomic_number, position, energy=None):
        return build_molecule(
            atomic_number,
            position,
            energy=energy,
            basis_name=self.basis_name,
            xc_method=self.xc_method,
            grid_level=self.grid_level,
        )

    def load_preprocessed_mol(self, idx, atomic_number, position, energy=None):
        mol = build_pyscf_mol(atomic_number, position, basis_name=self.basis_name)

        # Organize data into named tuples
        one_electron = OneElectron(
            overlap=np.array(self.overlap[idx], dtype=np.float64),
            kinetic=np.array(self.kinetic[idx], dtype=np.float64),
            nuclear=np.array(self.nuclear[idx], dtype=np.float64),
        )

        two_electron = TwoElectron(eri=np.array(self.eri[idx]))
        mesh = Mesh(
            points=np.array(self.grid_points[idx], dtype=np.float64),
            weights=np.array(self.grid_weights[idx], dtype=np.float64),
        )

        occ = np.full(mol.nao, 2.0)
        mask = occ.cumsum() > mol.nelectron
        occ = np.where(mask, 0.0, occ)

        hamiltonian = Hamiltonian(
            kinetic=one_electron.kinetic,
            nuclear=one_electron.nuclear,
            overlap=one_electron.overlap,
            eri=two_electron.eri,
            occupancy=occ,
            mesh=mesh,
            xc_method=self.xc_method,
            gridAO=mol.eval_gto("GTOval_cart_deriv1", mesh.points, 4),
        )
        # Map orbital to atom center and tokenize orbitals
        orbital_index = np.array(self.orbital_index[idx])
        orbital_tokens = np.array(self.orbital_tokens[idx])
        return Data(
            atomic_number=atomic_number,
            position=position.astype(np.float64),
            energy=energy,
            hamiltonian=hamiltonian,
            orbital_tokens=orbital_tokens,
            orbital_index=orbital_index,
        )


class GraphDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_args: dict,
        basis_name: str = "sto-3g",
        xc_method: str = "lda",
        grid_level: int = 3,
        load_preprocessed: bool = False,
    ):
        super(GraphDataset, self).__init__(
            dataset_name,
            dataset_args,
            basis_name,
            xc_method,
            grid_level,
            load_preprocessed,
        )

    def __getitem__(self, idx):
        data = self.dataset[idx]
        atomic_number = data["atomic_number"]
        position = data["position"]
        energy = data.get("energy", None)

        # need position to be [batch_size, n_nodes, 3]
        # expect the batch to have all the same molecule
        center = position.mean(axis=0, keepdims=True)
        position -= center

        if self.load_preprocessed:
            data = self.load_preprocessed_mol(
                idx, atomic_number, position, energy=energy
            )
        else:
            data = self.process_mol(atomic_number, position, energy=energy)

        return build_graph(
            atomic_number=data.atomic_number,
            position=data.position,
            hamiltonian=data.hamiltonian,
            orbital_index=data.orbital_index,
            orbital_tokens=data.orbital_tokens,
            energy=data.energy,
        )


class SupervisedGraphDataset(BaseDataset):
    def __init__(
        self,
        dataset_name: str,
        dataset_args: dict,
        basis_name: str = "6-31g*",
        xc_method: str = "pbe",
        grid_level: int = 3,
        load_preprocessed: bool = False,
    ):
        super(SupervisedGraphDataset, self).__init__(
            dataset_name,
            dataset_args,
            basis_name,
            xc_method,
            grid_level,
            load_preprocessed,
        )

    def __getitem__(self, idx):
        data = self.dataset[idx]
        atomic_number = data["atomic_number"]
        position = data["position"]
        energy = data.get("energy", None)

        # need position to be [batch_size, n_nodes, 3]
        # expect the batch to have all the same molecule
        center = position.mean(axis=0, keepdims=True)
        position -= center

        return build_graph_no_hamil(
            atomic_number=atomic_number,
            position=position,
            energy=energy,
        )
