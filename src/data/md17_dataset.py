import os
import os.path as osp
import tarfile
from typing import List

import numpy as np
from ase.db import connect
from loguru import logger as log
from tqdm import tqdm

from src.data.utils import download_url, load_memmap


class MD17Dataset:
    __url__ = "http://quantum-machine.org/data/schnorb_hamiltonian"
    __atom_symbols__ = ["n", "H", "He", "Li", "Be", "B", "C", "N", "O"]
    __num_atoms_map__ = {
        "water": 3,
        "ethanol": 9,
        "malondialdehyde": 9,
        "uracil": 12,
        "aspirin": 21,
    }

    def __init__(self, root="dataset/", name="water"):
        # water, ethanol, malondialdehyde, uracil
        self.name = name
        self.root = root
        self.atom_types = None

        self.download()
        self.process()
        self.load()

    @property
    def data_dir(self):
        return osp.join(self.root, self.name)

    @property
    def raw_dir(self):
        return osp.join(self.data_dir, "raw")

    @property
    def processed_dir(self):
        return osp.join(self.data_dir, "processed")

    @property
    def raw_file_names(self):
        if self.name == "ethanol":
            return [
                f"schnorb_hamiltonian_{self.name}_dft.tgz",
                f"schnorb_hamiltonian_{self.name}_dft.db",
            ]
        elif self.name == "aspirin":
            return [
                f"schnorb_hamiltonian_{self.name}_quambo.db",
                f"schnorb_hamiltonian_{self.name}_quambo.db",
            ]
        else:
            return [
                f"schnorb_hamiltonian_{self.name}.tgz",
                f"schnorb_hamiltonian_{self.name}.db",
            ]

    @property
    def processed_file_names(self):
        return ["atomic_inputs.memmap", "energy.memmap"]

    def download(self):
        tar_filepath = osp.join(self.raw_dir, self.raw_file_names[0])
        db_filepath = osp.join(self.raw_dir, self.raw_file_names[1])

        if osp.exists(db_filepath):
            return

        if not osp.exists(tar_filepath):
            if self.name == "ethanol":
                url = f"{self.__url__}/schnorb_hamiltonian_{self.name}" + "_dft.tgz"
            else:
                url = f"{self.__url__}/schnorb_hamiltonian_{self.name}" + ".tgz"

            download_url(url, self.raw_dir)
            extract_path = self.raw_dir
            tar = tarfile.open(tar_filepath, "r")
            for item in tar:
                tar.extract(item, extract_path)

    def process(self):
        processed_filepaths = [
            osp.join(self.processed_dir, filenames)
            for filenames in self.processed_file_names
        ]
        if all(map(osp.exists, processed_filepaths)):
            return

        db = connect(osp.join(self.raw_dir, self.raw_file_names[1]))
        data_list = []
        if not getattr(self, "atom_types"):
            self.atom_types = "".join(
                [self.__atom_symbols__[i] for i in next(db.select(1))["numbers"]]
            )

        for row in tqdm(db.select()):
            data_list.append(self.get_mol(row))

        os.makedirs(self.processed_dir, exist_ok=True)

        print("Saving...")
        self.save(data_list=data_list)

    def get_mol(self, row):
        # from angstrom to bohr
        # make sure the original data type is float or double
        position = np.array(row["positions"] * 1.8897261258369282, dtype=float)
        atomic_number = np.array(row["numbers"], dtype=int).reshape(-1, 1)
        energy = np.array(row.data["energy"], dtype=float)
        force = np.array(row.data["forces"], dtype=float)

        return {
            "atomic_number": atomic_number,
            "position": position,
            "energy": energy,
            "force": force,
        }

    def init_memmap(self, num_samples, num_atoms):
        log.info(f"Saving to {self.processed_dir}")

        atomic_inputs_path = osp.join(self.processed_dir, "atomic_inputs.memmap")
        energy_path = osp.join(self.processed_dir, "energy.memmap")

        return {
            "atomic_inputs": np.memmap(
                atomic_inputs_path,
                mode="w+",
                dtype=np.float32,
                shape=(num_samples, num_atoms, 4),
            ),
            "energy": np.memmap(
                energy_path, mode="w+", dtype=np.float32, shape=(num_samples, 1)
            ),
        }

    def save(self, data_list: List[dict]):
        num_samples = len(data_list)
        num_atoms = self.__num_atoms_map__[self.name]

        memmap_dict = self.init_memmap(num_samples=num_samples, num_atoms=num_atoms)

        for i, data in enumerate(data_list):
            atomic_inputs = np.concatenate(
                (data["atomic_number"], data["position"]), axis=1
            )
            memmap_dict["atomic_inputs"][i, :, :] = atomic_inputs
            memmap_dict["energy"][i, :] = data["energy"]

        for key in memmap_dict:
            memmap_dict[key].flush()

    def load(self):
        processed_filepaths = [
            osp.join(self.processed_dir, filenames)
            for filenames in self.processed_file_names
        ]
        num_atoms = self.__num_atoms_map__[self.name]

        self.atomic_inputs = load_memmap(processed_filepaths[0], np.float32).reshape(
            -1, num_atoms, 4
        )
        self.energy = load_memmap(processed_filepaths[1], dtype=np.float32).reshape(
            -1, 1
        )

    def __len__(self):
        return len(self.atomic_inputs)

    def __getitem__(self, idx):
        atomic_number = np.array(self.atomic_inputs[idx, :, 0], dtype=int)
        position = np.array(self.atomic_inputs[idx, :, 1:])
        energy = np.array(self.energy[idx, :])

        return {"atomic_number": atomic_number, "position": position, "energy": energy}
