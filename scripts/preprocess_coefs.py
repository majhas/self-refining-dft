import os

import hydra
import numpy as np
import rootutils
from hydra.utils import instantiate
from omegaconf import DictConfig
from loguru import logger as log
from tqdm import tqdm

rootutils.setup_root(__file__, pythonpath=True)

from src.dft.property import run_pyscf_solver

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig):
    log.info(f"PROJECT ROOT: {os.environ['PROJECT_ROOT']}")
    log.info(f"SCRATCH DIR: {os.environ['SCRATCH_DIR']}")
    
    datamodule = instantiate(cfg.data)
    dataset = datamodule.dataset
    atomic_number = dataset[0]["atomic_number"]

    position = []
    coefficient = []
    energy = []
    
    for data in tqdm(dataset):
        pos = data["position"]
        e, c = run_pyscf_solver(
            atomic_number=atomic_number,
            position=pos,
            basis_name=cfg.basis_name,

        )

        position.append(pos[None,...])
        coefficient.append(c[None,...])
        energy.append(e.reshape(-1)[None,...])


    position = np.concatenate(position)
    coefficient = np.concatenate(coefficient)
    energy = np.concatenate(energy)

    results = {
        "atomic_number": atomic_number,
        "position": position,
        "coefficient": coefficient,
        "energy": energy,
    }

    result_path = os.path.join(
        dataset.basis_dir, 
        "preprocessed.npz" 
    )

    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    np.savez_compressed(result_path, **results)

if __name__ == "__main__":
    main()