import time

from loguru import logger as log
from pyscf.dft.rks import RKS
from tqdm import tqdm

from src.commons import build_pyscf_mol
from src.data.dataset import SupervisedGraphDataset


def main():
    basis_name = "6-31g"
    mol_name = "ethanol"
    dataset = SupervisedGraphDataset(
        mol_name,
        dataset_args={"root": "/network/scratch/m/majdi.hassan/data/md17"},
        basis_name=basis_name,
        xc_method="pbe",
    )
    data = [dataset[i] for i in range(100)]

    start = time.time()
    for d in tqdm(data):
        mol = build_pyscf_mol(d.atomic_number, d.position, basis_name=basis_name)
        mf = RKS(mol, xc="pbe")
        mf.max_cycle = 15000

        mf.run()

    end = time.time()
    runtime = end - start
    scaled_runtime = len(dataset) / len(data) * runtime
    log.info(f"Runtime for DFT on {mol_name} for {len(data)} conformers: {runtime} s")
    log.info(
        f"Scaled Runtime for {mol_name}: {scaled_runtime} s / {scaled_runtime / 3600} hr"
    )


if __name__ == "__main__":
    main()
