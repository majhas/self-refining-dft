import numpy as np

class NumpyDataset:
    def __init__(
        self,
        path
    ):
        data = np.load(path, allow_pickle=True)
        self.atomic_number = data["atomic_number"]
        self.position = data["position"]
        self.coefficient = data["coefficient"]
        self.energy = data["energy"]
        
    def __getitem__(self, idx):
        return {
            "atomic_number": self.atomic_number,
            "position": self.position[idx],
            "coefficient": self.coefficient[idx],
            "energy": self.energy[idx]
        }