import numpy as np


class H2Dataset:
    def __init__(
        self,
        dataset_size: int = 100_000,
        min_distance: float = 0.1,
        max_distance: float = 5.0,
    ):
        self.dataset_size = dataset_size
        self.min_distance = min_distance
        self.max_distance = max_distance
        self._distance = np.linspace(min_distance, max_distance, dataset_size)

    def __len__(self):
        return self.dataset_size

    def get_position(self, idx):
        distance = self._distance[idx].reshape(
            -1,
        )

        position = np.zeros(shape=(len(distance), 2, 3), dtype=np.float32)
        position[:, 1, 0] = distance

        return position

    def __getitem__(self, idx):
        position = self.get_position(idx)
        atomic_number = np.ones((len(position), 2), dtype=np.int32)

        position = position.squeeze(0)
        atomic_number = atomic_number.squeeze(0)

        return {
            "atomic_number": atomic_number,
            "position": position,
        }
