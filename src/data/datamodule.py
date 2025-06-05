import numpy as np
from loguru import logger as log


class DataIterator:
    def __init__(
        self, dataset, batch_size: int = 32, indices=None, size=None, seed: int = 42
    ):
        self.dataset = dataset
        self.indices = indices
        self.batch_size = batch_size
        self.size = size
        self.seed = seed

        np.random.seed(self.seed)
        if size is not None:
            self.indices = np.random.choice(self.indices, size, replace=False)

        self.reset()

    def generator_func(self, indices):
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i : i + self.batch_size]
            yield [self.dataset[batch_idx] for batch_idx in batch_indices]

    def reset(self):
        indices = self.indices
        if indices is None:
            indices = np.arange(len(self.dataset))

        if self.size is None or self.size == -1:
            self.size = len(indices)

        permutation = np.random.permutation(len(self.indices))
        permutated_indices = self.indices[permutation]
        self._generator = self.generator_func(permutated_indices)

    def __iter__(self):
        self.reset()
        return self

    def __next__(self):
        return next(self._generator)

    def __len__(self):
        return int(len(self.indices) / self.batch_size)


class DataModule:
    def __init__(
        self,
        dataset,
        batch_size: int = 4,
        seed: int = 42,
        train_size: int = None,
        valid_size: int = None,
        test_size: int = None,
        train_indices: list = None,
        valid_indices: list = None,
        test_indices: list = None,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.train_size = train_size
        self.valid_size = valid_size
        self.test_size = test_size

        if isinstance(train_indices, list) or isinstance(train_indices, np.ndarray):
            self.train_indices = train_indices
            self.valid_indices = valid_indices
            self.test_indices = test_indices
        elif isinstance(train_indices, str):
            self.train_indices = np.load(train_indices)
            self.valid_indices = np.load(valid_indices)
            self.test_indices = np.load(test_indices)
        else:
            self.train_indices = None
            self.valid_indices = None
            self.test_indices = None

        # Precompute indices only once
        self.setup()

    def setup(self):
        np.random.seed(self.seed)
        self.indices = np.random.permutation(len(self.dataset))

        if self.train_indices is None:
            dataset_size = len(self.dataset)
            self.train_size = (
                self.train_size
                if self.train_size is not None
                else int(0.8 * dataset_size)
            )
            self.valid_size = (
                self.valid_size
                if self.valid_size is not None
                else int(0.1 * dataset_size)
            )
            self.test_size = (
                self.test_size
                if self.test_size is not None
                else int(0.1 * dataset_size)
            )
            self.dataset_size = self.train_size + self.valid_size + self.test_size

            test_idx_start = self.train_size + self.valid_size

            self.train_indices = self.indices[: self.train_size]
            self.valid_indices = self.indices[self.train_size : test_idx_start]
            self.test_indices = self.indices[
                test_idx_start : test_idx_start + self.test_size
            ]
        else:
            self.train_size = len(self.train_indices)
            self.valid_size = len(self.valid_indices)
            self.test_size = len(self.test_indices)

        log.info(f"Train size: {len(self.train_indices)}")
        log.info(f"Valid size: {len(self.valid_indices)}")
        log.info(f"Test size: {len(self.test_indices)}")

    def train_dataloader(self, size=None, batch_size: int = None):
        return DataIterator(
            dataset=self.dataset,
            batch_size=self.batch_size if batch_size is None else batch_size,
            indices=self.train_indices,
            size=size,
        )

    def valid_dataloader(self, size=None, batch_size: int = None):
        return DataIterator(
            dataset=self.dataset,
            batch_size=self.batch_size if batch_size is None else batch_size,
            indices=self.valid_indices,
            size=size,
        )

    def test_dataloader(self, size=None, batch_size: int = None):
        return DataIterator(
            dataset=self.dataset,
            batch_size=self.batch_size if batch_size is None else batch_size,
            indices=self.test_indices,
            size=size,
        )
