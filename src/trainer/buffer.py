import random

import numpy as np


class Buffer:
    def __init__(self, max_size: int):
        self.max_size = max_size
        self._buffer = []

    def add(self, items):
        if not isinstance(items, list):
            items = [items]

        for item in items:
            self._buffer.append(np.array(item))

        if len(self._buffer) >= self.max_size:
            self._buffer = self._buffer[-self.max_size :]

    def sample(self, num_samples: int = 1):
        num_samples = int(num_samples)
        num_samples = min(num_samples, len(self._buffer))
        if num_samples == 1:
            return [self._buffer[0]]

        return random.sample(self._buffer, num_samples)

    def empty(self, idx=None):
        if idx is not None:
            self.remove_items(idx=idx)
        else:
            self._buffer.clear()

    def remove_items(self, idx):
        self._buffer = [item for i, item in enumerate(self._buffer) if i not in idx]

    def __len__(self):
        return len(self._buffer)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self._buffer[idx.start, idx.stop, idx.step]

        return self._buffer[idx]

    def __iter__(self):
        for item in self._buffer:
            yield item

    def save(self, filepath):
        np.save(filepath, np.array(self._buffer))

    def load(self, filepath):
        self._buffer = list(np.load(filepath))
