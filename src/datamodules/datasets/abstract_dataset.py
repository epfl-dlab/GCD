import logging
from typing import Dict, List

import numpy as np

from torch.utils.data import Dataset

log = logging.getLogger(__name__)


class BaseDataset(Dataset):
    """
    A dataset implements 2 functions
        - __len__  (returns the number of samples in our dataset)
        - __getitem__ (returns a sample from the dataset at the given index idx)
    """

    def __init__(self, seed, **params):
        super().__init__()
        # we don't load the data here to allow for lazy loading
        self.data: List[Dict] = None
        self.params = params
        self.random_state = np.random.RandomState(seed)

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, idx):
        raise NotImplementedError()

    def _before_loading_data(self, **params):
        pass

    def _load_data(self, **params):
        raise NotImplementedError()

    def _after_loading_data(self, **params):
        self.compute_dataset_statistics(**params)

    def compute_dataset_statistics(self, **params):
        pass

    def load_data(self, **params):
        self._before_loading_data(**params)
        self._load_data(**params)
        self._after_loading_data(**params)

    def get_random_sample(self):
        idx = self.random_state.randint(0, len(self.data))
        return self.data[idx]

    def get_random_subset(self, k, seed):
        random_state = np.random.RandomState(seed) if seed is not None else self.random_state
        idxs = random_state.choice(len(self.data), k, replace=False)
        return [self.data[idx] for idx in idxs]

    def get_bootstrapped_data(self, seed):
        random_state = np.random.RandomState(seed) if seed is not None else self.random_state
        bootstrap_ids = random_state.choice(
            len(self.data), len(self.data), replace=True
        )

        bootstrap_data = [self.data[i] for i in bootstrap_ids]
        return bootstrap_data
