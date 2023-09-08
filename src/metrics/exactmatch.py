from typing import List, Union

import torch
from torchmetrics import Metric


class ExactMatchAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Union[List[any],torch.Tensor], target: Union[List[any],torch.Tensor]):
        assert len(preds) == len(target)

        self.correct += sum([pred == target for pred, target in zip(preds, target)])
        self.total += len(preds)

    def compute(self):
        return self.correct.float() / self.total
