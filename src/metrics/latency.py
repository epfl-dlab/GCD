import time

import torch
import torchmetrics


# TODO (saibo): to test

class LatencyMetric(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # Initialize state variables
        self.add_state("start_time", default=None, dist_reduce_fx="mean")
        self.add_state("total_time", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("num_measurements", default=torch.tensor(0), dist_reduce_fx="sum")

    def start_timer(self):
        self.start_time = time.time()

    def stop_timer(self):
        elapsed_time = time.time() - self.start_time
        self.total_time += elapsed_time
        self.num_measurements += 1

    def compute(self):
        # Compute average latency
        return self.total_time / self.num_measurements
