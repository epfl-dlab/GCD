from typing import List, Any, Dict, Union

import torch
from omegaconf import OmegaConf
from torchmetrics import MeanMetric, BootStrapper

import src.utils as utils
from src.metrics.exactmatch import ExactMatchAccuracy
from src.utils import general_helpers
from src.utils.evaluation_helpers import (
    extract_string_in_bracket,
)
from src.utils.hf_gen_utils import get_first_no_empty_generation

log = utils.get_only_rank_zero_logger(__name__)
from src.models.HFModelPL import HFModelPL


class EDHFModelPL(HFModelPL):
    def __init__(self, params: Union[Dict, OmegaConf]):
        super().__init__(params)

        # ~~~ Initialize metrics ~~~
        self.accuracy = ExactMatchAccuracy()
        self.bootstrap = BootStrapper(self.accuracy, num_bootstraps=1000, quantile=torch.tensor([0.025, 0.975]))
        self.step_vals = []

    def test_step_end(self, outputs: Dict[Any, Any]):
        outputs = super().test_step_end(outputs)

        structured_predictions = outputs["structured_predictions"]
        structured_targets = outputs["targets"]

        _correct = [pred == target for pred, target in zip(structured_predictions, structured_targets)]

        pred = torch.tensor(_correct)
        target = torch.ones_like(pred)

        step_val = self.accuracy(preds=pred, target=target)
        self.step_vals.append(step_val)

        self.log("test/accuracy_step", step_val, on_step=True, on_epoch=False, prog_bar=True)
        log.info("test/accuracy_step: %s", step_val.item())
        self.log("test/accuracy", self.accuracy.compute())
        log.info("test/accuracy: %s", self.accuracy.compute().item())
        # Log the output
        log.debug(f"predictions: {structured_predictions}, targets: {structured_targets}")


    def _get_structured_prediction(self, outputs: Dict[str, List[Any]]):
        first_no_empty_generations: List[str] = get_first_no_empty_generation(outputs["unflattened_predictions"])

        structured_prediction = [
            extract_string_in_bracket(texts) for texts in first_no_empty_generations
        ]
        return structured_prediction

    def test_epoch_end(self, outputs):
        """Outputs is a list of test_step outputs"""
        # Log metrics aggregated across steps and processes (in ddp)
        acc = self.accuracy.compute()
        all_step_vals = torch.stack(self.step_vals)
        targets = torch.ones_like(all_step_vals)
        self.bootstrap.update(all_step_vals, targets)
        output = self.bootstrap.compute()
        self.log("test/accuracy", acc)
        self.log("test/accuracy_ci_0.025", output["quantile"][0])
        self.log("test/accuracy_ci_0.975", output["quantile"][1])

        return {
            "test/accuracy": acc,
            "test/accuracy_95ci": output["quantile"]
        }
