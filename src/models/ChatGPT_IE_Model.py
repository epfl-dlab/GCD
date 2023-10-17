from typing import List, Any, Dict, Union

from omegaconf import OmegaConf
from torchmetrics import MeanMetric

import src.utils as utils
from src.metrics import TSPrecision, TSRecall, TSF1
from src.utils import general_helpers
from src.utils.evaluation_helpers import (
    extract_string_in_bracket,
)
from src.utils.hf_gen_utils import get_first_no_empty_generation

log = utils.get_only_rank_zero_logger(__name__)
from src.models.ChatGPTModel import ChatGPTModel


class ChatGPT_IE_Model(ChatGPTModel):
    def __init__(self, params: Union[Dict, OmegaConf]):
        super().__init__(params)

        # ~~~ Inference ~~~
        linearization_class_id = self.hparams.get("linearization_class_id", None)
        assert linearization_class_id is not None, "linearization_class_id must be specified, but got None"
        log.info(f"Linearization class ID: {linearization_class_id}")

        self.linearization_class = utils.get_linearization_class(linearization_class_id)

        # ~~~ Initialize metrics ~~~
        self.ts_precision = TSPrecision()
        self.ts_recall = TSRecall()
        self.ts_f1 = TSF1()

    def test_step_end(self, outputs: Dict[Any, Any]):
        # import pdb;
        # pdb.set_trace()
        outputs = super().test_step_end(outputs)
        # pdb.set_trace()

        structured_predictions = [
            self.linearization_class.text_to_triplet_list(
                text=text,
                verbose=self.hparams.inference.get("verbose_flag_in_convert_to_triple"),
                return_set=True,
            )
            for text in outputs["structured_predictions"]
        ]
        structured_targets = [
            self.linearization_class.text_to_triplet_list(
                text=text,
                verbose=self.hparams.inference.get("verbose_flag_in_convert_to_triple"),
                return_set=True,
            )
            for text in outputs["targets"]
        ]

        log.debug(f"Structured predictions: {structured_predictions}")
        log.debug(f"Structured targets: {structured_targets}")
        # Update the metrics
        p = self.ts_precision(structured_predictions, structured_targets)
        r = self.ts_recall(structured_predictions, structured_targets)
        f1 = self.ts_f1(structured_predictions, structured_targets)

        # Log the metrics to wandb
        self.log(f"test_{self.global_rank}/precision_step", p, on_step=True, on_epoch=False, prog_bar=True)
        self.log(f"test_{self.global_rank}/recall_step", r, on_step=True, on_epoch=False, prog_bar=True)
        self.log(f"test_{self.global_rank}/f1_step", f1, on_step=True, on_epoch=False, prog_bar=True)

        # log the metrics to console
        log.info(f"test_{self.global_rank}/precision_step: {p}")
        log.info(f"test_{self.global_rank}/recall_step: {r}")
        log.info(f"test_{self.global_rank}/f1_step: {f1}")

        # the below logging may not be necessary because every compute() will do sync across processes
        # this is expensive. https://torchmetrics.readthedocs.io/en/stable/references/metric.html#torchmetrics-metric
        running_p = self.ts_precision.compute()
        running_r = self.ts_recall.compute()
        running_f1 = self.ts_f1.compute()
        self.log(f"test/precision", running_p, rank_zero_only=True)
        self.log("test/recall", running_r, rank_zero_only=True)
        self.log("test/f1", running_f1, rank_zero_only=True)

        log.info(f"test/precision: {running_p}")
        log.info(f"test/recall: {running_r}")
        log.info(f"test/f1: {running_f1}")

        return outputs

    def _get_structured_prediction(self, outputs: Dict[str, List[Any]]):
        first_no_empty_generations: List[str] = get_first_no_empty_generation(outputs["unflattened_predictions"])

        structured_predictions = first_no_empty_generations
        return structured_predictions

    def test_epoch_end(self, outputs):
        """Outputs is a list of test_step outputs"""
        # Log metrics aggregated across steps and processes (in ddp)
        self.log("test/precision", self.ts_precision.compute(), rank_zero_only=True)
        self.log("test/recall", self.ts_recall.compute(), rank_zero_only=True)
        self.log("test/f1", self.ts_f1.compute(), rank_zero_only=True)

        return {
            "test/acc": self.ts_precision.compute(),
            "test/recall": self.ts_precision.compute(),
            "test/f1": self.ts_precision.compute(),
        }
