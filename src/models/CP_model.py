from typing import Any, Dict, Union, List, Optional
from typing import Any, Dict

from PYEVALB import parser
from PYEVALB import scorer
from omegaconf import OmegaConf
from torchmetrics import MeanMetric

import src.utils as utils
from src.utils import general_helpers
from src.utils.evaluation_helpers import (
    extract_string_in_bracket,
    rm_space_ptb,
)
from src.utils.hf_gen_utils import get_first_no_empty_generation

log = utils.get_only_rank_zero_logger(__name__)
from src.models.HFModelPL import HFModelPL


class CPHFModelPL(HFModelPL):

    def __init__(self, params: Union[Dict, OmegaConf]):
        super().__init__(params)

        # self.work_dir = None
        self.evalb_scorer = scorer.Scorer()

        # ~~~ Initialize metrics ~~~
        self.evalb_valid_percentage = MeanMetric()
        self.evalb_length = MeanMetric()
        self.evalb_state = MeanMetric()
        self.evalb_recall = MeanMetric()
        self.evalb_prec = MeanMetric()
        self.evalb_tag_accracy = MeanMetric()

        self.evalb_matched_brackets = MeanMetric()
        self.evalb_gold_brackets = MeanMetric()
        self.evalb_test_brackets = MeanMetric()
        self.evalb_cross_brackets = MeanMetric()
        self.evalb_words = MeanMetric()
        self.evalb_correct_tags = MeanMetric()
        self.evalb_ID = (
            MeanMetric()
        )  # for compatibility with the other metrics, c.f. test_step_end
        self.STATISTICS = [
            "valid_percentage",
            "length",
            "state",
            "recall",
            "prec",
            "tag_accracy",
            "matched_brackets",
            "gold_brackets",
            "test_brackets",
            "cross_brackets",
            "words",
            "correct_tags",
            "ID",
        ]

    @staticmethod
    def parse(line:str) -> Optional[Any]:
        try:
            structured_prediction:Optional[Any] = parser.create_from_bracket_string(line)
        except AttributeError or parser.ParsingError:
            structured_prediction = None
        return structured_prediction

    def score(self, gold_tree, test_tree):
        if gold_tree is None or test_tree is None:
            return None
        return self.evalb_scorer.score_trees(gold_tree, test_tree)

    def _get_structured_prediction(self, outputs: Dict[str, List[Any]]) -> List[Optional[str]]:
        """
        This function is used to get the structured prediction in a string format from the outputs of the model.
        N.B. the output is supposed to be a list of strings, each string is a structured prediction that can be further processed.
        For example, in the case of a constituency parsing task, the output is only cleaned from the spaces and the brackets.
        The parsing is done in the test_step_end function.
        """
        first_no_empty_generations: List[str] = get_first_no_empty_generation(outputs["unflattened_predictions"])

        structured_prediction: List[Optional[str]] = [rm_space_ptb(extract_string_in_bracket(text)) for text in first_no_empty_generations]
        return structured_prediction

    def test_step_end(self, outputs: Dict[Any, Any]):
        outputs = super().test_step_end(outputs)

        structured_targets: List[Optional[str]] = [self.parse(rm_space_ptb(target)) for target in outputs["targets"]]
        structured_predictions: List[Optional[str]] = [self.parse(pred) for pred in outputs["structured_predictions"]]

        # Update the metrics
        for i in range(len(structured_predictions)):
            valid = 0
            pred_tree = structured_predictions[i]
            gold_tree = structured_targets[i]
            log.debug(f"outputs: {pred_tree}")
            log.debug(f"targets: {gold_tree}")
            result: Optional = self.score(gold_tree, pred_tree)

            if result is None or result.state == 1:
                # no update on metrics
                log.debug(f"outputs: failed to parse {pred_tree}")
            else:
                log.debug(f"outputs: successfully parsed {pred_tree}")
                valid = 1
                for stat in result.STATISTICS_TABLE:
                    log.info(stat + " = " + str(getattr(result, stat)))
                    # get self attribute based on stat name
                    metric = getattr(self, f"evalb_{stat}")
                    metric.update(getattr(result, stat))
                    acc_result = metric.compute()
                    self.log(
                        f"test/{stat}_step",
                        acc_result,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=True,
                    )
                    log.info(f"acc_{stat}= {acc_result}")

            # valid percentage is updated in both cases
            # while the other metrics are updated only when the parsing is successful
            self.evalb_valid_percentage.update(valid)

    def test_epoch_end(self, outputs):
        """Outputs is a list of test_step outputs"""

        for statistics in self.STATISTICS:
            metric = getattr(self, f"evalb_{statistics}")
            acc_result = metric.compute()
            self.log(
                f"test/{statistics}",
                acc_result,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
            print(f"acc_{statistics}", acc_result)

        results = {
            f"test/evalb_{statistics}": getattr(self, f"evalb_{statistics}").compute()
            for statistics in self.STATISTICS
        }
        return results
