from typing import Any, Dict
from typing import Any, Dict

from PYEVALB import parser
from PYEVALB import scorer
from torchmetrics import MeanMetric

import src.utils as utils
from src.utils import general_helpers
from src.utils.evaluation_helpers import (
    extract_string_in_bracket,
    rm_space_ptb,
)
from src.utils.hf_gen_utils import get_first_no_empty_generation

log = utils.get_only_rank_zero_logger(__name__)
from src.models.HFModelPLOld import HFModelPL


class CPHFModelPL(HFModelPL):
    def __init__(
        self,
        hparams_overrides: Dict = None,
        hf_config_overrides: Dict = None,
        from_pretrained=False,
        trie_constraint_module: Dict = None,
        gf_constraint_module: Dict = None,
        **kwargs,
    ):
        super().__init__(
            hparams_overrides=hparams_overrides,
            hf_config_overrides=hf_config_overrides,
            from_pretrained=from_pretrained,
            trie_constraint_module=trie_constraint_module,
            gf_constraint_module=gf_constraint_module,
            **kwargs,
        )

        self.output_dir = None
        self.work_dir = None
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

    def test_step_end(self, outputs: Dict[Any, Any]):

        non_empty_top_candidates = get_first_no_empty_generation(outputs["predictions"])
        outputs["final_prediction"] = non_empty_top_candidates

        targets = [text for text in outputs["targets"]]

        # Update the metrics
        for i in range(len(outputs["final_prediction"])):
            final_prediction = outputs["final_prediction"][i]
            target = targets[i]
            valid = 0
            # final_prediction = final_prediction + " ]" if final_prediction[-1] != "]" else final_prediction
            try:
                processed_output = rm_space_ptb(
                    extract_string_in_bracket(final_prediction)
                )
                processed_target = rm_space_ptb(target)
                log.info(f"outputs: {processed_output}")
                log.info(f"targets: {processed_target}")
                gold_tree = parser.create_from_bracket_string(processed_target)
                test_tree = parser.create_from_bracket_string(processed_output)
                result = self.evalb_scorer.score_trees(gold_tree, test_tree)

                if result.state == 0:  # valid, 1 for invalid
                    valid = 1
                for stat in result.STATISTICS_TABLE:
                    log.info(stat + " = " + str(getattr(result, stat)))
                    # get self attribute based on stat name
                    metric = getattr(self, f"evalb_{stat}")
                    metric.update(getattr(result, stat))

            except Exception as e:
                log.info(f"outputs: failed to parse {final_prediction} with error {e}")
            finally:
                self.evalb_valid_percentage.update(valid)
                for stat in self.STATISTICS:
                    metric = getattr(self, f"evalb_{stat}")
                    acc_result = metric.compute()
                    self.log(
                        f"test/{stat}_step",
                        acc_result,
                        on_step=True,
                        on_epoch=False,
                        prog_bar=True,
                    )
                    log.info(f"acc_{stat}= {acc_result}")

        self._write_step_output(
            **outputs,
        )

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

    def sample(
        self,
        input_data,
        input_is_processed_batch=False,
        seed=None,
        skip_special_tokens=True,
        return_generation_outputs=False,
        return_generation_inputs=False,
        convert_to_triplets=False,
        prefix_allowed_tokens_fn=None,
        return_only_generated_text=True,
        **kwargs,
    ):
        results = super().sample(
            input_data=input_data,
            input_is_processed_batch=input_is_processed_batch,
            seed=seed,
            skip_special_tokens=skip_special_tokens,
            return_generation_outputs=return_generation_outputs,
            return_generation_inputs=return_generation_inputs,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            return_only_generated_text=return_only_generated_text,
            **kwargs,
        )

        results = self.on_sample_end(results, convert_to_triplets=convert_to_triplets)

        return results

    def on_sample_end(self, generation_outputs, **kwargs):

        decoded_sequences = generation_outputs["decoded_sequences"]
        num_return_sequences = self.hparams.inference["hf_generation_params"].get(
            "num_return_sequences", 1
        )

        grouped_decoded_sequences = general_helpers.chunk_elements(
            decoded_sequences, num_return_sequences
        )

        generation_outputs["grouped_decoded_outputs"] = grouped_decoded_sequences

        return generation_outputs
