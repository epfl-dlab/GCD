from typing import List, Any, Dict

from torchmetrics import MeanMetric

import src.utils as utils
from src.utils import general_helpers
from src.utils.evaluation_helpers import (
    extract_string_in_bracket,
)
from src.utils.hf_gen_utils import get_first_no_empty_generation

log = utils.get_only_rank_zero_logger(__name__)
from src.models.HFModelPLOld import HFModelPL


class ELHFModelPL(HFModelPL):
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

        # ~~~ Initialize metrics ~~~
        self.accuracy = MeanMetric()

    def test_step_end(self, outputs: Dict[Any, Any]):
        non_empty_top_candidates = get_first_no_empty_generation(outputs["predictions"])
        final_prediction = [
            extract_string_in_bracket(texts) for texts in non_empty_top_candidates
        ]

        outputs["final_prediction"] = final_prediction
        self._write_step_output(
            **outputs,
        )

        targets = [text for text in outputs["targets"]]

        # Update the metrics
        for i in range(len(final_prediction)):
            correct = final_prediction[i] == targets[i]
            self.accuracy.update(correct)
        acc = self.accuracy.compute()
        # Log the loss
        print(f"acc: {acc}, predictions: {final_prediction}, targets: {targets}")
        print(f"outputs: {outputs}")
        self.log("test/accuracy_step", acc, on_step=True, on_epoch=False, prog_bar=True)

    def test_epoch_end(self, outputs):
        """Outputs is a list of test_step outputs"""
        # Log metrics aggregated across steps and processes (in ddp)
        acc = self.accuracy.compute()
        self.log("test/accuracy", acc)

        return {
            "test/accuracy": acc,
        }

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
