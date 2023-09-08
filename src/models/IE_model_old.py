from typing import Any, Dict

import src.utils as utils
from src.metrics import TSF1, TSPrecision, TSRecall
from src.utils import general_helpers
from src.utils.hf_gen_utils import get_first_no_empty_generation

log = utils.get_only_rank_zero_logger(__name__)
from src.models.HFModelPLOld import HFModelPL


class IEHFModelPL(HFModelPL):
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

        # ~~~ Inference ~~~
        linearization_class_id = self.hparams.get("linearization_class_id", None)
        log.info(f"Linearization class ID: {linearization_class_id}")

        self.linearization_class = utils.get_linearization_class(linearization_class_id)
        self.output_dir = None

        # ~~~ Initialize metrics ~~~
        self.ts_precision = TSPrecision()
        self.ts_recall = TSRecall()
        self.ts_f1 = TSF1()

    def test_step_end(self, outputs: Dict[Any, Any]):
        # Get the data in the format expected by the metrics
        non_empty_top_candidates = get_first_no_empty_generation(outputs["predictions"])

        outputs["final_prediction"] = non_empty_top_candidates
        print(outputs["final_prediction"])
        predictions = [
            self.linearization_class.text_to_triplet_list(
                text=text,
                verbose=self.hparams.inference.get("verbose_flag_in_convert_to_triple"),
                return_set=True,
            )
            for text in outputs["final_prediction"]
        ]
        # outputs["final_prediction"] = [texts[idx] for texts in outputs["predictions"]]

        targets = [
            self.linearization_class.text_to_triplet_list(
                text=text,
                verbose=self.hparams.inference.get("verbose_flag_in_convert_to_triple"),
                return_set=True,
            )
            for text in outputs["targets"]
        ]

        self._write_step_output(
            **outputs,
        )

        # Update the metrics
        p = self.ts_precision(predictions, targets)
        r = self.ts_recall(predictions, targets)
        f1 = self.ts_f1(predictions, targets)

        # Log the loss
        self.log("test/precision_step", p, on_step=True, on_epoch=False, prog_bar=True)
        self.log("test/recall_step", r, on_step=True, on_epoch=False, prog_bar=True)
        self.log("test/f1_step", f1, on_step=True, on_epoch=False, prog_bar=True)

        self.log("test/precision", self.ts_precision.compute())
        self.log("test/recall", self.ts_recall.compute())
        self.log("test/f1", self.ts_f1.compute())

    def test_epoch_end(self, outputs):
        """Outputs is a list of test_step outputs"""
        # Log metrics aggregated across steps and processes (in ddp)
        self.log("test/precision", self.ts_precision.compute())
        self.log("test/recall", self.ts_recall.compute())
        self.log("test/f1", self.ts_f1.compute())

        return {
            "test/acc": self.ts_precision.compute(),
            "test/recall": self.ts_precision.compute(),
            "test/f1": self.ts_precision.compute(),
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

        # convert_to_triplets in kwargs
        if "convert_to_triplets" in kwargs:
            convert_to_triplets = kwargs["convert_to_triplets"]
        else:
            convert_to_triplets = False
        if convert_to_triplets:
            decoded_triplets = [
                self.linearization_class.text_to_triplet_list(seq)
                for seq in decoded_sequences
            ]
            grouped_decoded_sequences = general_helpers.chunk_elements(
                decoded_triplets, num_return_sequences
            )
        else:
            grouped_decoded_sequences = general_helpers.chunk_elements(
                decoded_sequences, num_return_sequences
            )

        generation_outputs["grouped_decoded_outputs"] = grouped_decoded_sequences

        return generation_outputs
