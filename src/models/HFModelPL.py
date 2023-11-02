import os
from pathlib import Path

from typing import Dict, Union, List, Any

import hydra
import pytorch_lightning as pl
import torch
import transformers
from omegaconf import OmegaConf, open_dict
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)


import src.utils as utils
from src.utils.evaluation_helpers import upload_outputs_to_wandb
from src.utils.hf_gen_utils import unflatten_generations
from src.utils.dict_utils import dict_of_lists_to_list_of_dicts
from src.utils.file_io import auto_write_text
from src.utils.generic_text_collator import GenericTextCollator

log = utils.get_only_rank_zero_logger(__name__, stdout=True)
import logging
log.setLevel(logging.INFO)


class HFModelPL(pl.LightningModule):
    def __init__(self, params: Union[Dict, OmegaConf]):
        super().__init__()
        # equivalent to self.hparams = config
        self.save_hyperparameters(params)
        self._init_model_and_tokenizer()
        self._init_collator()
        self._init_constraint_module()

    def _init_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model_name_or_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        hf_model_config = AutoConfig.from_pretrained(self.hparams.pretrained_model_name_or_path)
        hf_model_config.update(self.hparams.hf_extra_model_config)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.hparams.pretrained_model_name_or_path,
            config=hf_model_config,
            torch_dtype=torch.half if self.hparams.half_precision else torch.float,
        )
        if self.hparams.half_precision and not self.hparams.use_accelerate:
            self.model = self.model.half()

    def _init_collator(self):
        self.collator = GenericTextCollator(
            tokenizer=self.tokenizer, **self.hparams.collator_parameters
        )

    def _init_constraint_module(self):
        if self.hparams.gf_constraint_module is not None and self.hparams.trie_constraint_module is not None:
            raise ValueError(
                "You cannot specify both a GF constraint module and a Trie constraint module"
            )

        # ~~~ Constraint generation ~~~
        if self.hparams.gf_constraint_module is not None:
            log.info("Running inference with GF-CONSTRAINED decoding")
            self.constraint_module = hydra.utils.instantiate(self.hparams.gf_constraint_module)
        elif self.hparams.trie_constraint_module is not None:
            log.info("Running inference with Trie-CONSTRAINED decoding")
            self.constraint_module = hydra.utils.instantiate(
                self.hparams.trie_constraint_module, model=self
            )
        else:
            log.info("Running UNCONSTRAINED inference.")
            self.constraint_module = None

    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None, **kwargs):
        output = self.model(
            input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            **kwargs,
        )
        return output

    def process_batch(self, batch):
        batch["decoder_input_ids"] = self.model._shift_right(batch["tgt_input_ids"])

        return batch

    def _get_predictions_for_batch(self, batch) -> Dict[str, Any]:
        # ~~~ Prediction related ~~~
        # Generate predictions
        if self.constraint_module is None:
            sample_prefix_allowed_tokens_fn = None
        else:
            sample_prefix_allowed_tokens_fn = (
                self.constraint_module.get_prefix_allowed_tokens_fn(batch=batch)
            )
            assert sample_prefix_allowed_tokens_fn is not None

        hf_generation_params = self.hparams.inference["hf_generation_params"].copy()
        sampling_params = {
            "input_is_processed_batch": True,  # TODO
            "return_generation_inputs": True,
            "return_generation_outputs": True,
            "output_scores": True,
            "hf_generation_params": hf_generation_params,
        }

        sample_output = self.sample(
            batch,
            prefix_allowed_tokens_fn=sample_prefix_allowed_tokens_fn,
            **sampling_params,
        )

        return sample_output

    @torch.no_grad()
    def sample(
            self,
            input_data,
            input_is_processed_batch=False,
            seed=None,
            skip_special_tokens=True,
            return_generation_outputs=False,
            return_generation_inputs=False,
            remove_prefix_from_generation=True,
            prefix_allowed_tokens_fn=None,
            inference=None,
            **kwargs,
    ):
        """Input data is a list of strings or a processed batch (contains src_input_ids,
        and src_attention_mask as expected in training)"""
        # if the model is not in evaluation mode, set it and remember to reset it
        is_in_training_mode = self.training
        if is_in_training_mode:
            self.eval()

        inference = inference or {}
        runtime_hf_generation_kwargs = inference.get("hf_generation_params", {})
        extra_hf_generation_params = self.hparams.inference.get("hf_generation_params", {}).copy()

        # By default, new keys are not allowed in hydras DictConfig
        # so we need to open it to update it
        # c.f. https://stackoverflow.com/questions/66295334/create-a-new-key-in-hydra-dictconfig-from-python-file
        with open_dict(extra_hf_generation_params):
            extra_hf_generation_params.update(runtime_hf_generation_kwargs)
            extra_hf_generation_params["return_dict_in_generate"] = True

        if seed is None:
            seed = inference.get("seed", None)
        if seed:
            transformers.trainer_utils.set_seed(seed)

        # Get input_ids and attention masks
        if not input_is_processed_batch:
            input_data = self.collator.collate_input(input_data)

        input_ids = input_data["src_input_ids"].to(self.device)
        attention_mask = input_data["src_attention_mask"].to(self.device)

        if prefix_allowed_tokens_fn is None and self.constraint_module is not None:
            raise RuntimeError("this should not happen")

        generate_kwargs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
            **extra_hf_generation_params,
        }
        generation_outputs = self.model.generate(**generate_kwargs)

        if remove_prefix_from_generation:
            # Remove the input_ids from the generated sequences
            len_input = len(input_ids[0])
            generation_outputs.sequences = generation_outputs.sequences[:, len_input:]
            if getattr(generation_outputs, "beam_indices", None) is not None:
                generation_outputs.beam_indices = generation_outputs.beam_indices[:, len_input:]

        decoded_sequences: List[str] = self.tokenizer.batch_decode(
            generation_outputs.sequences, skip_special_tokens=skip_special_tokens
        )

        # reset the model back to training mode if it was originally in training mode
        if is_in_training_mode:
            self.train()

        # convert to batched decoded sequences: from batch_size x num_beams, seq_len to batch_size, num_beams, seq_len
        num_beams = len(decoded_sequences) // len(input_ids)
        grouped_decoded_sequences = unflatten_generations(decoded_sequences, num_beams=num_beams)

        results = {"decoded_sequences": decoded_sequences,
                      "grouped_decoded_sequences": grouped_decoded_sequences}

        if return_generation_inputs:
            results["generate_kwargs"] = generate_kwargs
        if return_generation_outputs:
            results["generation_outputs"] = generation_outputs

        return results

    def on_sample_end(self, generation_outputs, **kwargs):
        raise NotImplementedError

    def test_epoch_end(self, outputs):
        pass

    def on_test_epoch_end(self):
        upload_outputs_to_wandb(
            getattr(self, "hparams_to_log", {}),
            self._get_predictions_dir_path(self.hparams.output_dir),
            logger=self.logger,
        )


    def _write_step_output(self, step_output: Dict):
        prediction_outputs = step_output

        prediction_outputs_path = os.path.join(
            self._get_predictions_dir_path(),
            f"testing_output_{self.global_rank}.prediction.jsonl",
        )
        prediction_outputs_summary = dict_of_lists_to_list_of_dicts(prediction_outputs)
        # write_gzipped_jsonlines(prediction_outputs_path, prediction_outputs_summary, mode="a+")
        auto_write_text(prediction_outputs_path, prediction_outputs_summary, mode="a+")
        pass

    def test_step(self, batch: Dict, batch_idx: int = None) -> Dict:
        raw_input = [sample["text"] for sample in batch["raw"]]
        raw_target = [sample["target"] for sample in batch["raw"]]
        ids = batch["id"]

        sample_output: Dict[str, Any] = self._get_predictions_for_batch(batch)


        return {
            "ids": ids,
            "inputs": raw_input,
            "targets": raw_target,
            "candidate_predictions": sample_output["grouped_decoded_sequences"],
        }

    def test_step_end(self, outputs: Dict[str, List[Any]]):
        final_predictions: List[Any] = self._get_final_prediction(outputs)

        outputs["final_predictions"] = final_predictions

        if self.hparams.output_dir:
            self._write_step_output(step_output=outputs)

        return outputs

    def _get_final_prediction(self, outputs: Dict[str, List[Any]]):
        final_predictions: List[Any] = [predictions[0] for predictions in outputs["candidate_predictions"]]
        return final_predictions

    def _get_predictions_dir_path(self, output_dir=None, create_if_not_exists=True):
        output_dir = output_dir or self.hparams.output_dir
        if output_dir is not None:
            predictions_folder = os.path.join(output_dir, "predictions")
        else:
            predictions_folder = "predictions"

        if create_if_not_exists:
            Path(predictions_folder).mkdir(parents=True, exist_ok=True)

        return predictions_folder


# Define configuration class for better parameter management
