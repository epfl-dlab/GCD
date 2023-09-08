import os
import statistics
import time
import numpy as np
from typing import List, Any, Dict

import hydra
import pandas as pd
import torch
import transformers
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities import rank_zero_only
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

import src.utils as utils
from src.utils.generic_text_collator import GenericTextCollator
from src.utils import general_helpers, evaluation_helpers

log = utils.get_only_rank_zero_logger(__name__)


class HFModelPL(LightningModule):
    def __init__(
        self,
        hparams_overrides: Dict = None,
        hf_config_overrides: Dict = None,
        from_pretrained=False,
        trie_constraint_module: Dict = None,
        gf_constraint_module: Dict = None,
        **kwargs,
    ):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters(
            logger=False,
            ignore=[
                "hparams_overrides",
                "hf_config_overrides",
                "datamodule",
                "collator",
                "constraint_module",
            ],
        )

        if hparams_overrides is not None:
            self._override_checkpoint_hparams(hparams_overrides)

        # ~~~ Load the tokenizer ~~~
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hparams.pretrained_model_name_or_path
        )
        # set padding token to pad_token_id
        self.tokenizer.pad_token = (
            self.tokenizer.eos_token
        )  # llama doesn't have a pad token, so we use eos instead
        # using the eos token as the pad token is recommended in the error message:
        # Please select a token to use as `pad_token` `(tokenizer.pad_token = tokenizer.eos_token e.g.)`

        # ~~~ Get the HF config ~~~
        hf_config = AutoConfig.from_pretrained(
            self.hparams.pretrained_model_name_or_path
        )
        # Override the HF config with values from the checkpoint (if loading from checkpoint)
        if self.hparams.get("hf_config", None):
            hf_config.update(self.hparams.hf_config.to_dict())
        # Override HF config parameters (if it applies)
        if hf_config_overrides is not None:
            hf_config.update(hf_config_overrides)
        # Update the hparams with the updated config
        self.hparams.hf_config = hf_config

        # ~~~ Load the model ~~~
        if from_pretrained:
            if self.hparams.half_precision:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.hparams.pretrained_model_name_or_path,
                    config=self.hparams.hf_config,
                    torch_dtype=torch.half,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.hparams.pretrained_model_name_or_path,
                    config=self.hparams.hf_config,
                )
        else:
            self.model = AutoModelForCausalLM(config=self.hparams.hf_config)

        # half_precision = kwargs.get("half_precision", False)
        if self.hparams.half_precision & (not self.hparams.use_accelerate):
            self.model = self.model.half()

        log.info("HF model config:")
        log.info(self.hparams.hf_config)

        # ~~~ Set collator ~~~
        self.collator = kwargs.get("collator", None)
        if self.collator is None:
            self.collator = self._get_default_collator()
        else:
            self.collator.set_tokenizer(self.tokenizer)

        if gf_constraint_module is not None and trie_constraint_module is not None:
            raise ValueError(
                "You cannot specify both a GF constraint module and a Trie constraint module"
            )

        # ~~~ Constraint generation ~~~
        if gf_constraint_module is not None:
            log.info("Running inference with GF-CONSTRAINED decoding")
            self.constraint_module = hydra.utils.instantiate(gf_constraint_module)
        elif trie_constraint_module is not None:
            log.info("Running inference with Trie-CONSTRAINED decoding")
            self.constraint_module = hydra.utils.instantiate(
                trie_constraint_module, model=self
            )
        else:
            log.info("Running UNCONSTRAINED inference.")
            self.constraint_module = None

    def _override_checkpoint_hparams(self, hparams_overrides: dict):
        """
        Overrides the hyperparameters of a checkpoint at an arbitrary depth
        :param hparams_overrides:
        :return:
        """
        general_helpers.rec_dict_update(self.hparams, hparams_overrides)
        log.info("Some values of the original hparams were overridden")
        log.info("Hyper-parameters:")
        log.info(self.hparams)

    def _get_default_collator(self):
        return GenericTextCollator(
            tokenizer=self.tokenizer, **self.hparams.default_collator_parameters
        )

    def forward(
        self, input_ids, attention_mask, decoder_attention_mask, labels=None, **kwargs
    ):
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

    def _get_predictions_for_batch(self, batch, raw_input):
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
        hf_generation_params.update(
            {
                "input_is_processed_batch": True,
                "return_generation_inputs": True,
                "return_generation_outputs": True,
                "output_scores": True,
            }
        )

        sample_output = self.sample(
            batch,
            prefix_allowed_tokens_fn=sample_prefix_allowed_tokens_fn,
            **hf_generation_params,
        )

        return sample_output

    def test_step(self, batch, batch_idx):
        raw_input = [sample["text"] for sample in batch["raw"]]
        raw_target = [sample["target"] for sample in batch["raw"]]
        ids = batch["id"]
        sample_output = self._get_predictions_for_batch(batch, raw_input)

        # self._sanity_check(batch, batch_idx, stage="test", predictions=sample_output["grouped_decoded_outputs"]) TODO disabled for now to include ED task
        # self._write_step_output(
        #     batch_idx=batch_idx, ids=ids, raw_input=raw_input, raw_target=raw_target, sample_output=sample_output
        # )

        return_object = {
            "ids": ids,
            "inputs": raw_input,
            "targets": raw_target,
            "predictions": sample_output["grouped_decoded_outputs"],  # here
        }
        return return_object

    def test_step_end(self, outputs: List[Any]):
        raise NotImplementedError

    def _write_step_output(
        self,
        ids,
        inputs,
        targets,
        predictions,
        final_prediction,
    ):
        # ~~~ Write prediction outputs to file ~~~
        num_return_sequences = len(predictions[0])
        # sequences = predictions["generation_outputs"].sequences
        # assert isinstance(sequences, torch.Tensor)
        # prediction_ids = general_helpers.chunk_elements(sequences.tolist(), num_return_sequences)

        # tokenizer_output = self.tokenize(raw_input, raw_target)
        # target_decoder_input_ids = tokenizer_output["decoder_input_ids"]
        prediction_outputs = {
            "id": ids,
            "input": inputs,
            # "input_ids": sample_output["generation_inputs"]["input_ids"].tolist(),
            "target": targets,
            # "target_ids": target_decoder_input_ids.tolist(),
            "prediction": predictions,
            # "prediction_ids": str(prediction_ids),
            "final_prediction": final_prediction,
        }
        # if seeds is not None:
        #     prediction_outputs["seed"] = seeds

        prediction_outputs_path = os.path.join(
            evaluation_helpers.get_predictions_dir_path(self.output_dir),
            f"testing_output_{self.global_rank}.prediction.jsonl.gz",
        )

        prediction_outputs_summary = general_helpers.get_list_of_dicts(
            prediction_outputs
        )
        general_helpers.write_gzipped_jsonlines(
            prediction_outputs_path, prediction_outputs_summary, mode="a+"
        )

        # ––––– Log a few batches during inference as a sanity check
        # if batch_idx in [0, 3] and self.global_rank == 0:
        #     # pred_json = json.dumps(prediction_outputs_summary)
        #     # log.info(f"test_output/batch_{batch_idx}:\n{pred_json}")
        #
        #     pred_df = pd.DataFrame(prediction_outputs)
        #     utils.general_helpers.log_df(path=f"test_batch_summary/batch_{batch_idx}", df=pred_df, logger=self.logger)

    def test_epoch_end(self, outputs):
        raise NotImplementedError

    def on_test_epoch_end(self):
        if (
            hasattr(torch.distributed, "is_initialized")
            and torch.distributed.is_initialized()
        ):
            torch.distributed.barrier()

        # Temporary solution to Hydra + PL + DDP issue
        # https://github.com/Lightning-AI/lightning/pull/11617#issuecomment-1245842064
        # https://github.com/ashleve/lightning-hydra-template/issues/393
        # problem should be resolved in PL version 1.8.3
        general_helpers._move_predictions_for_subprocesses(
            evaluation_helpers.get_predictions_dir_path(os.getcwd()),
            evaluation_helpers.get_predictions_dir_path(self.output_dir),
        )

        evaluation_helpers.upload_outputs_to_wandb(
            getattr(self, "hparams_to_log", {}),
            evaluation_helpers.get_predictions_dir_path(self.output_dir),
            logger=self.logger,
        )

    @torch.no_grad()
    def sample(
        self,
        input_data,
        input_is_processed_batch=False,
        seed=None,
        skip_special_tokens=True,
        return_generation_outputs=False,
        return_generation_inputs=False,
        prefix_allowed_tokens_fn=None,
        return_only_generated_text=True,
        **kwargs,
    ):
        """Input data is a list of strings or a processed batch (contains src_input_ids,
        and src_attention_mask as expected in training)"""
        # if the model is not in evaluation mode, set it and remember to reset it
        training = self.training
        if training:
            self.eval()

        hf_generation_params = self.hparams.inference["hf_generation_params"].copy()
        hf_generation_params.update(kwargs)
        hf_generation_params["return_dict_in_generate"] = True

        if seed is None:
            seed = self.hparams.inference.get("seed", None)
        if seed:
            transformers.trainer_utils.set_seed(seed)

        # Get input_ids and attention masks
        if not input_is_processed_batch:
            input_data = self.collator.collate_input(input_data)

        input_ids = input_data["src_input_ids"].to(self.device)
        attention_mask = input_data["src_attention_mask"].to(self.device)

        if prefix_allowed_tokens_fn is None and self.constraint_module is not None:
            raise RuntimeError("this should not happen")
            # prefix_allowed_tokens_fn = (
            #     self.constraint_module.get_prefix_allowed_tokens_fn(batch_info=None)
            # )

        generate_kwargs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
            **hf_generation_params,
        }
        generation_outputs = self.model.generate(**generate_kwargs)

        if return_only_generated_text:
            # Remove the input_ids from the generated sequences
            len_input = len(input_ids[0])
            generation_outputs.sequences = generation_outputs.sequences[:, len_input:]
            if getattr(generation_outputs, "beam_indices", None) is not None:
                generation_outputs.beam_indices = generation_outputs.beam_indices[
                    :, len_input:
                ]

        decoded_sequences = self.tokenizer.batch_decode(
            generation_outputs.sequences, skip_special_tokens=skip_special_tokens
        )

        if training:
            self.train()

        results = {"decoded_sequences": decoded_sequences}
        if return_generation_inputs:
            results["generate_kwargs"] = generate_kwargs
        if return_generation_outputs:
            results["generation_outputs"] = generation_outputs

        # log all items in results
        for key, value in results.items():
            log.debug(f"sample: {key} = {value}")
        return results

    def on_sample_end(self, generation_outputs, **kwargs):
        raise NotImplementedError

    def measure_generation_latency(
        self, inputs, num_new_tokens=100, num_runs=10, **hf_generation_params
    ):

        batch_size = inputs["input_ids"].shape[0]

        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        batch_info = None  # TODO

        if self.constraint_module is not None:
            prefix_allowed_tokens_fn = (
                self.constraint_module.get_prefix_allowed_tokens_fn(
                    batch_info=batch_info
                )
            )
        else:
            prefix_allowed_tokens_fn = None

        generate_kwargs = {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
            "max_new_tokens": num_new_tokens,
            "min_new_tokens": num_new_tokens,
            "num_return_sequences": batch_size,
            **hf_generation_params,
        }

        latency_per_token = []

        for _ in range(num_runs):
            start = time.time()
            generation_outputs = self.model.generate(**generate_kwargs)
            elapsed_time = time.time() - start
            latency_per_token.append((elapsed_time / num_new_tokens * batch_size))

        mean_latency_per_token = np.mean(latency_per_token)
        std_latency_per_token = np.std(latency_per_token)

        new_logger.info(f"constraint_module: {self.constraint_module}")
        new_logger.debug(f"generation_outputs: {generation_outputs}")
        new_logger.info(f"Mean    Latency per token: {mean_latency_per_token:.4f}")
        new_logger.info(f"Std     Latency per token: {std_latency_per_token:.4f}")

        return mean_latency_per_token, std_latency_per_token
