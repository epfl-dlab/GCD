import os
from pathlib import Path

from typing import Dict, Union, List, Any

import hydra
import pytorch_lightning as pl
import torch
import transformers
from omegaconf import OmegaConf, open_dict

import src.utils as utils
from src.utils.evaluation_helpers import upload_outputs_to_wandb
from src.utils.hf_gen_utils import unflatten_generations
from src.utils.dict_utils import dict_of_lists_to_list_of_dicts
from src.utils.file_io import auto_write_text
from src.utils.generic_text_collator import GenericTextCollator
from src.constrained_generation.gf_constrained import GF_ConstrainedWithRenaming
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)
import time

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random,
) 
log = utils.get_only_rank_zero_logger(__name__)
import openai
import wandb 

class ChatGPTModel(pl.LightningModule):
    def __init__(self, params: Union[Dict, OmegaConf]):
        super().__init__()
        # equivalent to self.hparams = config
        self.save_hyperparameters(params)
        self.model_type = params.name
        self.endpoint_name = params.endpoint_name
        openai.api_key = params.openai_api_key
        #self._init_constraint_module()
        #self._init_model_and_tokenizer()
        #self._init_collator()
        self.constraint_module = None
        
    def _init_model_and_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.hparams.pretrained_model_name_or_path)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _init_collator(self):
        self.collator = GenericTextCollator(
            tokenizer=self.tokenizer, **self.hparams.collator_parameters
        )
    
    def _init_constraint_module(self):
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
            
    @retry(wait=wait_random(min=5, max=60))
    def _api_request(self,prompt,**kwargs):
        if self.endpoint_name == "chat_completion":
            response = openai.ChatCompletion.create(
                model=self.model_type,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                **kwargs)
            return response['choices'][0]['message']['content']

        elif self.endpoint_name == "completion":
            response = openai.Completion.create(
                model=self.model_type,
                prompt=prompt,
                **kwargs)
            return response['choices'][0]['text']
    
    def _call_openai(self, prompts,**kwargs):
        responses = []
        for prompt in prompts:
            responses.append(self._api_request(prompt,**kwargs))
        return responses

        
    def forward(self, input_ids, attention_mask, decoder_attention_mask, labels=None, **kwargs):
        input_text = kwargs["text"]
        output = _call_openai(input_text)
        return output

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

        chatgpt_generation_params = self.hparams.inference["chatgpt_generation_params"].copy()
        sampling_params = {
            "input_is_processed_batch": True,  # TODO
            "return_generation_inputs": True,
            "return_generation_outputs": True,
            "output_scores": True,
            "chatgpt_generation_params": chatgpt_generation_params,
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
            remove_prefix_from_generation=False,
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
        runtime_generation_kwargs = inference.get("chatgpt_generation_params", {})
        extra_generation_params = self.hparams.inference.get("chatgpt_generation_params", {}).copy()

        with open_dict(extra_generation_params):
            extra_generation_params.update(runtime_generation_kwargs)
            #extra_generation_params["return_dict_in_generate"] = True

        if seed is None:
            seed = inference.get("seed", None)
        if seed:
            transformers.trainer_utils.set_seed(seed)

        # Get input_ids and attention masks
        if not input_is_processed_batch:
            input_data = self.collator.collate_input(input_data)
            
        #input_ids = input_data["src_input_ids"].to(self.device)
        #attention_mask = input_data["src_attention_mask"].to(self.device)
        if "raw" in input_data.keys():
            input_prompts = [sample["text"] for sample in input_data["raw"]]
        elif "text" in input_data.keys():
            input_prompts = [sample for sample in input_data["text"]]
            
        if prefix_allowed_tokens_fn is None and self.constraint_module is not None:
            raise RuntimeError("this should not happen")

        generate_kwargs = {
            #"prefix_allowed_tokens_fn": prefix_allowed_tokens_fn,
            **extra_generation_params,
        }
        generation_outputs = self._call_openai(input_prompts, **generate_kwargs)

        if remove_prefix_from_generation:
            # Remove the input_ids from the generated sequences
            len_input = len(input_ids[0])
            generation_outputs.sequences = generation_outputs.sequences[:, len_input:]
            if getattr(generation_outputs, "beam_indices", None) is not None:
                generation_outputs.beam_indices = generation_outputs.beam_indices[:, len_input:]

        decoded_sequences: List[str] = generation_outputs
        # reset the model back to training mode if it was originally in training mode
        if is_in_training_mode:
            self.train()

        # convert to batched decoded sequences: from batch_size x num_beams, seq_len to batch_size, num_beams, seq_len
        num_beams = len(decoded_sequences) // len(input_prompts)
        grouped_decoded_sequences = unflatten_generations(decoded_sequences, num_beams=num_beams)

        results = {"decoded_sequences": decoded_sequences,
                      "grouped_decoded_sequences": grouped_decoded_sequences}

        if return_generation_inputs:
            results["generate_kwargs"] = generate_kwargs
        if return_generation_outputs:
            results["generation_outputs"] = generation_outputs
            
        return results
    
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
        log.info(batch_idx)
        if "raw" in batch.keys():
            raw_input = [sample["text"] for sample in batch["raw"]]
            raw_target = [sample["target"] for sample in batch["raw"]]
        elif "text" in batch.keys() and "target" in batch.keys():
            raw_input = [sample for sample in batch["text"]]
            raw_target = [sample for sample in batch["target"]]
        ids = batch["id"]
        if type(ids) == torch.Tensor:
            ids = ids.tolist()
        sample_output: Dict[str, Any] = self._get_predictions_for_batch(batch)
        
        #logger = self.logger.experiment[0]
        #table_output = [ids + raw_input + raw_target + sample_output["grouped_decoded_sequences"][0]]
        #columns=["id", "input", "target", "prediction"]
        #table = wandb.Table(data=table_output, columns=columns)
        #logger.log({"test/sentences":table})
        log.info(raw_input)
        log.info(sample_output["grouped_decoded_sequences"][0])
        log.info(raw_target)
        
        return {
            "ids": ids,
            "inputs": raw_input,
            "targets": raw_target,
            "unflattened_predictions": sample_output["grouped_decoded_sequences"],
        }

    def test_step_end(self, outputs: Dict[str, List[Any]]):
        structured_prediction: List[Any] = self._get_structured_prediction(outputs)

        outputs["structured_predictions"] = structured_prediction

        if self.hparams.output_dir:
            self._write_step_output(step_output=outputs)

        return outputs

    def _get_structured_prediction(self, outputs: Dict[str, List[Any]]):
        structured_prediction: List[Any] = [predictions[0] for predictions in outputs["unflattened_predictions"]]
        return structured_prediction

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
