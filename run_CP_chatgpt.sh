#! /usr/bin/bash

# Model selection
name="gpt-3.5-turbo-0613" # set default to 1B

# Task and experiment options
task_option="cp"
exp_option="stable"  # beam = 4
endpoint_name="chat_completion"

# Model and training configurations
model_option="ChatGPTmodel_cp"
trainer_option="${2:-cpu}" # set default to cpu
datamodule_option="CP/cp_ptb_stable"
gf_constraint_module_option="re_ptb"

# Command execution
python run_inference.py \
    +experiment/CP/inference="$exp_option" \
    datamodule="$datamodule_option" \
    trainer="$trainer_option" \
    model="$model_option" \
    +constraint/gf_constraint_module/CP@model.gf_constraint_module="$gf_constraint_module_option" \
    model.pretrained_model_name_or_path="/dlabdata1/llama_hf/1B/" \
    model.half_precision=false \
    logger.wandb.offline=false \
    model.openai_api_key="$API_KEY" \
    model.endpoint_name="$endpoint_name" \
    model.name="$name"

