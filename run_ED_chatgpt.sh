#! /usr/bin/bash

# Model and device selection
name="gpt-3.5-turbo" # set default to 1B

# Uncomment if you want to export LLAMA_DIR
# export LLAMA_DIR="$LLAMA_DIR"
# Experiment configuration
debug_k="${2}"
exp_option=stable  # beam = 4
trainer_option="${2:-cpu}" # set default to cpu
model_option=ChatGPTmodel_ed

for name in gpt-3.5-turbo-0301; 
do
    endpoint_name="chat_completion"
    for ds in msnbc ace2004; 
    do
        datamodule_option="ED/ed_${ds}_stable"
        grammar_module="ED/canonical/$ds"

        echo "DATAMODULE=$datamodule_option"
        echo "with constraints=$grammar_module"

        python run_inference.py \
            +experiment/ED/inference="$exp_option" \
            datamodule="$datamodule_option" \
            trainer="$trainer_option" \
            model="$model_option" \
            model.pretrained_model_name_or_path="/dlabdata1/llama_hf/1B/" \
            +constraint/gf_constraint_module/ED@model.gf_constraint_module=canonical \
            model.half_precision=false \
            model.gf_constraint_module.grammar_module="$grammar_module" \
            model.openai_api_key="$API_KEY" \
            model.endpoint_name="$endpoint_name" \
            model.name="$name"\
            logger.wandb.offline=false \
            hydra.verbose=false 
    done
done

for name in gpt-3.5-turbo-instruct-0914 text-davinci-003 davinci-002; 
    do
    endpoint_name="completion"
    for ds in msnbc ace2004; 
    do
        datamodule_option="ED/ed_${ds}_stable"
        grammar_module="ED/canonical/$ds"

        echo "DATAMODULE=$datamodule_option"
        echo "with constraints=$grammar_module"

        python run_inference.py \
            +experiment/ED/inference="$exp_option" \
            datamodule="$datamodule_option" \
            trainer="$trainer_option" \
            model="$model_option" \
            model.pretrained_model_name_or_path="/dlabdata1/llama_hf/1B/" \
            +constraint/gf_constraint_module/ED@model.gf_constraint_module=canonical \
            model.half_precision=false \
            model.gf_constraint_module.grammar_module="$grammar_module" \
            model.openai_api_key="$API_KEY" \
            model.endpoint_name="$endpoint_name" \
            model.name="$name"\
            logger.wandb.offline=false \
            hydra.verbose=false
    done
done