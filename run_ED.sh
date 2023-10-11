#!/bin/bash

# Model and device selection
model="${1:-saibo/llama-1B}" # set default to 1B

# Uncomment if you want to export LLAMA_DIR
# export LLAMA_DIR="$LLAMA_DIR"

# Experiment configuration
exp_option=stable  # beam = 4
trainer_option="${2:-cpu}" # set default to cpu
model_option=HFmodel_ed

# Loop over datasets and run inference
for ds in aquaint msnbc ace2004 wiki aida clueweb; do
    datamodule_option="ED/ed_${ds}_stable"
    grammar_module="ED/canonical/$ds"

    echo "DATAMODULE=$datamodule_option"
    echo "with constraints=$grammar_module"

    python run_inference.py \
        +experiment/ED/inference="$exp_option" \
        datamodule="$datamodule_option" \
        trainer="$trainer_option" \
        model="$model_option" \
        +constraint/gf_constraint_module/ED@model.gf_constraint_module=canonical \
        model.pretrained_model_name_or_path="$HF_MODELS_DIR/$model" \
        model.half_precision=false \
        model.gf_constraint_module.grammar_module="$grammar_module" \
        datamodule.debug_k=16 \
        logger.wandb.offline=false \
        hydra.verbose=false # set log level to debug
        # Additional options (uncomment if needed):
        # --cfg job --resolve

done
