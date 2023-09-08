#!/bin/bash

# Command-line arguments
model="${1:-saibo/llama-1B}" # set default to 1B
#n_device="$4"      # Options: 1, 1, 2, 3, "1,2"
format="${2:-fe}" # Options: fe, sc, feR, scR
debug_k="${3}"  # Options: 1, 2, 4, 8, 16, 32, 64, 128, 256

# Compute linearization from format
linearize="${format:0:2}"

# Uncomment if you want to export LLAMA_DIR
# export LLAMA_DIR="/scratch/saibo"

# Experiment configuration
exp_option="stable_$linearize"  # beam = 2 by default
trainer_option=cpu
model_option=HFmodel_ie

# Loop over datamodules and run inference
for datamodule_option in IE/ie_rebel_small_stable IE/ie_synthie_small_stable
do
    echo "DATAMODULE=$datamodule_option"
    echo "with constraints"

    python run_inference.py \
        +experiment/IE/inference="$exp_option" \
        +constraint/gf_constraint_module/IE@model.gf_constraint_module=fe_wikinre \
        datamodule="$datamodule_option" \
        trainer="$trainer_option" \
        model="$model_option" \
        model.pretrained_model_name_or_path="$HF_MODELS_DIR/$model" \
        model.half_precision=false \
        datamodule.debug_k="$debug_k" \
        logger.wandb.offline=false \
        hydra.verbose=false
        # --cfg job --resolve
        # Additional options (uncomment if needed):

done
