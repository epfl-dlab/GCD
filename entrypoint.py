import os
import subprocess
import argparse
from hydra import compose, initialize
from omegaconf import OmegaConf

import logging

logger = logging.getLogger(__name__)

MODEL_DIR = os.getenv("MODEL_DIR")



def run_experiment(model, task, grammar_type, dataset):

    running_strategy = "cpu"
    model_path = f"{MODEL_DIR}/{model}"
    exp_cfg = "llama_dep_basic"
    datamodule_cfg = f"ED/ed_{dataset}_stable"
    half_precision = False

    trainer_cfg_arg = f"trainer={running_strategy}"
    exp_cfg_arg = f"+experiment/{task}/inference={exp_cfg}"
    datamodule_cfg_arg = f"datamodule={datamodule_cfg}"
    half_precision_arg = f"model.half_precision={half_precision}"
    grammar_cfg_arg = f"+constraint/gf_constraint_module/{task}@model.gf_constraint_module={grammar_type}_{dataset}"
    model_arg = f"model.pretrained_model_name_or_path={model_path}"

    overrides = [
        trainer_cfg_arg,
        exp_cfg_arg,
        datamodule_cfg_arg,
        half_precision_arg,
        grammar_cfg_arg,
        model_arg
    ]

    with initialize(config_path="configs/hydra_conf", job_name="test_app"):
        cfg = compose(config_name="inference_root", overrides=overrides)
    print(OmegaConf.to_yaml(cfg))



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Experiments")
    # Define valid tasks, grammars, and datasets using a nested dictionary.
    TASKS = {
        "IE": {
            "grammar_type": ["fe", "sc"],
            "dataset": ["wikinre", "rebel_1M", "rebel_6M"],
        },
        "CP": {"grammar_type": ["re"], "dataset": ["ptb"]},
        "ED": {
            "grammar_type": ["minimal", "canonical"],
            "dataset": ["aida", "ace2004", "aquaint", "clueweb", "msnbc", "wiki"],
        },
    }

    # General Arguments
    parser.add_argument(
        "--model", type=str, help="Model name (e.g., 7B, 13B, 33B, 65B)",
        default="llama-1b",
    )
    parser.add_argument(
        "--task",
        choices=TASKS.keys(),
        required=True,
        help="Specify the task to be performed.",
    )
    parser.add_argument(
        "--grammar_type", type=str, default=None, help="Specify the grammar type."
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="Specify the dataset to be used."
    )

    args = parser.parse_args()

    run_experiment(args.model, args.task, args.grammar_type, args.dataset)
