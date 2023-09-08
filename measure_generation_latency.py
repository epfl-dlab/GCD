import argparse
import os
import time

from typing import Dict
import logging

from hydra import compose, initialize
from omegaconf import OmegaConf, DictConfig
from src.models.HFModelPLOld import HFModelPL

from src.utils.rank_zero_logger import get_only_rank_zero_logger

log = get_only_rank_zero_logger(__name__)


MODELS_DIR = os.environ["HF_MODELS_DIR"]
PGF_DIR = "assets/pgf"
initialize(config_path="configs/hydra_conf", version_base="1.2")


def measure_decoding_latency(pretrained_model_path: str, gf_grammar_module=None):
    # mandatory for PyTorch Lightning
    collator_parameters = {
        "max_input_length": 24,
        "padding": "longest",
        "truncation": True,
    }

    model = HFModelPL(
        from_pretrained=True,
        pretrained_model_name_or_path=pretrained_model_path,
        collator_parameters=collator_parameters,
        use_accelerate=False,
        half_precision=False,
        gf_constraint_module=gf_grammar_module,
    )

    texts = [""]

    inputs = model.tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    model.measure_generation_latency(inputs, num_new_tokens=10)


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
    parser.add_argument("--model", type=str, required=True, help="llama_1b")
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

    parser.add_argument("--log_dir", default="logs", type=str)
    parser.add_argument("--log_name", type=str)
    # set log path

    args = parser.parse_args()

    time_str = time.strftime("%m-%d-%H-%M")
    if not args.log_name:
        log_name = "{}".format(time_str)
    else:
        log_name = args.log_name
    log_dir = os.path.join(args.log_dir, log_name)
    # new_logger.set_logger_dir(log_dir)

    pretrained_model_path = os.path.join(MODELS_DIR, "llama_1b")

    if args.grammar_type:
        grammar_module_cfg_path = f"constraint/gf_constraint_module/{args.task}/{args.grammar_type}_{args.dataset}.yaml"

        gf_constraint_module: DictConfig = compose(config_name=grammar_module_cfg_path)[
            "model"
        ]["gf_constraint_module"]

        gf_constraint_module["grammar_dir"] = PGF_DIR
    else:
        gf_constraint_module = None

    measure_decoding_latency(pretrained_model_path, gf_constraint_module)

    measure_decoding_latency(pretrained_model_path, None)
