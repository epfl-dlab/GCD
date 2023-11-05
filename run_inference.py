import wandb
from hydra.core.config_store import ConfigStore

from src.models.ED_model import EDHFModelPL
from src.models.config import HFModelPLConfig, IEHFModelPLConfig
from src.utils import hydra_custom_resolvers
from pathlib import Path
from src import utils
import hydra
import os
from omegaconf import DictConfig, OmegaConf

from src.utils import general_helpers
from typing import List

import pytorch_lightning as pl
from pytorch_lightning import LightningModule, LightningDataModule, Callback, Trainer
from pytorch_lightning.loggers import LightningLoggerBase

from src.utils.hf_model_utils import get_hf_model_short_name

# wandb.init(reinit=True, mode="online")


log = utils.get_only_rank_zero_logger(__name__, stdout=True)

def run_inference(cfg: DictConfig):
    assert (
        cfg.output_dir is not None
    ), "Path to the directory in which the predictions will be written must be given"
    cfg.output_dir = general_helpers.get_absolute_path(cfg.output_dir)
    log.info(f"Output directory: {cfg.output_dir}")

    # Set seed for random number generators in PyTorch, Numpy and Python (random)
    if cfg.get("seed"):
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating data module <{cfg.datamodule._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(
        cfg.datamodule, _recursive_=False
    )

    log.info(f"Instantiating model <{cfg.model._target_}>")

    model_class = hydra.utils.get_class(cfg.model._target_)
    model = model_class(cfg.model)

    # datamodule.set_tokenizer(model.tokenizer)
    # If defined, use the model's collate function (otherwise proceed with the PyTorch's default collate_fn)
    if getattr(model, "collator", None):
        datamodule.set_collate_fn(model.collator.collate_fn)

    log.info("Instantiating loggers...")
    logger: List[LightningLoggerBase] = general_helpers.instantiate_loggers(
        cfg.get("logger")
    )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, logger=logger
    )  # callbacks=callbacks)

    logging_object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "logger": logger,
        "trainer": trainer,
    }
    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(logging_object_dict)

    log.info("Starting testing!")
    model.output_dir = cfg.output_dir
    model.work_dir = cfg.work_dir
    trainer.test(model=model, datamodule=datamodule)

    # for predictions use trainer.predict(...)
    # predictions = trainer.predict(model=model, dataloaders=dataloaders, ckpt_path=cfg.ckpt_path)

    metric_dict = trainer.callback_metrics
    if metric_dict:
        log.info("Metrics dict:")
        log.info(metric_dict)


cs = ConfigStore.instance()
cs.store(group="model", name="base_model", node=HFModelPLConfig)
cs.store(group="model", name="ie_model", node=IEHFModelPLConfig)

OmegaConf.register_new_resolver("hf_model_name", get_hf_model_short_name)
OmegaConf.register_new_resolver("replace_slash", lambda x: x.replace("/", "_"))


@hydra.main(
    version_base="1.2", config_path="configs/hydra_conf", config_name="inference_root"
)
def main(hydra_config: DictConfig):
    utils.run_task(hydra_config, run_inference)


if __name__ == "__main__":
    main()
