import os

import hydra.utils
from hydra import compose, initialize
from omegaconf import OmegaConf
from pytorch_lightning import LightningDataModule
from transformers import LlamaTokenizer

# a bit like init a config file system
initialize(config_path="configs/hydra_conf", job_name="test_app", version_base="1.2")

llama_dir = os.environ.get("LLAMA_DIR", None)
# load tokenizer
tokenizer = LlamaTokenizer.from_pretrained(os.path.join(llama_dir, "1B"))


cfg = compose(
    config_name="datamodule/CP/cp_ptb_stable.yaml",
    overrides=[
        "+data_dir=data",
        "+seed=42",
        "+assets_dir=assets",
        "datamodule.batch_size=2",
    ],
)

datamodule: LightningDataModule = hydra.utils.instantiate(cfg, _recursive_=False)[
    "datamodule"
]


datamodule.set_tokenizer(tokenizer)
# If defined, use the model's collate function (otherwise proceed with the PyTorch's default collate_fn)
# datamodule.set_collate_fn(model.collator.collate_fn)

datamodule.setup("test")

x = datamodule.test_dataloader()
