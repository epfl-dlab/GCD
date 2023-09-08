import os
from unittest import TestCase
from hydra import compose, initialize
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf, DictConfig
from configs.const import PGF_DIR, MODELS_DIR, HYDRACFG_DIR

from src.models.HFModelPLOld import HFModelPL
from src.models.config import HFModelPLConfig, IEHFModelPLConfig
from src.utils.hf_model_utils import get_hf_model_short_name


class TestHFModelPL(TestCase):
    def setUp(self):

        pretrained_model_path = os.path.join(MODELS_DIR, "1b")

        task = "CP"
        dataset = "ptb"
        grammar_type = "re"

        grammar_module_cfg_path = (
            f"constraint/gf_constraint_module/{task}/{grammar_type}_{dataset}.yaml"
        )

        with initialize(config_path="../configs/hydra_conf", version_base="1.2"):
            gf_constraint_module: DictConfig = compose(
                config_name=grammar_module_cfg_path
            )["constraint"]["gf_constraint_module"]

        gf_constraint_module["grammar_dir"] = PGF_DIR

        cs = ConfigStore.instance()
        cs.store(group="model", name="base_model", node=HFModelPLConfig)
        cs.store(group="model", name="ie_model", node=IEHFModelPLConfig)

        OmegaConf.register_new_resolver("hf_model_name", get_hf_model_short_name)
        OmegaConf.register_new_resolver("replace_slash", lambda x: x.replace("/", "_"))

        self.model = HFModelPL(
            from_pretrained=True,
            pretrained_model_name_or_path=pretrained_model_path,
            default_collator_parameters={},
            use_accelerate=False,
            half_precision=False,
            gf_constraint_module=gf_constraint_module,
        )

    def test_measure_generation_latency(self):
        texts = [""]

        pass
