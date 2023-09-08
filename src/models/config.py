from dataclasses import dataclass, field
from typing import Optional, Dict

from omegaconf import MISSING

from src.utils.hf_model_utils import get_hf_model_short_name


@dataclass
class HFModelPLConfig:
    name: Optional[str] = None
    _target_: Optional[str] = None
    pretrained_model_name_or_path: Optional[str] = None
    gf_constraint_module: Optional[Dict] = None
    trie_constraint_module: Optional[Dict] = None
    hf_extra_model_config: Dict = field(default_factory=dict)
    half_precision: bool = False
    use_accelerate: bool = False
    collator_parameters: Dict = field(default_factory=dict)
    inference: Dict = field(default_factory=dict)
    output_dir: Optional[str] = None

    def __post_init__(self):
        if "hf_generation_params" not in self.inference:
            self.inference["hf_generation_params"] = {}

        self.name = get_hf_model_short_name(self.pretrained_model_name_or_path)

@dataclass
class IEHFModelPLConfig(HFModelPLConfig):
    # linearization_class_id: Optional[str] = None
    linearization_class_id: str = MISSING
