
from .datamodule import DataModule
from src.datamodules.datasets.prompt_dataset import PromptDataset
from src.datamodules.datasets.ie_dataset import IEDataset

IEInputDataset = IEDataset
ELInputDataset = PromptDataset
CPInputDataset = PromptDataset
