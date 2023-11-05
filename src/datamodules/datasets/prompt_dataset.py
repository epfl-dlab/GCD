from typing import Dict

from .filtered_text_dataset import FilteredTextDataset


class PromptDataset(FilteredTextDataset):

    def __init__(self, prompter=None, **params):
        super().__init__(**params)
        self.prompter = prompter

    def _preprocess_data_point(self, dp: Dict, **kwargs) -> Dict:
        if self.prompter is not None:
            dp["text"] = self.prompter.materialize(runtime_input=dp, **kwargs)
        return dp
