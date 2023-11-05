import json
from typing import Dict, List

import transformers

from .abstract_dataset import BaseDataset


class TextDataset(BaseDataset):

    def __init__(self, tokenizer=None, **params):
        # tokenizer should be allowed to be None
        super().__init__(**params)
        self.tokenizer = tokenizer
        if type(self.tokenizer) == str:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer)
        if tokenizer is not None:
            assert self.tokenizer.eos_token_id is not None, """Tokenizer must have an EOS token but got None,
            this will cause None being added to the tokenized input. Tensor can not handle None values.
            Solution:
            1. add an EOS token to the tokenizer
            2. Make sure you are using an autoregressive model such as gpt2, bart, t5, etc.(not bert, roberta, etc.)
            2. change the implementation of this dataset
            """

    def __getitem__(self, idx) -> Dict:
        dp: Dict = self.data[idx]

        if self.tokenizer is not None:
            target_text = dp["target"]
            target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
            target_ids = target_ids + [self.tokenizer.eos_token_id]

            return {**dp, "target_ids": target_ids}
        else:
            return {**dp}

    def __len__(self):
        return len(self.data) if self.data is not None else 0

    def _load_data(self, path, **kwargs):
        self.data = self._read_data_from_file(path, **kwargs)

    def _read_data_from_file(self, path: str, n: int=None, **params) -> List[Dict]:
        # load from json
        ext = path.split(".")[-1]
        if ext == "json":
            data: List[Dict] = self._read_data_from_json(path)
        elif ext == "jsonl":
            data = self._read_data_from_jsonl(path)
        elif ext == "txt":
            data = self._read_data_from_txt(path)
        else:
            raise ValueError(f"Unrecognized file extension: {ext}")

        if n is not None:
            data = data[:n]
        return data

    def _read_data_from_json(self, path: str) -> List[Dict]:
        with open(path, "r") as f:
            data = json.load(f)
        return data

    def _read_data_from_jsonl(self, path: str) -> List[Dict]:
        with open(path, "r") as f:
            data = [json.loads(line) for line in f]
        return data

    def _read_data_from_txt(self, path: str) -> List[Dict]:
        with open(path, "r") as f:
            lines = [line.strip() for line in f]
        data = [{"id": i, "text": line} for i, line in enumerate(lines)]
        return data

class InMemoryTextDataset(TextDataset):
    def __init__(self, tokenizer, **params):
        super().__init__(tokenizer=tokenizer, **params)

    def load_data(self, data: List[Dict], **kwargs):
        super().load_data(path=None, data=data, **kwargs)

    def _load_data(self, path, data: List[Dict] = None, **kwargs):
        # check if data format is correct
        datum = data[0]
        if "id" not in datum:
            raise ValueError("Data must have 'id' field")
        if "text" not in datum:
            raise ValueError("Data must have 'text' field")
        if "target" not in datum:
            raise ValueError("Data must have 'target' field")

        self.data = data
