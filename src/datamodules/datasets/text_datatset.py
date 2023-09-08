import json
from typing import Dict, List

import transformers

from .abstract_dataset import BaseDataset


class TextDataset(BaseDataset):

    def __init__(self, tokenizer, **params):
        super().__init__(**params)
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            raise ValueError("Please provide a tokenizer")
        elif type(self.tokenizer) == str:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer)

    def __getitem__(self, idx) -> Dict:
        dp: Dict = self.data[idx]
        target_text = dp["target"]
        target_ids = self.tokenizer(target_text, add_special_tokens=False)["input_ids"]
        # add eos token
        target_ids = target_ids + [self.tokenizer.eos_token_id]

        return {
            "id": dp["id"],
            "text": dp["text"],
            "target": target_text,
            "target_ids": target_ids,
        }

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
        return data[:n] if n is not None else data

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
