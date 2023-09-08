import logging
from typing import Dict, List, Callable

from tqdm import tqdm

from .text_datatset import TextDataset

log = logging.getLogger(__name__)


def _get_num_tokens(text, tokenizer) -> int:
    return len(tokenizer(text)["input_ids"])


class FilteredTextDataset(TextDataset):

    def __init__(self, tokenizer, max_num_tokens_input=None, max_num_tokens_target=None, **params):
        super().__init__(tokenizer, **params)
        self.num_filtered_datapoints: Dict[str, int] = {}
        self.num_filtered_datapoints_constrained = 0
        self.params["max_num_tokens_input"] = max_num_tokens_input if max_num_tokens_input is not None else float("inf")
        self.params["max_num_tokens_target"] = max_num_tokens_target if max_num_tokens_target is not None else float(
            "inf")

    def _before_loading_data(self, **params):
        self.prepare_filtering(**params)

    def _after_loading_data(self, **params):
        #######################
        #
        # Post Loading, logging
        #
        #######################

        path = self.params.get("path", None)

        log.info(f"Loaded {len(self.data)} datapoints from {path}")

        log.info(
            f"[# tokens filtered out] {self.num_filtered_datapoints} "
        )

    def prepare_filtering(self, **params):
        pass

    @staticmethod
    def _get_input_num_tokens(
            dp: Dict,
            field: str,
            tokenizer,
            transformation: Callable = None,
            src_field: str = None,
    ) -> int:
        num_tokens_dict = dp.get("num_tokens_dict", {})

        # if the number of tokens is already computed, return it
        if field in num_tokens_dict:
            return num_tokens_dict[field]

        # otherwise, compute it and store it
        if transformation is None:
            num_tokens_dict[field] = _get_num_tokens(dp[field], tokenizer)
        else:
            num_tokens_dict[field] = _get_num_tokens(
                transformation(dp[src_field]), tokenizer
            )
        dp["num_tokens_dict"] = num_tokens_dict
        return num_tokens_dict[field]

    def _are_num_tokens_within_bounds(self, dp: Dict, bounds: Dict) -> bool:
        """
        bounds: {"input": 512, "target": 512}
        """
        if bounds is None:
            return True

        self.num_filtered_datapoints={field:0 for field in bounds.keys()}

        for field, bound in bounds.items():
            num_tokens = FilteredTextDataset._get_input_num_tokens(
                dp, field, self.tokenizer
            )
            if num_tokens > bound:
                self.num_filtered_datapoints[field] += 1
                return False
        # all within bounds
        return True

    def __filter_data(self, data: List[Dict], bounds: Dict[str, int], n: int = None) -> List[Dict]:
        filtered_data: List[Dict] = []

        for dp in tqdm(data, desc=f"filtering data"):
            dp["text"] = dp["text"].strip()
            if self._include_datapoint(dp):
                dp: Dict = self._preprocess_data_point(dp)
                if self._are_num_tokens_within_bounds(dp, bounds):
                    filtered_data.append(dp)
            # jump out of the loop if we have enough datapoints
            if n is not None and len(filtered_data) == n:
                break
        return filtered_data

    def _load_data(self, path: str, n: int = None, bounds: Dict[str, int] = None, **params):

        log.info(f"Loading the data from: {path} -- ")

        # we read all the data with n=None and then filter it to keep only n datapoints
        data: List[Dict] = self._read_data_from_file(path,n=None)

        self.data: List[Dict] = self.__filter_data(data, n=n, bounds=bounds)

    def _include_datapoint(self, dp: dict) -> bool:
        # some conditions here to include or reject datapoint
        return True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        dp = self.data[idx]
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

    def _preprocess_data_point(self, dp: Dict, **kwargs) -> Dict:
        # dp["text"] = self.prompter.materialize(runtime_input=dp["text"], **kwargs)
        return dp

    def compute_dataset_statistics(self, **params):
        raise NotImplementedError
