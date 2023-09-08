from abc import ABC
from typing import Optional, Callable, Iterable, List

import torch


class HFConstrained(ABC):
    def get_prefix_allowed_tokens_fn(
        self, **batch_info: Optional[dict]
    ) -> Callable[[int, torch.Tensor], Iterable[int]]:
        raise NotImplementedError()


class ToyHFConstrained(ABC):

    def __init__(self, allowed_tokens: List[int]):
        self.allowed_tokens = allowed_tokens
        self.prefix_allowed_tokens_fn = lambda **kwargs: self.allowed_tokens

    def get_prefix_allowed_tokens_fn(
        self, **batch_info: Optional[dict]
    ) -> Callable[[int, torch.Tensor], Iterable[int]]:
        return self.prefix_allowed_tokens_fn
