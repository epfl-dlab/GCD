import copy
import os.path
from typing import Optional, Callable, Iterable, List
from functools import partial
import torch

from src.constrained_generation import HFConstrained
from src.constrained_generation.gf_runtime import GFServerRuntimeForLM


def get_prefix_allowed_tokens_from_gf(
    gf_runtime, batch_id, sent_tokens, non_language_token_lengths, pgfs
):
    """

    Args:
        gf_runtime:
        batch_id:  index of the dp in the batch
        sent_tokens:
        non_language_token_lengths:
        pgfs:

    Returns:

    """
    # global non_language_token_len
    sent_tokens = (
        sent_tokens.squeeze(0) if isinstance(sent_tokens, torch.Tensor) else sent_tokens
    )
    non_language_token_len = non_language_token_lengths[batch_id]
    partial_language_tokens = sent_tokens[non_language_token_len:]
    pgf: str = pgfs[batch_id]
    return gf_runtime.get_prefix_allowed_tokens_for_LM(partial_language_tokens, pgf=pgf)


class GF_Constrained(HFConstrained):
    """
    A constraint module instantiated with a number of grammar files.
    """

    _dp2pgf: dict[
        int, str
    ] = None  # mapping from dp_id to pgf file name, useful for input dependent grammar selection

    def __init__(
        self,
        default_grammar: str,
        grammar_module: str,
        grammar_dir: str = None,
        dp2pgf: dict[int, str] = None,
        name:str= None,
    ):
        super().__init__()
        self.name = name # the name will be used to identify the constraint module in the experiment run name
        self.grammar_dir = (
            os.path.join(grammar_dir, grammar_module)
            if grammar_dir is not None
            else grammar_module
        )
        self.gf_runtime_for_lm = GFServerRuntimeForLM(
            default_pgf=default_grammar, grammar_dir=self.grammar_dir
        )
        self.set_dp2pgf_lookup(dp2pgf=dp2pgf)
        self._default_pgf = default_grammar

    def set_dp2pgf_lookup(self, dp2pgf: dict = None):
        if dp2pgf is None:
            try:
                self._dp2pgf = self._infer_dp2pgf()
            except:
                raise RuntimeError(
                    "dp2pgf is not provided and cannot be inferred from the grammar_dir"
                )
        else:
            self._dp2pgf = dp2pgf

    def _infer_dp2pgf(self) -> dict[int, str]:
        # list all pgf files in the grammar_dir
        pgf_files = [f for f in os.listdir(self.grammar_dir) if f.endswith(".pgf")]
        # infer dp2pgf
        dp2pgf = {}
        for pgf_file in pgf_files:
            pgf_name = pgf_file.split(".")[0]
            dp_idx = int(pgf_name.split("_")[-1])
            dp2pgf[dp_idx] = pgf_file
        return dp2pgf

    def get_pgf_for_dp(self, dp_id: int) -> str:
        if dp_id not in self._dp2pgf:
            return self._default_pgf
        else:
            return self._dp2pgf[dp_id]

    def get_prefix_allowed_tokens_fn(
        self, **batch_info: Optional[dict]
    ) -> Callable[[int, torch.Tensor], Iterable[int]]:
        prompt_token_ids = batch_info["batch"][
            "input_ids"
        ]  # this is the format for huggingface tokenizer
        dp_ids: List[int] = batch_info["batch"].get("id", None)
        pgfs: List[str] = [self.get_pgf_for_dp(dp_id) for dp_id in dp_ids]
        prefix_allowed_tokens_fn = self.__get_prefix_allowed_tokens_fn(
            prompt_token_ids=prompt_token_ids,
            pgfs=pgfs,  # assume here that dp_ids are pgfs
        )

        return prefix_allowed_tokens_fn

    def __get_prefix_allowed_tokens_fn(
        self, prompt_token_ids, pgfs: List[str]
    ) -> Callable[[int, torch.Tensor], Iterable[int]]:
        # input_token_ids and prompt_token_ids are tensors of shape (1, seq_len) or (seq_len,)

        non_language_token_lengths: List[int] = [
            prompt_token_ids.shape[1] for _ in range(len(prompt_token_ids))
        ]

        def wrapped_get_prefix_allowed_tokens_fn(
            batch_id: int, sent_tokens: torch.Tensor
        ) -> Iterable[int]:
            return get_prefix_allowed_tokens_from_gf(
                self.gf_runtime_for_lm,
                batch_id,
                sent_tokens,
                non_language_token_lengths,
                pgfs,
            )

        return wrapped_get_prefix_allowed_tokens_fn


class GF_ConstrainedWithRenaming(GF_Constrained):
    def __init__(
        self,
        default_grammar: str,
        grammar_module: str,
        grammar_dir: str = None,
        dp2pgf: dict[int, str] = None,
        name:str= None,
    ):
        super().__init__(
            default_grammar=default_grammar,
            grammar_module=grammar_module,
            grammar_dir=grammar_dir,
            dp2pgf=dp2pgf,
            name=name,
        )

    def get_prefix_allowed_tokens_fn(
        self, **batch_info: Optional[dict]
    ) -> Callable[[int, torch.Tensor], Iterable[int]]:
        try:
            # prompt_token_ids = batch_info["batch"]["input_ids"] # this is the format for huggingface tokenizer
            return super().get_prefix_allowed_tokens_fn(**batch_info)
        except KeyError:
            renamed_batch_info = copy.deepcopy(batch_info)
            renamed_batch_info["batch"]["input_ids"] = batch_info["batch"][
                "src_input_ids"
            ]
            renamed_batch_info["batch"]["attention_mask"] = batch_info["batch"][
                "src_attention_mask"
            ]
            return super().get_prefix_allowed_tokens_fn(**renamed_batch_info)
