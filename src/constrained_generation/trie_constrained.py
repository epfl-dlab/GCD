from typing import Iterable, Optional, Callable

import numpy as np
import torch
import os

from src.constrained_generation.trie import Trie
from src.constrained_generation import HFConstrained

import src.constrained_generation.trie_constrained_generation_utils as utils
from src.utils import get_only_rank_zero_logger
from src.utils import get_linearization_class

log = get_only_rank_zero_logger(__name__)


class Trie_Constrained_IE(HFConstrained):
    def __init__(self, model, linearization_class_id, entity_trie, relation_trie):
        self.linearization_class_id = linearization_class_id
        self.prefix_allowed_tokens_fn = _get_prefix_allowed_tokens_fn(
            model,
            entity_trie=entity_trie,
            relation_trie=relation_trie,
            linearization_class_id=linearization_class_id,
        )

    def get_prefix_allowed_tokens_fn(
        self, **batch_info: Optional[dict]
    ) -> Callable[[int, torch.Tensor], Iterable[int]]:
        return self.prefix_allowed_tokens_fn

    @classmethod
    def from_constrained_world(
        cls,
        model,
        linearization_class_id,
        path_to_relid2name_mapping,
        path_to_entid2name_mapping,
        constrained_world_id=None,
        path_to_constrained_world_dir=None,
        constrained_worlds_dir=None,
        path_to_trie_cache_dir=None,
        identifier=None,
        override=False,
    ):
        linearization_class = get_linearization_class(linearization_class_id)
        if identifier is not None:
            assert (
                path_to_trie_cache_dir is not None
            ), "Trie cache identifier is given but path_to_trie_cache_dir is None."

            path_to_trie_cache = os.path.join(path_to_trie_cache_dir, identifier)

            if os.path.isdir(path_to_trie_cache):
                if not override:
                    log.info("Loading cached tries from %s", path_to_trie_cache)
                    entity_trie = Trie.load(
                        os.path.join(path_to_trie_cache, f"entity_trie.pickle")
                    )
                    relation_trie = Trie.load(
                        os.path.join(path_to_trie_cache, "relation_trie.pickle")
                    )
                    return cls(
                        model, linearization_class_id, entity_trie, relation_trie
                    )
                log.info("Overriding cached tries from %s.", path_to_trie_cache)
        else:
            path_to_trie_cache = (
                None  # to prevent asserts failing in the helper functions
            )

        log.info("Building the tries now. This might take a few minutes...")

        entity_ids, relation_ids = utils.read_constrained_world(
            constrained_world_id=constrained_world_id,
            path_to_constrained_world_dir=path_to_constrained_world_dir,
            constrained_worlds_dir=constrained_worlds_dir,
        )

        log.info("Reading id to name mappings...")

        relation_names = utils.get_names_for_ids(
            relation_ids,
            path_to_relid2name_mapping,
            keep_spaces=linearization_class.keep_spaces_relations,
        )
        entity_names = utils.get_names_for_ids(
            entity_ids,
            path_to_entid2name_mapping,
            keep_spaces=linearization_class.keep_spaces_entities,
        )

        log.info(
            f"{len(entity_ids) - len(entity_names)} out of {len(entity_ids)} entities were not found in the mapping"
        )
        log.info(
            f"{len(relation_ids) - len(relation_names)} out of {len(relation_ids)} relations were not found in the mapping"
        )

        entity_trie = utils.get_trie_from_strings(
            entity_names,
            tokenizer=model.tokenizer,
            output_folder_path=path_to_trie_cache,
            trie_name="entity_trie",
        )

        relation_trie = utils.get_trie_from_strings(
            relation_names,
            tokenizer=model.tokenizer,
            output_folder_path=path_to_trie_cache,
            trie_name="relation_trie",
        )

        # Return a factory to avoid hydra in the model constructor?
        # lambda model: cls(model, entity_trie, relation_trie)
        return cls(model, linearization_class_id, entity_trie, relation_trie)

    @classmethod
    def from_string_iterables(
        cls,
        model,
        linearization_class_id,
        entity_names: Iterable[str],
        relation_names: Iterable[str],
    ):
        entity_trie = utils.get_trie_from_strings(
            entity_names, tokenizer=model.tokenizer
        )
        relation_trie = utils.get_trie_from_strings(
            relation_names, tokenizer=model.tokenizer
        )

        return cls(model, linearization_class_id, entity_trie, relation_trie)


def _get_prefix_allowed_tokens_fn(
    model, entity_trie, relation_trie, linearization_class_id
):
    EOS_TOKEN = model.tokenizer.eos_token_id
    linearization_class = get_linearization_class(linearization_class_id)

    state_id2token_ids = {
        "sub_id": np.array(
            utils.encode(
                linearization_class.subject_id, model.tokenizer, keep_eos=False
            )
        ),
        "rel_id": np.array(
            utils.encode(
                linearization_class.relation_id, model.tokenizer, keep_eos=False
            )
        ),
        "obj_id": np.array(
            utils.encode(linearization_class.object_id, model.tokenizer, keep_eos=False)
        ),
        "et_id": np.array(
            utils.encode(linearization_class.et_id, model.tokenizer, keep_eos=False)
        ),
    }

    if linearization_class_id == "subject_collapsed":
        state_id2next_state_ids = {
            "sub_id": ["rel_id"],
            "rel_id": ["obj_id"],
            "obj_id": ["rel_id", "et_id"],
            "et_id": ["sub_id"],
        }
    else:
        state_id2next_state_ids = {
            "sub_id": ["rel_id"],
            "rel_id": ["obj_id"],
            "obj_id": ["et_id"],
            "et_id": ["sub_id"],
        }

    state_id2next_states_ids = {
        state_id: [
            state_id2token_ids[next_state_id] for next_state_id in next_state_ids
        ]
        for state_id, next_state_ids in state_id2next_state_ids.items()
    }

    state_id2next_states_first_ids = {
        state_id: list(set([next_state_ids[0] for next_state_ids in next_states_ids]))
        for state_id, next_states_ids in state_id2next_states_ids.items()
    }

    def _get_next_states_ids(state_id):
        return state_id2next_states_ids[state_id]

    def _get_next_states_first_ids(state_id):
        return state_id2next_states_first_ids[state_id]

    def _get_allowed_tokens_from_trie(suffix, trie, current_state_id):
        allowed_tokens = trie.get(suffix)

        if EOS_TOKEN in allowed_tokens:
            allowed_tokens.remove(EOS_TOKEN)
            allowed_tokens.extend(_get_next_states_first_ids(current_state_id))

        return allowed_tokens

    def _get_allowed_tokens_for_generating_state_identifier(suffix, next_state_id):
        while next_state_id.size > 1:
            window = next_state_id[:-1]

            if suffix.size < window.size:
                next_state_id = window
                continue

            if np.array_equal(window, suffix[-len(window) :]):
                return [next_state_id[-1]]

            next_state_id = window

        return []

    def get_allowed_tokens(state_id, suffix):
        next_states_ids = _get_next_states_ids(state_id)

        allowed_tokens = set()
        for next_state_id in next_states_ids:
            allowed_tokens = allowed_tokens.union(
                _get_allowed_tokens_for_generating_state_identifier(
                    suffix, next_state_id
                )
            )

        if len(allowed_tokens) > 0:
            return list(allowed_tokens)

        # ~~~ otherwise ~~~
        if state_id == "et_id":
            allowed_tokens = [EOS_TOKEN]
            allowed_tokens.extend(_get_next_states_first_ids(state_id))

            return allowed_tokens

        elif state_id == "rel_id":
            return _get_allowed_tokens_from_trie(suffix, relation_trie, state_id)

        return _get_allowed_tokens_from_trie(suffix, entity_trie, state_id)

    def _get_state_id_and_suffix_start(sent_ids):
        last_token_idx_plus_one = len(sent_ids)

        while last_token_idx_plus_one > 0:
            for state_id, pattern in state_id2token_ids.items():
                pat_size = pattern.size

                if last_token_idx_plus_one < pat_size:
                    continue

                window = sent_ids[
                    last_token_idx_plus_one - pat_size : last_token_idx_plus_one
                ]
                if np.array_equal(window, pattern):
                    return state_id, last_token_idx_plus_one

            last_token_idx_plus_one -= 1

        return "et_id", 0

    def prefix_allowed_tokens_fn(
        batch_id: int, sent_ids: torch.Tensor
    ) -> Iterable[int]:
        sent_ids = sent_ids.cpu().numpy()

        # ToDo: Is this necessary? It was for genie for some weird reason that I didn't figure out
        if len(sent_ids) > 1 and sent_ids[-1] == EOS_TOKEN:
            return []

        state_id, suffix_start_idx = _get_state_id_and_suffix_start(sent_ids)
        allowed_tokens = get_allowed_tokens(state_id, sent_ids[suffix_start_idx:])
        return allowed_tokens

    return prefix_allowed_tokens_fn
