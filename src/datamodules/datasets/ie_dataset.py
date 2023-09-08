import json
import logging
import os
from collections import Counter
from typing import Dict, List, Tuple

from .prompt_dataset import PromptDataset
from src.ie.linearization import get_linearization_class
from src.ie.ordering import apply_ordering_heuristic_to_datapoint

log = logging.getLogger(__name__)


def _get_triplet_surface_form(triplet: Dict) -> Tuple[str, str, str]:
    return (
        triplet["subject"]["surfaceform"],
        triplet["predicate"]["surfaceform"],
        triplet["object"]["surfaceform"],
    )


def _get_triplets_surface_form(triplets: List[Dict]) -> List[Tuple[str, str, str]]:
    return [_get_triplet_surface_form(triplet) for triplet in triplets]


def get_triplets_surface_form(dp: Dict) -> List[Tuple[str, str, str]]:
    return _get_triplets_surface_form(dp["triplets"])


def add_target_linearization(dp:Dict, tokenizer, linearization_class):

    assert "target" not in dp, "trying to add linearization as target but there is already a target"

    target_dict = dp.get("target_dict", {})

    if linearization_class.identifier not in target_dict:
        triplets = get_triplets_surface_form(dp)
        target_text, target_ids = linearization_class.triplet_list_to_text(
            triplets, tokenizer
        )
        target_dict[linearization_class.identifier] = (target_text, target_ids)
        dp["target_dict"] = target_dict

    dp["target"] = target_dict[linearization_class.identifier][0]
    dp["target_ids"] = target_dict[linearization_class.identifier][1]

    return target_dict[linearization_class.identifier]


def _are_triplets_with_same_subjects_consecutive(triplets):
    past_subjects = set()
    curr_subject = None

    for triplet in triplets:
        if curr_subject is None:
            curr_subject = triplet[0]

        if triplet[0] != curr_subject:
            if triplet[0] in past_subjects:
                # the new subject has already been seen before
                return False
            past_subjects.add(triplet[0])
            curr_subject = triplet[0]

    return True


def are_triplets_with_same_subject_consecutive(data):
    for idx, dp in enumerate(data):
        triplets = get_triplets_surface_form(dp)
        if _are_triplets_with_same_subjects_consecutive(
                triplets
        ):
            continue
        log.info(
            "DP with:\n"
            "IDX:{}\n"
            "TEXT:{}\n"
            "Has triplets with non-consecutive subjects: {}".format(
                idx, dp["text"], triplets
            )
        )
        return False

    return True


def read_constrained_world(
    constrained_world_id=None,
    path_to_constrained_world_dir=None,
    constrained_worlds_dir=None,
):
    assert {constrained_world_id is None, path_to_constrained_world_dir is None} == {
        True,
        False,
    }, "Either specify a `constrained_world` or a path_to_constrained_world_dir, not both."

    if path_to_constrained_world_dir is None:
        path_to_constrained_world_dir = os.path.join(
            constrained_worlds_dir, constrained_world_id
        )

    with open(
        os.path.join(path_to_constrained_world_dir, "entities.json")
    ) as json_file:
        entities = set(json.load(json_file))

    with open(
        os.path.join(path_to_constrained_world_dir, "relations.json")
    ) as json_file:
        relations = set(json.load(json_file))

    return entities, relations


class IEDataset(PromptDataset):

    def __init__(self, constrained_world_id, linearization_class_id:str, linearization_class_id_for_filtering: str,
                 path_to_constrained_world_dir: str = None, constrained_worlds_dir: str = None, **params):
        """
        linearization_class_id: fully_expanded or subject_collapsed
        """
        super().__init__(**params)
        self.linearization_class = get_linearization_class(linearization_class_id)
        self.linearization_class_for_filtering = get_linearization_class(linearization_class_id_for_filtering)
        self.entities_to_keep, self.relations_to_keep = self._read_constrained_world(constrained_world_id=constrained_world_id,
             path_to_constrained_world_dir=path_to_constrained_world_dir,
             constrained_worlds_dir=constrained_worlds_dir)

    def load_data(self, path:str, **kwargs):
        data = super().load_data(path=path, **kwargs)
        return data

    def _after_loading_data(self, **params):
        super()._after_loading_data(**params)

        if params.get("verify_triplet_ordering", False):
            log.info("Verifying ordering of triplets...")
            are_triplets_with_same_subject_consecutive(self.data)

    def _preprocess_data_point(self, dp: Dict, **kwargs) -> Dict:
        if kwargs.get("apply_ordering_heuristic", False):
            # apply ordering heuristic
            apply_ordering_heuristic_to_datapoint(dp)

        # add linearized target to datapoint, this has to be done after applying the ordering heuristic
        add_target_linearization(dp, self.tokenizer, self.linearization_class)
        super()._preprocess_data_point(dp, **kwargs)
        return dp

    def _read_constrained_world(self, constrained_world_id=None, path_to_constrained_world_dir=None,
                                constrained_worlds_dir=None):

        # Read parameters
        world_id = constrained_world_id
        path_to_constrained_world_dir = path_to_constrained_world_dir
        constrained_worlds_dir = constrained_worlds_dir

        # Check if the required parameters are set
        if not world_id and not path_to_constrained_world_dir:
            return

        return read_constrained_world(
            constrained_world_id=world_id,
            path_to_constrained_world_dir=path_to_constrained_world_dir,
            constrained_worlds_dir=constrained_worlds_dir,
        )

    def _include_datapoint(self, dp: Dict) -> bool:
        """
        Only include datapoints that have all entities and relations in the constrained world.
        """
        if all(entity["uri"] in self.entities_to_keep for entity in dp["entities"]) \
                and \
            all(entity["uri"] in self.relations_to_keep for entity in dp["relations"]):
            return True
        self.num_filtered_datapoints_constrained += 1
        return False

    def compute_dataset_statistics(self, **kwargs) -> Dict:

        # collect all entities
        entity_sets = []
        for dp in self.data:
            entity_sets.append(
                set(
                    [triplet["subject"]["surfaceform"] for triplet in dp["triplets"]]
                    + [triplet["object"]["surfaceform"] for triplet in dp["triplets"]]
                )
            )
        flat_entity_sets = [
            entity for entity_set in entity_sets for entity in entity_set
        ]

        # collect all relations
        relation_sets = [
            set([triplet["predicate"]["surfaceform"] for triplet in dp["triplets"]])
            for dp in self.data
        ]
        flat_relation_sets = [rel for rel_set in relation_sets for rel in rel_set]

        ent_freq = Counter(
                flat_entity_sets
            )  # The number of datapoints in which the entity appears
        rel_freq = Counter(
                flat_relation_sets
            )  # The number of datapoints in which a relation occurs

        stats = {
            "num_datapoints": len(self.data),
            "num_triplets": sum([len(dp["triplets"]) for dp in self.data]),
            "num_unique_entities": len(set(flat_entity_sets)),
            "num_unique_relations": len(set(flat_relation_sets)),
            "rel_freq": rel_freq,
            "ent_freq": ent_freq,
        }

        # remove rel_freq and ent_freq from display, because they are too large
        display_stats = {key: value for key, value in stats.items() if key != "rel_freq" and key != "ent_freq"}

        log.info(f"Dataset statistics: {display_stats}")

        return stats

    def __getitem__(self, idx):
        dp = self.data[idx]

        # target_text, _ = add_target_linearization(
        #     dp, self.tokenizer, self.linearization_class
        # )



        return {
            "id": dp["id"],
            "text": dp["text"],
            "target": dp["target"],
            "target_ids": dp["target_ids"],
        }
