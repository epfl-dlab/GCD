from typing import List

from src import utils

log = utils.get_only_rank_zero_logger(__name__, stdout=True)
import logging
log.setLevel(logging.INFO)


def unflatten_generations(generations: List[str], bs: int=None, num_beams: int=None) -> List[List[str]]:
    """Split a list of generations into batches of size bs

    c.f. https://huggingface.co/docs/transformers/v4.19.2/en/internal/generation_utils#transformers.generation_utils.BeamSearchDecoderOnlyOutput

    Huggingface's Transformers library uses a flattened shape for the outputs of beam search.
    This function generates a flattened list of generations.
    By flattening it into a single dimension, it simplifies the logic and handling in certain parts of the codebase.

    If you want to match outputs to their corresponding inputs or if you're handling post-processing, having the outputs
    in the shape (batch_input_size,num_beam,max_seq_len) can be more intuitive.

    """
    if bs is None and num_beams is None:
        raise ValueError("Either bs or num_beams must be provided")

    if bs is not None and num_beams is not None:
        assert len(generations) == bs * num_beams, f"len(generations)={len(generations)} but bs={bs} and num_beams={num_beams}"

    if num_beams is None:
        num_beams = len(generations) // bs
    return [generations[i:i + num_beams] for i in range(0, len(generations), num_beams)]


def get_first_no_empty_generation(predictions: List[List[str]]) -> List[str]:
    """
    `(text for text in texts if text)` returns a generator that yields all non-empty strings in texts.
    `next(iterator, default)` returns the next item from the iterator or the default value if the iterator is exhausted.
    """
    results: List[str] = []
    for texts in predictions:
        first_non_empty_text = next((text for text in texts if text), "")
        results.append(first_non_empty_text)
        if not first_non_empty_text:
            log.warning(f"All generations are empty for {texts}")
    return results
