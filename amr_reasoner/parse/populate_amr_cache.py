from __future__ import annotations
import math

from tqdm import tqdm
from typing import Iterable

from amr_reasoner.parse.AmrParseCache import AmrParseCache
from amr_reasoner.parse.AmrParser import AmrParser
from amr_reasoner.util import chunk_list


def populate_amr_cache(
    sentences: Iterable[str],
    parser: AmrParser,
    cache: AmrParseCache,
    batch_size: int = 128,
    verbose: bool = False,
    # setting this to None means never save the cache
    save_cache_every_n_batches: int | None = 10,
) -> None:
    sentences_to_parse: set[str] = set()
    for sentence in tqdm(sentences):
        if not cache.get(sentence):
            sentences_to_parse.add(sentence)

    for batch_num, batch in enumerate(
        tqdm(
            chunk_list(list(sentences_to_parse), batch_size),
            total=math.ceil(len(sentences_to_parse) / batch_size),
        )
    ):
        amr_annotations = parser.generate_amr_annotations(batch)
        for sentence, amr_annotation in zip(batch, amr_annotations):
            cache.set(sentence, amr_annotation)
        if verbose:
            print(f"Text: {batch[0]}")
            print(f"AMR: {amr_annotations[0]}")
        if (
            batch_num > 0
            and save_cache_every_n_batches is not None
            and batch_num % save_cache_every_n_batches == 0
        ):
            if verbose:
                print("Saving cache...")
            cache.save()
