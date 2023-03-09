from __future__ import annotations

import re

from tensor_theorem_prover import Constant, Predicate, cosine_similarity, max_similarity
from tensor_theorem_prover.similarity import SimilarityFunc

from amr_reasoner.AmrMergeGenerator import MERGED_TOKENS_SUFFIX

STRIP_FRAME_NUMBER_RE = re.compile(r"-\d+$")


_symbols_sans_frame_numbers: dict[str, str] = {}


def strip_frame_number(symbol: str) -> str:
    if symbol not in _symbols_sans_frame_numbers:
        stripped_symbol = re.sub(STRIP_FRAME_NUMBER_RE, "", symbol)
        _symbols_sans_frame_numbers[symbol] = stripped_symbol
    return _symbols_sans_frame_numbers[symbol]


def adjusted_symbol_compare(
    item1: Constant | Predicate, item2: Constant | Predicate
) -> float:
    if item1.symbol == item2.symbol and MERGED_TOKENS_SUFFIX not in item1.symbol:
        return 1.0
    return 0.0


def amr_similarity_func(
    partial_symbol_similarity: float = 0.6,
) -> SimilarityFunc:
    def partial_symbol_compare(
        item1: Constant | Predicate, item2: Constant | Predicate
    ) -> float:
        if MERGED_TOKENS_SUFFIX not in item1.symbol and strip_frame_number(
            item1.symbol
        ) == strip_frame_number(item2.symbol):
            return partial_symbol_similarity
        return 0.0

    return max_similarity(
        [cosine_similarity, adjusted_symbol_compare, partial_symbol_compare]
    )
