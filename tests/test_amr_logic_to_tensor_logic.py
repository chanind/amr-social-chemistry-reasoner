from __future__ import annotations

from typing import cast

import numpy as np
import penman
from amr_logic_converter import AmrLogicConverter
from tensor_theorem_prover import And, Atom

from amr_reasoner.amr_logic_to_tensor_logic import amr_logic_to_tensor_logic
from amr_reasoner.AmrMergeGenerator import MERGED_TOKENS_SUFFIX, AmrMergeGenerator

converter = AmrLogicConverter()


def test_amr_logic_to_tensor_logic_handles_multiple_annotation_indices() -> None:
    amr_str = """
        (b / bad-07~2
            :ARG1 (s / say-01~4
                :ARG1 (t / thing~5
                    :mod (r / racist~8))))
    """
    embeddings = [np.random.rand(5) for _ in range(10)]
    orig_tree = penman.parse(amr_str)
    alternate_trees = AmrMergeGenerator().generate_merged_amrs(orig_tree)
    orig_tree_logic = amr_logic_to_tensor_logic(
        converter.convert(orig_tree), embeddings
    )
    # the first alternate should be completely collapsed
    collapsed_logic = amr_logic_to_tensor_logic(
        converter.convert(alternate_trees[1]), embeddings
    )
    first_term_orig = cast(Atom, cast(And, orig_tree_logic).args[0])
    first_term_collapsed = cast(Atom, cast(And, collapsed_logic).args[0])
    assert (
        first_term_orig.predicate.symbol + MERGED_TOKENS_SUFFIX
        == first_term_collapsed.predicate.symbol
    )
    assert first_term_orig.predicate.embedding is not None
    assert first_term_collapsed.predicate.embedding is not None
    assert not np.array_equal(
        first_term_orig.predicate.embedding, first_term_collapsed.predicate.embedding
    )
