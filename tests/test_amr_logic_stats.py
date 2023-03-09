from __future__ import annotations

import penman
from amr_logic_converter import AmrLogicConverter

from amr_reasoner.amr_logic_stats import count_amr_logic_terms
from amr_reasoner.datasets.social_chemistry.RotProcessor import RotProcessor


def test_count_amr_logic_terms_basic() -> None:
    amr_str = """
        (b / bad-07~2
            :ARG1 (s / say-01~4
                :ARG1 (t / thing~5
                    :mod (r / racist~8))))
    """
    tree = penman.parse(amr_str)
    converter = AmrLogicConverter()
    logic = converter.convert(tree)
    assert count_amr_logic_terms(logic) == 7


def test_count_amr_logic_terms_with_implication_and_negation() -> None:
    amr_str = """
        # ::tok You should n't make jokes about divorce .
        (r / recommend-01~1
            :ARG1 (j / joke-01~4
                :ARG0 (y / you~0)
                :ARG2 (d / divorce-01~6))
            :polarity -~2)
        """
    tree = penman.parse(amr_str)
    # use the RotProcessor to convert the logic, since it will generate an implication
    processor = RotProcessor()
    converter = processor.rot_logic_converter
    logic = converter.convert(tree)
    assert (
        str(logic)
        == "(joke-01(J) ∧ :ARG0(J, Y) ∧ you(Y) ∧ :ARG2(J, D) ∧ divorce-01(D)) → ¬good(J)"
    )
    assert count_amr_logic_terms(logic) == 6
