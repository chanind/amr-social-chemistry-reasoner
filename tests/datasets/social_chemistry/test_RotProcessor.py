from __future__ import annotations

import penman
import tensor_theorem_prover as ttp

from amr_reasoner.datasets.social_chemistry.RotProcessor import (
    RotProcessor,
    prune_amr_tree,
)


def test_RotProcessor() -> None:
    processor = RotProcessor()
    [logic] = processor.rots_to_logic_bulk(
        [
            """
        # ::tok It 's bad to say things that are racist .
        (b / bad-07~2
            :ARG1 (s / say-01~4
                :ARG1 (t / thing~5
                    :mod (r / racist~8))))
        """
        ]
    )
    assert type(logic) == ttp.Implies
    assert (
        str(logic)
        == "(say-01(S) ∧ :ARG1(S,T) ∧ thing(T) ∧ :mod(T,R) ∧ racist(R)) → bad(S)"
    )


def test_negated_rots() -> None:
    processor = RotProcessor()
    [logic] = processor.rots_to_logic_bulk(
        [
            """
        # ::tok You should n't make jokes about divorce .
        (r / recommend-01~1
            :ARG1 (j / joke-01~4
                :ARG0 (y / you~0)
                :ARG2 (d / divorce-01~6))
            :polarity -~2)
        """
        ]
    )
    # assert type(logic) == ttp.Implies
    assert (
        str(logic)
        == "(joke-01(J) ∧ :ARG0(J,Y) ∧ you(Y) ∧ :ARG2(J,D) ∧ divorce-01(D)) → ¬good(J)"
    )


def test_prune_amr_tree_ignores_sane_trees() -> None:
    amr_tree = penman.parse(
        """
    # ::tok It 's okay to be uncomfortable around mentally ill people .
    (o / okay-04~2
        :ARG1 (c / comfortable-02~5
            :ARG0 (a / around~6
                :op1 (p / person~9
                    :ARG1-of (i / ill-01~8
                        :mod (m / mental~7))))
            :polarity -~10)
        :domain (i2 / it~0))
    """
    )
    pruned_tree = prune_amr_tree(amr_tree)
    assert amr_tree == pruned_tree


def test_prune_amr_tree_removes_rel() -> None:
    amr_tree = penman.parse(
        """
    # ::tok It 's ok to avoid sick people in order to prevent catching their illness .
    (o / okay-04~2
        :ARG1 (a / avoid-01~4
            :purpose (p2 / prevent-01~10
                :ARG1 (c / catch-02~11
                    :ARG2 (p / person~6
                        :ARG1-of (s / sick-05~5))))
            :ARG1 p)
        :rel (i2 / ill-01~13)
        :rel (i / Directory~13)
        :rel (p3 / publication~14))
    """
    )
    fixed_amr_tree = penman.parse(
        """
    # ::tok It 's ok to avoid sick people in order to prevent catching their illness .
    (o / okay-04~2
        :ARG1 (a / avoid-01~4
            :purpose (p2 / prevent-01~10
                :ARG1 (c / catch-02~11
                    :ARG2 (p / person~6
                        :ARG1-of (s / sick-05~5))))
            :ARG1 p))
    """
    )
    pruned_tree = prune_amr_tree(amr_tree)
    assert amr_tree != pruned_tree
    assert fixed_amr_tree == pruned_tree


def test_prune_amr_tree_leave_rel_if_nothing_else_is_there() -> None:
    amr_tree = penman.parse(
        """
    # ::tok It is ok to be annoyed with some of your classes .
    (o / okay-04~2
        :rel (a / annoy-01~5
            :ARG1 (d / drawer~5
                :quant (s / some~7)
                :ARG1-of (i / include-91~8
                    :ARG2 (c / class~10
                        :poss (y / you~9))))))
    """
    )
    pruned_tree = prune_amr_tree(amr_tree)
    assert amr_tree == pruned_tree
