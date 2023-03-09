from __future__ import annotations

import penman

from amr_reasoner.amr_util import count_nodes_in_amr_tree, find_amr_tree_depth


def test_find_amr_tree_depth_basic() -> None:
    amr_str = """
        (b / bad-07~2
            :ARG1 (s / say-01~4
                :ARG1 (t / thing~5
                    :mod (r / racist~8))))
    """
    tree = penman.parse(amr_str)
    assert find_amr_tree_depth(tree) == 4


def test_find_amr_tree_depth_complex() -> None:
    amr_str = """
        (h / have-degree-91 :polarity -
            :ARG1 (h2 / he)
            :ARG2 (t / tall)
            :ARG3 (e / enough)
            :ARG6 (r / ride-01
                    :ARG0 h2
                    :ARG1 (r2 / rollercoaster)))
    """
    tree = penman.parse(amr_str)
    assert find_amr_tree_depth(tree) == 3


def test_count_nodes_in_amr_tree_basic() -> None:
    amr_str = """
        (b / bad-07~2
            :ARG1 (s / say-01~4
                :ARG1 (t / thing~5
                    :mod (r / racist~8))))
    """
    tree = penman.parse(amr_str)
    assert count_nodes_in_amr_tree(tree) == 4


def test_count_nodes_in_amr_tree_complex() -> None:
    amr_str = """
        (h / have-degree-91 :polarity -
            :ARG1 (h2 / he)
            :ARG2 (t / tall)
            :ARG3 (e / enough)
            :ARG6 (r / ride-01
                    :ARG0 h2
                    :ARG1 (r2 / rollercoaster)))
    """
    tree = penman.parse(amr_str)
    assert count_nodes_in_amr_tree(tree) == 8
