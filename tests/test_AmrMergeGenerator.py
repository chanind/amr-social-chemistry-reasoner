from __future__ import annotations

import penman
from amr_logic_converter.extract_instances_from_amr_tree import (
    extract_instances_from_amr_tree,
)
from syrupy.assertion import SnapshotAssertion

from amr_reasoner.AmrMergeGenerator import (
    AmrMergeGenerator,
    _find_branches_after_depth,
    _find_child_instances,
    _find_sub_alignments,
    _map_parents,
)


def test_find_sub_alignments_with_basic_amr() -> None:
    amr_str = """
    # ::tok It 's bad to say things that are racist .
    (b / bad-07~2
        :ARG1 (s / say-01~4
            :ARG1 (t / thing~5
                :mod (r / racist~8))))
    """
    tree = penman.parse(amr_str)
    assert _find_sub_alignments(tree.node, None) == {2, 4, 5, 8}


def test_map_parents_with_basic_amr() -> None:
    amr_str = """
    # ::tok It 's reasonable for someone to call themselves an American if they 're from an American country .
    (r / reasonable-02~2
        :ARG1 (c / call-01~6
            :ARG0 (s / someone~4)
            :ARG1 s
            :ARG2 (p / person~7
                :mod (c2 / country~9
                    :name (n / name~9
                        :op1 "America"~10))))
        :condition (b / be-from-91~15
            :ARG1 s
            :ARG2 c2))
    """
    tree = penman.parse(amr_str)
    instances = extract_instances_from_amr_tree(tree)
    assert _map_parents(tree, instances) == {
        "s": {"c", "b"},
        "c": {"r"},
        "p": {"c"},
        "c2": {"p", "b"},
        "n": {"c2"},
        "b": {"r"},
    }


def test_find_sub_alignments_with_alignments_in_constants() -> None:
    amr_str = """
    # ::tok It 's reasonable for someone to call themselves an American if they 're from an American country .
    (r / reasonable-02~2
        :ARG1 (c / call-01~6
            :ARG0 (s / someone~4)
            :ARG1 s
            :ARG2 (p / person~7
                :mod (c2 / country~9
                    :name (n / name~9
                        :op1 "America"~10))))
        :condition (b / be-from-91~15
            :ARG1 s
            :ARG2 c2))
    """
    tree = penman.parse(amr_str)
    assert _find_sub_alignments(tree.node, None) == {2, 6, 4, 7, 9, 10, 15}
    # c node
    assert _find_sub_alignments(tree.node[1][1][1], None) == {6, 4, 7, 9, 10}


def test_find_sub_alignments_with_alignments_can_limit_by_depth() -> None:
    amr_str = """
    # ::tok It 's reasonable for someone to call themselves an American if they 're from an American country .
    (r / reasonable-02~2
        :ARG1 (c / call-01~6
            :ARG0 (s / someone~4)
            :ARG1 s
            :ARG2 (p / person~7
                :mod (c2 / country~9
                    :name (n / name~9
                        :op1 "America"~10))))
        :condition (b / be-from-91~15
            :ARG1 s
            :ARG2 c2))
    """
    tree = penman.parse(amr_str)
    assert _find_sub_alignments(tree.node, 1) == {2, 6, 15}
    assert _find_sub_alignments(tree.node, 2) == {2, 6, 4, 7, 15}
    # c node
    assert _find_sub_alignments(tree.node[1][1][1], 1) == {6, 4, 7}
    assert _find_sub_alignments(tree.node[1][1][1], 0) == {6}


def test_generate_merged_alternative_amrs_basic(snapshot: SnapshotAssertion) -> None:
    amr_str = """
    # ::tok It 's bad to say things that are racist .
    (b / bad-07~2
        :ARG1 (s / say-01~4
            :ARG1 (t / thing~5
                :mod (r / racist~8))))
    """
    tree = penman.parse(amr_str)
    generator = AmrMergeGenerator()
    # too lazy to write this out exlicitly...
    merged_trees = generator.generate_merged_amrs(tree)
    assert len(merged_trees) == 4
    assert merged_trees == snapshot


def test_generate_merged_alternative_amrs_cannot_merge_across_negation(
    snapshot: SnapshotAssertion,
) -> None:
    amr_str = """
    # ::tok It 's good to say things that are racist .
    (b / good-07~2
        :ARG1 (s / say-01~4
            :ARG1 (t / thing~5
                :mod (r / racist~8)
                :polarity -)))
    """
    tree = penman.parse(amr_str)
    generator = AmrMergeGenerator()
    # too lazy to write this out exlicitly...
    merged_trees = generator.generate_merged_amrs(tree)
    assert len(merged_trees) == 2
    assert merged_trees == snapshot


def test_generate_merged_alternative_amrs_cannot_merge_across_negation_with_internal_merges(
    snapshot: SnapshotAssertion,
) -> None:
    amr_str = """
    # ::tok It 's good to say things that are racist .
    (b / good-07~2
        :ARG1 (s / say-01~4
            :ARG1 (t / thing~5
                :mod (r / racist~8)
                :polarity -)))
    """
    tree = penman.parse(amr_str)
    generator = AmrMergeGenerator(max_internal_merge_depth=2)
    # too lazy to write this out exlicitly...
    merged_trees = generator.generate_merged_amrs(tree)
    assert len(merged_trees) == 4
    assert merged_trees == snapshot


def test_generate_merged_alternative_amrs_can_limit_merges_by_depth() -> None:
    amr_str = """
    # ::tok It 's bad to say things that are racist .
    (b / bad-07~2
        :ARG1 (s / say-01~4
            :ARG1 (t / thing~5
                :mod (r / racist~8))))
    """
    tree = penman.parse(amr_str)
    generator1 = AmrMergeGenerator(min_merge_depth=1)
    generator2 = AmrMergeGenerator(min_merge_depth=2)
    generator3 = AmrMergeGenerator(min_merge_depth=3)
    assert len(generator1.generate_merged_amrs(tree)) == 3
    assert len(generator2.generate_merged_amrs(tree)) == 2
    assert len(generator3.generate_merged_amrs(tree)) == 1


def test_generate_merged_alternative_amrs_with_long_chain_and_internal_merges(
    snapshot: SnapshotAssertion,
) -> None:
    amr_str = """
    # ::tok It 's bad to say things that are racist .
    (b / bad-07~2
        :ARG1 (s / say-01~4
            :ARG1 (t / thing~5
                :ARG0 (s2 / say-01~6
                    :ARG1 (t2 / thing~7
                        :ARG0 (s3 / say-01~8
                            :ARG1 (t3 / thing~9)))))))
    """
    tree = penman.parse(amr_str)
    generator = AmrMergeGenerator(max_internal_merge_depth=2)
    assert len(generator.generate_merged_amrs(tree)) == 52
    assert generator.generate_merged_amrs(tree) == snapshot


def test_generate_merged_alternative_amrs_generate_multiple_merges_in_the_same_output(
    snapshot: SnapshotAssertion,
) -> None:
    amr_str = """
    (b / blah~0
        :ARG0 (a / ablah~1
            :ARG1 (c / cblah~2))
        :ARG1 (s / sblah~3
            :ARG1 (t / tblah~4)))
    """
    tree = penman.parse(amr_str)
    generator = AmrMergeGenerator(min_merge_depth=1)
    merged_trees = generator.generate_merged_amrs(tree)
    assert len(merged_trees) == 4
    assert merged_trees == snapshot


def test_can_handle_weird_amr_parses() -> None:
    amr_str = """
    # ::tok It 's okay to be disappointed when your partner does n't enjoy the same things you do .
    (o / okay-04~2
        :ARG1 (d / disappoint-01~5
            :ARG1 (y / you~7))
        :rel (p2 / person~8
            :ARG0-of (h / have-rel-role-91~8
                :ARG1 y
                :ARG2 (p / partner~8)))
        :rel (e / enjoy-01~9
            :ARG0 (e2 / "enjoy-01></"~11
                :ARG1 (t / thing~12
                    :ARG1-of (r / rate-entity-91~9))
                :ARG1-of (s / same-01~13
                    :ARG2 (t2 / thing~14
                        :ARG1-of r))))
        :rel (v / vibrant~6)
        :rel -~10)
    """
    tree = penman.parse(amr_str)
    generator = AmrMergeGenerator(min_merge_depth=1)
    merged_trees = generator.generate_merged_amrs(tree)
    assert len(merged_trees) == 3


def test_generate_merged_alternative_amrs_with_complex_amr(
    snapshot: SnapshotAssertion,
) -> None:
    amr_str = """
    # ::tok It 's reasonable for someone to call themselves an American if they 're from an American country .
    (r / reasonable-02~2
        :ARG1 (c / call-01~6
            :ARG0 (s / someone~4)
            :ARG1 s
            :ARG2 (p / person~7
                :mod (c2 / country~9
                    :name (n / name~9
                        :op1 "America"~10))))
        :condition (b / be-from-91~15
            :ARG1 s
            :ARG2 c2))
    """
    tree = penman.parse(amr_str)
    generator = AmrMergeGenerator()
    # too lazy to write this out exlicitly...
    merged_trees = generator.generate_merged_amrs(tree)
    assert len(merged_trees) == 4
    assert merged_trees == snapshot


def test_generate_merged_alternative_amrs_skips_coreferenced_nodes_by_default() -> None:
    amr_str = """
    (b1 / blah~0
        :ARG0 (a / ablah~1
            :ARG1 (c / cblah~2))
        :ARG1 c
        :ARG2 (s / sblah~3
            :ARG1 (t / tblah~4
                :ARG0 c)))
    """
    tree = penman.parse(amr_str)
    generator = AmrMergeGenerator()
    merged_trees = generator.generate_merged_amrs(tree)
    assert len(merged_trees) == 2


def test_generate_merged_alternative_amrs_can_merge_coreferences_if_specified(
    snapshot: SnapshotAssertion,
) -> None:
    amr_str = """
    (b1 / blah~0
        :ARG0 (a / ablah~1
            :ARG1 (c / cblah~2))
        :ARG1 c
        :ARG2 (s / sblah~3
            :ARG1 (t / tblah~4
                :ARG0 c)))
    """
    tree = penman.parse(amr_str)
    generator = AmrMergeGenerator(allow_collapsing_coreferences=True)
    merged_trees = generator.generate_merged_amrs(tree)
    assert len(merged_trees) == 5
    assert merged_trees == snapshot


def test_generate_merged_alternative_amrs_with_complex_amr_and_internal_merge(
    snapshot: SnapshotAssertion,
) -> None:
    amr_str = """
    # ::tok It 's reasonable for someone to call themselves an American if they 're from an American country .
    (r / reasonable-02~2
        :ARG1 (c / call-01~6
            :ARG0 (s / someone~4)
            :ARG1 s
            :ARG2 (p / person~7
                :mod (c2 / country~9
                    :name (n / name~9
                        :op1 "America"~10))))
        :condition (b / be-from-91~15
            :ARG1 s
            :ARG2 c2))
    """
    tree = penman.parse(amr_str)
    generator = AmrMergeGenerator(max_internal_merge_depth=1)
    # too lazy to write this out exlicitly...
    merged_trees = generator.generate_merged_amrs(tree)
    assert len(merged_trees) == 9
    assert merged_trees == snapshot


def test_generate_merged_alternative_amrs_restricting_max_collapsed_per_node(
    snapshot: SnapshotAssertion,
) -> None:
    amr_str = """
    # ::tok It 's reasonable for someone to call themselves an American if they 're from an American country .
    (r / reasonable-02~2
        :ARG1 (c / call-01~6
            :ARG0 (s / someone~4)
            :ARG1 s
            :ARG2 (p / person~7
                :mod (c2 / country~9
                    :name (n / name~9
                        :op1 "America"~10))))
        :condition (b / be-from-91~15
            :ARG1 s
            :ARG2 c2))
    """
    tree = penman.parse(amr_str)
    generator = AmrMergeGenerator(max_collapsed_per_node=2)
    # too lazy to write this out exlicitly...
    merged_trees = generator.generate_merged_amrs(tree)
    assert len(merged_trees) == 3
    assert merged_trees == snapshot


def test_generate_alternatives_doesnt_delete_parts_of_the_graph(
    snapshot: SnapshotAssertion,
) -> None:
    amr_str = """
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
    tree = penman.parse(amr_str)
    generator = AmrMergeGenerator(max_collapsed_per_node=5, min_merge_depth=1)
    merged_trees = generator.generate_merged_amrs(tree)
    assert merged_trees == snapshot
    assert len(merged_trees) == 4


def test_find_child_instances() -> None:
    amr_str = """
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
    tree = penman.parse(amr_str)
    instances = extract_instances_from_amr_tree(tree)
    # no max depth find everything
    assert _find_child_instances(tree.node, 0, None, instances) == {
        "c",
        "a",
        "p",
        "i",
        "m",
        "i2",
    }
    assert _find_child_instances(tree.node, 0, 2, instances) == {"c", "i2", "a"}
    assert _find_child_instances(tree.node, 0, 1, instances) == {"c", "i2"}
    assert _find_child_instances(tree.node, 0, 0, instances) == set()


def test_find_branches_after_depth() -> None:
    amr_str = """
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
    tree = penman.parse(amr_str)
    assert _find_branches_after_depth(tree.node, 4) == [
        (":mod", ("m", [("/", "mental~7")]))
    ]
    assert _find_branches_after_depth(tree.node, 0) == tree.node[1][1:]
