from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

from amr_logic_converter.extract_instances_from_amr_tree import (
    extract_instances_from_amr_tree,
)
from amr_logic_converter.map_instances_to_nodes import map_instances_to_nodes
from penman.surface import Alignment
from penman.tree import Branch, Node, Tree

from amr_reasoner.amr_util import count_nodes_in_amr_tree

MERGED_TOKENS_SUFFIX = "MERGED"


@dataclass
class MergeContext:
    instances: frozenset[str]
    instance_parents_map: dict[str, set[str]]
    instance_nodes_map: dict[str, Node]
    tree: Tree


class AmrMergeGenerator:
    min_merge_depth: int
    max_collapsed_per_node: Optional[int]
    max_internal_merge_depth: Optional[int]
    allow_collapsing_coreferences: bool

    def __init__(
        self,
        min_merge_depth: int = 0,
        max_collapsed_per_node: Optional[int] = None,
        max_internal_merge_depth: Optional[int] = None,
        allow_collapsing_coreferences: bool = False,
    ) -> None:
        self.min_merge_depth = min_merge_depth
        self.max_collapsed_per_node = max_collapsed_per_node
        self.max_internal_merge_depth = max_internal_merge_depth
        self.allow_collapsing_coreferences = allow_collapsing_coreferences

    def generate_merged_amrs(self, tree: Tree) -> list[Tree]:
        """
        Generate alternative AMRs for the given AMR by merging nodes.
        """
        alternatives = [tree]
        # keep trying to generate new alternatives until we can't generate any more
        while True:
            num_new_alternatives = 0
            for alternative in alternatives:
                instances = extract_instances_from_amr_tree(alternative)
                ctx = MergeContext(
                    instances=instances,
                    instance_parents_map=_map_parents(alternative, instances),
                    instance_nodes_map=map_instances_to_nodes(alternative, instances),
                    tree=alternative,
                )
                generated_alternatives = self._generate_merged_alternative_amrs(
                    alternative,
                    alternative.node,
                    ctx,
                    0,
                )
                for generated_alternative in generated_alternatives:
                    if generated_alternative not in alternatives:
                        alternatives.append(generated_alternative)
                        num_new_alternatives += 1
            if num_new_alternatives == 0:
                break
        return alternatives

    def calculate_collapsability(self, tree: Tree) -> float:
        """
        Calculate what percentage of the tree can be collapsed following the rules of this generator.
        """
        merges = self.generate_merged_amrs(tree)
        merges.sort(key=count_nodes_in_amr_tree)

        unmerged_nodes = count_nodes_in_amr_tree(merges[-1])
        max_merged_nodes = count_nodes_in_amr_tree(merges[0])

        # just making this explicit to always run exactly 0 rather than a floating point number that's super close to 0
        if unmerged_nodes == max_merged_nodes:
            return 0

        collapsability = 1 - (max_merged_nodes - 1) / (unmerged_nodes - 1)
        return collapsability

    def _generate_merged_alternative_amrs(
        self,
        original_tree: Tree,
        node: Node,
        ctx: MergeContext,
        depth: int,
    ) -> list[Tree]:
        """Generate alternative AMRs for the given AMR by merging nodes."""
        alternatives = [original_tree]
        collapse_depths: list[None | int] = [None]
        if self.max_internal_merge_depth is not None:
            collapse_depths += list(range(1, self.max_internal_merge_depth + 1))
        for collapse_depth in collapse_depths:
            if depth >= self.min_merge_depth and self._can_collapse_node(
                node, ctx, collapse_depth=collapse_depth
            ):
                collapse = _collapse_node(original_tree, node, collapse_depth, ctx)
                if collapse not in alternatives:
                    alternatives.append(collapse)
            _instance, instance_info = node
            _predicate, *edges = instance_info
            for role, target in edges:
                if isinstance(target, tuple):
                    child_alternatives = self._generate_merged_alternative_amrs(
                        original_tree,
                        target,
                        ctx,
                        depth + 1,
                    )
                    for child_alternative in child_alternatives:
                        if child_alternative not in alternatives:
                            alternatives.append(child_alternative)
        return alternatives

    def _can_collapse_node(
        self,
        node: Node,
        ctx: MergeContext,
        collapse_depth: Optional[int] = None,
    ) -> bool:
        """
        Whether the given node can be safely collapsed.
        E.g. does it not have contain elements corefenced at higher level in the tree?
        """
        if not _node_has_alignment(node):
            return False

        instances_to_merge = _find_child_instances(
            node, 0, collapse_depth, ctx.instances
        )
        instances_to_merge_excluding_referenced_instances = _find_child_instances(
            node, 0, collapse_depth, ctx.instances, True
        )
        instances_including_cur = {node[0], *instances_to_merge}

        if len(node[1]) <= 1:
            return False
        # don't collapse nodes that have negation
        _instance, instance_info = node
        _predicate, *edges = instance_info
        nodes_to_merge = [
            ctx.instance_nodes_map[instance] for instance in instances_to_merge
        ]
        node_to_merge_including_cur = [node, *nodes_to_merge]
        if _contains_negation(nodes_to_merge):
            return False
        if _contains_existing_merge(node_to_merge_including_cur):
            return False

        num_contained_alignments = len(_find_sub_alignments(node, collapse_depth))
        if (
            self.max_collapsed_per_node is not None
            and num_contained_alignments > self.max_collapsed_per_node
        ):
            return False

        for instance in instances_to_merge:
            parents = ctx.instance_parents_map[instance]
            for parent in parents:
                if parent not in instances_including_cur:
                    # if the instance is fully defined here and coref collapses are OK, we're allowed to collapse it
                    # and move the instance definition to one of the other coreferences in the tree
                    if (
                        self.allow_collapsing_coreferences
                        and instance
                        in instances_to_merge_excluding_referenced_instances
                    ):
                        continue
                    return False
        return True


def _collapse_node(
    original_tree: Tree,
    node: Node,
    collapse_depth: Optional[int],
    ctx: MergeContext,
) -> Tree:
    """Return a new Tree with the children of the given node all collapsed into the node itself."""
    collapsed_alignments = sorted(list(_find_sub_alignments(node, collapse_depth)))
    alignment_strs = [str(alignment) for alignment in collapsed_alignments]
    instance, instance_info = node
    slash, predicate = instance_info[0]
    base_predicate = strip_punctuation(predicate.split("~")[0])
    updated_predicate = (
        f"{base_predicate}{MERGED_TOKENS_SUFFIX}~{','.join(alignment_strs)}"
    )
    branches = _find_branches_after_depth(node, collapse_depth)
    updated_node: Node = (instance, [(slash, updated_predicate), *branches])
    if _contains_negation([node]):
        updated_node[1].append((":polarity", "-"))

    new_root_node = _copy_tree_and_replace_node(original_tree.node, node, updated_node)
    repopulated_root_node = _repopulate_empty_references(new_root_node, set(), ctx)
    return Tree(repopulated_root_node, original_tree.metadata)


def _copy_tree_and_replace_node(
    cur_tree_node: Node,
    node_to_replace: Node,
    replacement_node: Node,
) -> Node:
    """Return a new Node with the given node replaced with the given replacement node when encountered."""
    if cur_tree_node == node_to_replace:
        return replacement_node
    instance, instance_info = cur_tree_node
    predicate, *edges = instance_info
    updated_edges = []
    for role, target in edges:
        if isinstance(target, tuple):
            updated_edges.append(
                (
                    role,
                    _copy_tree_and_replace_node(
                        target, node_to_replace, replacement_node
                    ),
                )
            )
        else:
            updated_edges.append((role, target))
    return (instance, [predicate, *updated_edges])


def _repopulate_empty_references(
    cur_tree_node: Node,
    populated_instances: set[str],
    ctx: MergeContext,
) -> Node:
    """Populate empty references to nodes with the full original coreferenced node."""
    instance, instance_info = cur_tree_node
    populated_instances.add(instance)
    predicate, *edges = instance_info
    updated_edges = []
    for role, target in edges:
        if isinstance(target, tuple):
            updated_edges.append(
                (
                    role,
                    _repopulate_empty_references(target, populated_instances, ctx),
                )
            )
        elif target in ctx.instances and target not in populated_instances:
            updated_edges.append((role, ctx.instance_nodes_map[target]))
            populated_instances.add(target)
        else:
            updated_edges.append((role, target))
    return (instance, [predicate, *updated_edges])


def _contains_negation(nodes: list[Node]) -> bool:
    for node in nodes:
        _instance, instance_info = node
        _predicate, *edges = instance_info
        for role, target in edges:
            if role == ":polarity":
                return True
    return False


def _contains_existing_merge(nodes: list[Node]) -> bool:
    for node in nodes:
        _instance, instance_info = node
        predicate, *_edges = instance_info
        if MERGED_TOKENS_SUFFIX in predicate[1]:
            return True
    return False


def extract_instances_in_branch(
    node: Node, instances: frozenset[str]
) -> frozenset[str]:
    instances_in_branch = set()
    instance, instance_info = node
    if instance in instances:
        instances_in_branch.add(instance)
    for role, target in instance_info[1:]:
        if isinstance(target, tuple):
            instances_in_branch.update(extract_instances_in_branch(target, instances))
        elif target in instances:
            instances_in_branch.add(target)
    return frozenset(instances_in_branch)


def _extract_alignment(element: str) -> Alignment | None:
    """Parse an element and extract alignment info, if present"""
    # based on https://github.com/goodmami/penman/blob/f3b0c423a60f82b13fffeec73fa1a77bf75cd4dc/penman/layout.py#L211
    # this is a private method in penman, so copying it here in case the internal penman API changes
    # TODO: replace this by extracting this info directly from a penman graph
    alignment = None
    if "~" not in element:
        return None
    if element.startswith('"'):
        # need to handle alignments on strings differently
        # because strings may contain ~ inside the quotes (e.g., URIs)
        pivot = element.rindex('"') + 1
        if pivot < len(element):
            return Alignment.from_string(element[pivot:])
    else:
        alignment = element.partition("~")[2]
        return Alignment.from_string(alignment)
    return None


def _node_has_alignment(node: Node) -> bool:
    """
    Check if the given node has an alignment marker on it.
    """
    _instance, instance_info = node
    predicate, *edges = instance_info
    return _extract_alignment(predicate[1]) is not None


def _find_sub_alignments(node: Node, collapse_depth: Optional[int]) -> set[int]:
    """
    Find all alignments for this node + its children, up to max depth from here.
    """
    _instance, instance_info = node
    predicate, *edges = instance_info
    sub_alignments = set()
    alignment = _extract_alignment(predicate[1])
    if alignment:
        assert (
            len(alignment.indices) == 1
        ), "Only alignments with a single index are supported currently"
        sub_alignments.add(alignment.indices[0])
    if collapse_depth is not None and collapse_depth <= 0:
        return sub_alignments
    next_collapse_depth = None if collapse_depth is None else collapse_depth - 1
    for _role, target in edges:
        if isinstance(target, tuple):
            sub_alignments.update(_find_sub_alignments(target, next_collapse_depth))
        elif (
            isinstance(target, str)
            and (alignment := _extract_alignment(target)) is not None
        ):
            if alignment:
                assert (
                    len(alignment.indices) == 1
                ), "Only alignments with a single index are supported currently"
                sub_alignments.add(alignment.indices[0])
    return sub_alignments


def _map_parents(tree: Tree, instances: frozenset[str]) -> dict[str, set[str]]:
    """
    Map each instance to its parents in the tree.
    """
    parents: dict[str, set[str]] = defaultdict(set)

    def _map_parents_inner(node: Node, parent: Optional[str] = None) -> None:
        instance, instance_info = node
        if parent is not None:
            if instance in instances:
                parents[instance].add(parent)
        for _role, target in instance_info[1:]:
            if isinstance(target, tuple):
                _map_parents_inner(target, instance)
            elif target in instances:
                parents[target].add(instance)

    _map_parents_inner(tree.node)
    return parents


def _find_child_instances(
    node: Node,
    cur_depth: int,
    max_search_depth: Optional[int],
    instances: frozenset[str],
    exclude_referenced_instanced: bool = False,
) -> set[str]:
    """
    Find all instances that are children of the given node.
    Does not include the current node in results.
    """
    if max_search_depth is not None and cur_depth >= max_search_depth:
        return set()
    child_instances = set()
    instance, instance_info = node
    for _role, target in instance_info[1:]:
        if isinstance(target, tuple):
            if target[0] in instances:
                child_instances.add(target[0])
            child_instances.update(
                _find_child_instances(
                    target, cur_depth + 1, max_search_depth, instances
                )
            )
        elif target in instances and not exclude_referenced_instanced:
            child_instances.add(target)

    return child_instances


def _find_branches_after_depth(node: Node, depth: Optional[int]) -> list[Branch]:
    """
    Find all branches at the depth from the node
    """
    if depth is None:
        return []
    instance, instance_info = node
    if depth <= 0:
        return instance_info[1:]
    branches: list[Branch] = []
    for _role, target in instance_info[1:]:
        if isinstance(target, tuple):
            branches += _find_branches_after_depth(target, depth - 1)
    return branches


PUNCT_REGEX = re.compile(r"[\.,\?!\(\)\[\]\{\}\"\'\<\>\/\\]+")


def strip_punctuation(s: str) -> str:
    return PUNCT_REGEX.sub("", s)
