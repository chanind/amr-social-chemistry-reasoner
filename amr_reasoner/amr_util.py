from __future__ import annotations

from penman.tree import Node, Tree


def count_nodes_in_amr_tree(tree: Tree) -> int:
    return _count_nodes_in_amr_tree_inner(tree.node)


def _count_nodes_in_amr_tree_inner(node: Node) -> int:
    num_nodes = 1
    _instance, instance_info = node
    _predicate, *edges = instance_info
    for _role, target in edges:
        if isinstance(target, tuple):
            num_nodes += _count_nodes_in_amr_tree_inner(target)
        else:
            # if the child isn't a node, treat this depth as 2 (cur node + child)
            num_nodes += 1
    return num_nodes


def find_amr_tree_depth(tree: Tree) -> int:
    return _find_amr_tree_depth_inner(tree.node)


def _find_amr_tree_depth_inner(node: Node) -> int:
    _instance, instance_info = node
    _predicate, *edges = instance_info
    # if there's only the current node, depth is 1
    if len(edges) == 0:
        return 1
    subdepths = []
    for _role, target in edges:
        if isinstance(target, tuple):
            subdepths.append(1 + _find_amr_tree_depth_inner(target))
        else:
            # if the child isn't a node, treat this depth as 2 (cur node + child)
            subdepths.append(2)
    return max(subdepths)
