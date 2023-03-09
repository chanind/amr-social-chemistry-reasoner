from __future__ import annotations

from dataclasses import dataclass

import amr_logic_converter as alc
import numpy as np
import penman
from penman.tree import Tree

from amr_reasoner.amr_util import count_nodes_in_amr_tree, find_amr_tree_depth

from .AmrMergeGenerator import AmrMergeGenerator


@dataclass
class AmrLogicStats:
    """A class for storing statistics about an AMR graph and generated logic"""

    num_instance_nodes: int
    num_nodes: int
    num_logic_terms: int
    tree_depth: int
    num_possible_merges: int


@dataclass
class BulkStats:
    mean: float
    median: float
    stdev: float
    max: float
    min: float


@dataclass
class AmrLogicBulkStats:
    """A class for storing bulk statistics about AMR graphs and generated logic"""

    num_instance_nodes: BulkStats
    num_nodes: BulkStats
    num_logic_terms: BulkStats
    tree_depth: BulkStats
    num_possible_merges: BulkStats


class AmrLogicStatsCalculator:
    """A class for calculating bulk statistics about AMR graphs and generated logic"""

    def __init__(
        self,
        logic_converter: alc.AmrLogicConverter,
        min_merge_depth: int = 1,
        max_collapsed_per_node: int = 6,
    ):
        self.logic_converter = logic_converter
        self.merge_generator = AmrMergeGenerator(
            max_collapsed_per_node=max_collapsed_per_node,
            min_merge_depth=min_merge_depth,
            max_internal_merge_depth=0,
            allow_collapsing_coreferences=False,
        )

    def calculate(self, amr: Tree) -> AmrLogicStats:
        """Calculate statistics for an AMR graph."""
        num_instance_nodes = len(penman.interpret(amr).instances())
        num_nodes = count_nodes_in_amr_tree(amr)
        logic = self.logic_converter.convert(amr)
        num_logic_terms = count_amr_logic_terms(logic)
        tree_depth = find_amr_tree_depth(amr)
        num_possible_merges = len(self.merge_generator.generate_merged_amrs(amr)) - 1
        return AmrLogicStats(
            num_instance_nodes=num_instance_nodes,
            num_nodes=num_nodes,
            num_logic_terms=num_logic_terms,
            tree_depth=tree_depth,
            num_possible_merges=num_possible_merges,
        )

    def calculate_bulk(self, amrs: list[Tree]) -> AmrLogicBulkStats:
        """Calculate bulk statistics for a list of AMR graphs."""
        stats = []
        for amr in amrs:
            try:
                stats.append(self.calculate(amr))
            except Exception:
                print("Error calculating stats. Skipping.")
        return AmrLogicBulkStats(
            num_instance_nodes=_calculate_bulk_stats(
                [stat.num_instance_nodes for stat in stats]
            ),
            num_nodes=_calculate_bulk_stats([stat.num_nodes for stat in stats]),
            num_logic_terms=_calculate_bulk_stats(
                [stat.num_logic_terms for stat in stats]
            ),
            tree_depth=_calculate_bulk_stats([stat.tree_depth for stat in stats]),
            num_possible_merges=_calculate_bulk_stats(
                [stat.num_possible_merges for stat in stats]
            ),
        )


def _calculate_bulk_stats(values: list[float]) -> BulkStats:
    """Calculate bulk statistics for a list of values."""
    return BulkStats(
        mean=np.mean(values).item(),
        median=np.median(values).item(),
        stdev=np.std(values, ddof=1).item(),
        max=max(values),
        min=min(values),
    )


def count_amr_logic_terms(clause: alc.Clause) -> int:
    if type(clause) == alc.Atom:
        return 1
    if type(clause) == alc.Not or type(clause) == alc.All or type(clause) == alc.Exists:
        return count_amr_logic_terms(clause.body)
    if type(clause) == alc.And or type(clause) == alc.Or:
        return sum(count_amr_logic_terms(arg) for arg in clause.args)
    if type(clause) == alc.Implies:
        return count_amr_logic_terms(clause.antecedent) + count_amr_logic_terms(
            clause.consequent
        )
    raise ValueError(f"Unexpected value: {clause}")
