from __future__ import annotations

import math
from typing import Iterable, Optional

import amr_logic_converter as alc
import penman
import tensor_theorem_prover as ttp
import torch
from amr_logic_converter import AmrLogicConverter
from amr_logic_converter.AmrLogicConverter import (
    OverrideConjunctionCallbackInfo,
    OverrideQuantificationCallbackInfo,
)
from penman.tree import Node
from tqdm import tqdm

from amr_reasoner.amr_logic_to_tensor_logic import amr_logic_to_tensor_logic
from amr_reasoner.AmrMergeGenerator import AmrMergeGenerator
from amr_reasoner.embedding.EmbeddingGenerator import EmbeddingGenerator
from amr_reasoner.embedding.RobertaEmbeddingGenerator import RobertaEmbeddingGenerator
from amr_reasoner.util import chunk_list, default_device

GOOD = alc.Predicate("good")
BAD = alc.Predicate("bad")

ROT_VERDICT_PREDICATES_MAP: dict[str, list[tuple[alc.Predicate, bool]]] = {
    "polite-01": [(GOOD, True)],
    "good-02": [(GOOD, True)],
    "good-04": [(GOOD, True)],
    "bad-02": [(BAD, True)],
    "bad-07": [(BAD, True)],
    "expect-01": [(GOOD, True)],
    "okay-04": [(GOOD, True)],
    "recommend-01": [(GOOD, True)],
    "wrong-02": [(BAD, True)],
    "reasonable-02": [(GOOD, True)],
    "possible-01": [(GOOD, True)],
    "important-01": [(GOOD, True)],
    "rude-01": [(BAD, True)],
    "hurt-01": [(BAD, True)],
    "appropriate-02": [(GOOD, True)],
    "right-05": [(GOOD, True)],
    "fine-04": [(GOOD, True)],
    "frown-02": [(BAD, True)],
    "waste-01": [(GOOD, False)],
    "understand-01": [(GOOD, True)],
    "control-01": [(BAD, True)],
    "mean-04": [(BAD, True)],
    "obligate-01": [(GOOD, True)],
    "fortunate-01": [(GOOD, True)],
    "consider-01": [(GOOD, True)],  # aka: considerate
    "kind-01": [(GOOD, True)],
    "normal-02": [(GOOD, True)],
    "nice-01": [(GOOD, True)],
    "respect-01": [(GOOD, True)],
    "disrespect-01": [(BAD, True)],
    "reveal-01": [(BAD, True)],  # seems like this is bad in use, but debatable
    # questionable if these should be allowed...
    "juvenile": [(BAD, True)],
    "heart": [(GOOD, True)],
    "rude-01ophile": [(BAD, True)],
}


def build_consequent(
    targets: list[tuple[alc.Predicate, bool]], event_term: alc.Term
) -> alc.Clause:
    """
    Build the consequent of the logic statement for the given event term.
    """
    consequent_literals = []
    for predicate, is_positive in targets:
        literal: alc.Clause = predicate(event_term)
        if not is_positive:
            literal = alc.Not(literal)
        consequent_literals.append(literal)
    if len(consequent_literals) == 1:
        return consequent_literals[0]
    return alc.And(*(consequent_literals))


def prune_amr_tree(tree: penman.Tree) -> penman.Tree:
    """
    Remove :rel nodes since these are alomst always parsing errors
    """
    new_node = prune_amr_node_recursive(tree.node)
    return penman.Tree(new_node, tree.metadata)


def prune_amr_node_recursive(node: Node) -> Node:
    instance, instance_info = node
    predicate, *edges = instance_info
    new_edges: list[Node] = []
    has_non_rel_edge = False
    for role, target in edges:
        if role != ":rel":
            has_non_rel_edge = True
            break
    for role, target in edges:
        if role == ":rel" and has_non_rel_edge:
            continue
        if isinstance(target, tuple):
            new_edges.append((role, prune_amr_node_recursive(target)))
        else:
            new_edges.append((role, target))
    return (instance, [predicate, *new_edges])


def parse_amr(amr: str) -> penman.Tree:
    """
    Parse the AMR string and prune the tree.
    """
    tree = penman.parse(amr)
    return prune_amr_tree(tree)


class RotProcessor:
    rot_logic_converter: AmrLogicConverter
    statement_logic_converter: AmrLogicConverter
    embedding_generator: EmbeddingGenerator
    merge_generator: AmrMergeGenerator

    def __init__(
        self,
        device: torch.device = default_device(),
        roberta_model_name: str = "roberta-base",
        max_collapsed_per_node: Optional[int] = None,
        max_internal_merge_depth: Optional[int] = None,
        use_last_n_hidden_states: int = 1,
        allow_collapsing_coreferences: bool = False,
    ) -> None:
        self.embedding_generator = RobertaEmbeddingGenerator(
            device=device,
            model_name=roberta_model_name,
            use_last_n_hidden_states=use_last_n_hidden_states,
        )
        self.rot_logic_converter = AmrLogicConverter(
            use_implies_for_conditions=True,
            use_variables_for_instances=True,
            override_conjunction=self._override_conjunction,
            override_quantification=self._override_quantification,
        )
        self.statement_logic_converter = AmrLogicConverter()
        self.merge_generator = AmrMergeGenerator(
            max_collapsed_per_node=max_collapsed_per_node,
            min_merge_depth=1,
            max_internal_merge_depth=max_internal_merge_depth,
            allow_collapsing_coreferences=allow_collapsing_coreferences,
        )

    def _override_quantification(
        self, clause: alc.Clause, info: OverrideQuantificationCallbackInfo
    ) -> alc.Clause | None:
        # if there's a NOT around the whole AMR, this should just apply to the consequent, not the antecedent
        if not info.is_negated or info.depth > 0 or type(clause) != alc.Implies:
            return None
        return alc.Implies(clause.antecedent, alc.Not(clause.consequent))

    def _override_conjunction(
        self, info: OverrideConjunctionCallbackInfo
    ) -> alc.Clause | None:
        # if we're not at the top or if there's already a condition (will turn into a implies anyway) just carry on
        if (
            info.depth > 0
            or info.condition_term is not None
            # closture term should be None if we're at the top of the AMR tree
            or info.closure_term is not None
            # there should only be 1 relation from the top predicate and thus only 1 subterm
            # or len(info.subterms) > 1
            or info.predicate_term.symbol not in ROT_VERDICT_PREDICATES_MAP
        ):
            return None
        predicate_relations = []
        antecedents = []
        for term in info.subterms:
            cur_term = term
            polarity = True
            if type(term) is alc.Not:
                cur_term = term.body
                polarity = False
            if type(cur_term) is not alc.And:
                continue
            for subterm in cur_term.args:
                if (
                    type(subterm) is alc.Atom
                    and subterm.symbol[0] == ":"
                    and subterm.terms[0] == info.predicate_term.terms[0]
                ):
                    predicate_relations.append(subterm)
                else:
                    antecedents.append(subterm)
        assert len(predicate_relations) > 0, "No predicate relation found"
        assert len(antecedents) > 0, "No antecedents found"
        consequent = build_consequent(
            ROT_VERDICT_PREDICATES_MAP[info.predicate_term.symbol],
            predicate_relations[0].terms[1],
        )
        # Treat the top-level conjunction as an implication
        antecedent: alc.Clause = alc.And(*antecedents)
        if not polarity:
            antecedent = alc.Not(antecedent)
        return alc.Implies(antecedent, consequent)

    def rots_to_logic_bulk(
        self,
        rot_text_amrs: Iterable[str],
        skip_invalid_logic: bool = True,
        batch_size: int = 100,
    ) -> list[ttp.Clause]:
        """
        Parse the rot amr into logic statements for tensor-theorem-prover.
        """
        # extract the original text from the AMR tokenization to avoid incorrect tokenization lengths
        amr_trees = [parse_amr(amr) for amr in tqdm(set(rot_text_amrs))]
        result_logic = []
        for batch in tqdm(
            chunk_list(amr_trees, batch_size),
            total=math.ceil(len(amr_trees) / batch_size),
        ):
            batch_sentences = [amr.metadata["tok"] for amr in batch]
            word_embeddings = self.embedding_generator.generate_word_embeddings(
                batch_sentences
            )
            for amr_tree, embeddings in zip(batch, word_embeddings):
                try:
                    logic = self.rot_logic_converter.convert(amr_tree)
                    result_logic.append(amr_logic_to_tensor_logic(logic, embeddings))
                except (AssertionError, IndexError) as e:
                    if skip_invalid_logic:
                        continue
                    raise e
        return result_logic

    def rot_to_logic(self, rot_text_amr: str) -> ttp.Clause:
        """
        Parse the rot amr into logic statements for tensor-theorem-prover.
        """
        amr_tree = parse_amr(rot_text_amr)
        # extract the original text from the AMR tokenization to avoid incorrect tokenization lengths
        text = amr_tree.metadata["tok"]
        [embeddings] = self.embedding_generator.generate_word_embeddings([text])
        logic = self.rot_logic_converter.convert(amr_tree)
        return amr_logic_to_tensor_logic(logic, embeddings)

    def rot_to_logic_with_alternatives(self, rot_text_amr: str) -> list[ttp.Clause]:
        """
        parse the rot amr into logic statements, including alternative amr from collapsing nodes
        """
        amr_tree = parse_amr(rot_text_amr)
        # extract the original text from the AMR tokenization to avoid incorrect tokenization lengths
        text = amr_tree.metadata["tok"]
        [embeddings] = self.embedding_generator.generate_word_embeddings([text])
        alternative_trees = self.merge_generator.generate_merged_amrs(amr_tree)

        alternative_logics = [
            self.rot_logic_converter.convert(tree) for tree in alternative_trees
        ]
        return [
            amr_logic_to_tensor_logic(logic, embeddings) for logic in alternative_logics
        ]

    def statement_to_logic(self, statement_text_amr: str) -> ttp.Clause:
        """
        Parse the statement amr into logic statements for tensor-theorem-prover.
        """
        amr_tree = parse_amr(statement_text_amr)
        # extract the original text from the AMR tokenization to avoid incorrect tokenization lengths
        text = amr_tree.metadata["tok"]
        [embeddings] = self.embedding_generator.generate_word_embeddings([text])
        logic = self.statement_logic_converter.convert(amr_tree)
        return amr_logic_to_tensor_logic(logic, embeddings)

    def statement_to_logic_with_alternatives(
        self, statement_text_amr: str
    ) -> list[ttp.Clause]:
        """
        parse the statement amr into logic statements, including alternative amr from collapsing nodes
        """
        amr_tree = parse_amr(statement_text_amr)
        # extract the original text from the AMR tokenization to avoid incorrect tokenization lengths
        text = amr_tree.metadata["tok"]
        [embeddings] = self.embedding_generator.generate_word_embeddings([text])
        alternative_logics = [
            self.statement_logic_converter.convert(tree)
            for tree in self.merge_generator.generate_merged_amrs(amr_tree)
        ]
        return [
            amr_logic_to_tensor_logic(logic, embeddings) for logic in alternative_logics
        ]

    def verdict_goals(self) -> dict[str, ttp.Clause]:
        """
        Goals to query for various verdicts
        """
        Event = alc.Variable("Event")
        return {
            "GOOD": amr_logic_to_tensor_logic(GOOD(Event)),
            "BAD": amr_logic_to_tensor_logic(BAD(Event)),
            "NOT_GOOD": amr_logic_to_tensor_logic(alc.Not(GOOD(Event))),
            "NOT_BAD": amr_logic_to_tensor_logic(alc.Not(BAD(Event))),
        }
