from __future__ import annotations
from typing import Any, Optional

import numpy as np
import amr_logic_converter as alc
import tensor_theorem_prover as ttp

from amr_reasoner.embedding.EmbeddingGenerator import WordEmbeddings


def amr_logic_to_tensor_logic(
    clause: alc.Clause, embeddings: Optional[WordEmbeddings] = None
) -> ttp.Clause:
    if isinstance(clause, alc.And):
        return ttp.And(*(amr_logic_to_tensor_logic(f, embeddings) for f in clause.args))
    if isinstance(clause, alc.Or):
        return ttp.Or(*(amr_logic_to_tensor_logic(f, embeddings) for f in clause.args))
    if isinstance(clause, alc.Not):
        return ttp.Not(amr_logic_to_tensor_logic(clause.body, embeddings))
    if isinstance(clause, alc.Atom):
        embedding: Any = None
        if clause.alignment and embeddings is not None:
            embeddings_slices = [
                embeddings[index] for index in clause.alignment.indices
            ]
            embedding = np.stack(embeddings_slices).mean(axis=0)
        pred = ttp.Predicate(clause.symbol, embedding)
        return pred(
            *[amr_term_to_tensor_logic_term(arg, embeddings) for arg in clause.terms]
        )
    if isinstance(clause, alc.Exists):
        return ttp.Exists(
            ttp.Variable(clause.param.name),
            amr_logic_to_tensor_logic(clause.body, embeddings),
        )
    if isinstance(clause, alc.All):
        return ttp.All(
            ttp.Variable(clause.param.name),
            amr_logic_to_tensor_logic(clause.body, embeddings),
        )
    if isinstance(clause, alc.Implies):
        return ttp.Implies(
            amr_logic_to_tensor_logic(clause.antecedent, embeddings),
            amr_logic_to_tensor_logic(clause.consequent, embeddings),
        )
    raise ValueError(f"Unknown clause type: {type(clause)}")


def amr_term_to_tensor_logic_term(
    term: alc.Term, embeddings: Optional[WordEmbeddings] = None
) -> ttp.types.Term:
    if isinstance(term, alc.Constant):
        embedding: Any = None
        if term.alignment and embeddings is not None:
            embedding = embeddings[term.alignment.indices[0]]
        return ttp.Constant(term.value, embedding)
    if isinstance(term, alc.Variable):
        return ttp.Variable(term.name)
    raise ValueError(f"Unknown term type: {type(term)}")
