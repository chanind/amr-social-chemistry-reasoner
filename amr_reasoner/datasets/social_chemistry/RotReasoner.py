from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from tensor_theorem_prover import ResolutionProver
from tensor_theorem_prover.prover.Proof import Proof

from amr_reasoner.amr_similarity_func import amr_similarity_func
from amr_reasoner.parse.AmrParser import AmrParser

from .RotProcessor import RotProcessor


@dataclass
class RotVerdicts:
    """Verdicts from the RotReasoner"""

    verdict_proofs: dict[str, list[Proof]]

    @property
    def has_verdicts(self) -> bool:
        """Whether any verdicts were proven"""
        return any(self.verdicts)

    @property
    def verdicts(self) -> list[str]:
        """Verdicts that were able to be proven"""
        return [verdict for verdict, proofs in self.verdict_proofs.items() if proofs]

    def proofs(self, verdict: str) -> list[Proof] | None:
        return self.verdict_proofs.get(verdict)


class RotReasoner:
    prover: ResolutionProver
    processor: RotProcessor

    def __init__(
        self,
        max_proof_depth: int = 5,
        min_similarity_threshold: float = 0.5,
        max_resolvent_width: int = 10,
        max_collapsed_per_node: Optional[int] = None,
        max_internal_merge_depth: Optional[int] = None,
        use_last_n_hidden_states: int = 1,
        roberta_model_name: str = "roberta-base",
        allow_collapsing_coreferences: bool = False,
        processor: RotProcessor | None = None,
    ) -> None:
        self.prover = ResolutionProver(
            knowledge=[],
            max_proof_depth=max_proof_depth,
            min_similarity_threshold=min_similarity_threshold,
            max_resolution_attempts=1_000_000_000,
            max_resolvent_width=max_resolvent_width,
            similarity_func=amr_similarity_func(1.0),
            find_highest_similarity_proofs=False,
        )
        self.processor = processor or RotProcessor(
            max_collapsed_per_node=max_collapsed_per_node,
            use_last_n_hidden_states=use_last_n_hidden_states,
            roberta_model_name=roberta_model_name,
            max_internal_merge_depth=max_internal_merge_depth,
            allow_collapsing_coreferences=allow_collapsing_coreferences,
        )

    def extend_knowledge_from_rots(
        self, rot_amrs: Iterable[str], batch_size: int = 256
    ) -> None:
        self.prover.extend_knowledge(
            self.processor.rots_to_logic_bulk(rot_amrs, batch_size=batch_size)
        )

    def reset(self) -> None:
        self.prover.reset()

    def query_situation(
        self, situation_text_amr: str, max_proofs: Optional[int] = None
    ) -> RotVerdicts:
        goals = self.processor.verdict_goals()
        return RotVerdicts(
            dict(
                {
                    verdict: self.query_situation_for_verdict(
                        situation_text_amr, verdict, max_proofs
                    )
                    for verdict in goals.keys()
                }
            )
        )

    def query_situation_for_verdict(
        self, situation_text_amr: str, verdict: str, max_proofs: Optional[int] = None
    ) -> list[Proof]:
        goal = self.processor.verdict_goals()[verdict]
        situation_logic = self.processor.statement_to_logic(situation_text_amr)
        return self.prover.prove_all(
            goal, extra_knowledge=[situation_logic], max_proofs=max_proofs
        )

    def parse_and_query_situation(
        self, situation_text: str, parser: AmrParser, max_proofs: Optional[int] = None
    ) -> RotVerdicts:
        situation_amr = parser.generate_amr_annotations([situation_text])[0]
        return self.query_situation(situation_amr, max_proofs=max_proofs)
