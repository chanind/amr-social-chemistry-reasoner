from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Optional

import tensor_theorem_prover as ttp
from tqdm import tqdm

from amr_reasoner.datasets.social_chemistry.RotReasoner import RotReasoner

from .load_social_chemistry_data import SocialChemistrySample


def get_verdicts(
    situation_amr: str, rot_amr: str, reasoner: RotReasoner, verbose: bool = False
) -> dict[str, list[ttp.Proof]]:

    rots_logic = reasoner.processor.rot_to_logic_with_alternatives(rot_amr)
    situations_logic = reasoner.processor.statement_to_logic_with_alternatives(
        situation_amr
    )

    # Only check for the verdict that the ROT can actually prove
    # No need to waste time looking for the other 3 verdicts
    verdict_goals = reasoner.processor.verdict_goals()
    target_verdict = "GOOD"
    if isinstance(rots_logic[0], ttp.Implies):
        consequent = rots_logic[0].consequent
        verdict_prefix = ""
        if isinstance(consequent, ttp.Not):
            verdict_prefix = "NOT_"
            consequent = consequent.body
        if isinstance(consequent, ttp.Atom):
            target_verdict = verdict_prefix + consequent.predicate.symbol.upper()

    goal = verdict_goals[target_verdict]
    reasoner.reset()
    reasoner.prover.extend_knowledge(rots_logic)

    proofs = reasoner.prover.prove_all(
        goal, extra_knowledge=situations_logic, max_proofs=1
    )

    if verbose:
        print(rot_amr)

        print("ROT LOGIC")
        for logic in rots_logic:
            print(logic)

        print("\nSITUATION LOGIC")
        for logic in situations_logic:
            print(logic)

        print("\n\n")

        print(rot_amr)

        print("Situation Verdicts")
        print(situation_amr)

        if len(proofs) > 0:
            print(f"VERDICT: {target_verdict}")
            for proof in proofs:
                print(proof)
                print("\n")
        else:
            print("NO VERDICTS :(")

    return {target_verdict: proofs}


def has_verdict(verdicts: dict[str, list[ttp.Proof]]) -> bool:
    res = False
    for _verdict, proofs in verdicts.items():
        if len(proofs) == 0:
            continue
        res = True
    return res


def pick_random_negative(
    samples: list[SocialChemistrySample], pos_sample: SocialChemistrySample
) -> SocialChemistrySample:
    while True:
        sample = random.choice(samples)
        if sample != pos_sample:
            return sample


@dataclass
class SampleResult:
    sample: SocialChemistrySample
    verdicts: dict[str, list[ttp.Proof]]
    type: Literal["TP", "FP", "TN", "FN"]
    rot_amr: str
    situation_amr: str


@dataclass
class EvalResult:
    true_positives: list[SampleResult]
    false_positives: list[SampleResult]
    true_negatives: list[SampleResult]
    false_negatives: list[SampleResult]

    @property
    def num_true_positives(self) -> int:
        return len(self.true_positives)

    @property
    def num_false_positives(self) -> int:
        return len(self.false_positives)

    @property
    def num_true_negatives(self) -> int:
        return len(self.true_negatives)

    @property
    def num_false_negatives(self) -> int:
        return len(self.false_negatives)

    @property
    def total_samples(self) -> int:
        return (
            self.num_true_positives
            + self.num_false_positives
            + self.num_true_negatives
            + self.num_false_negatives
        )

    @property
    def precision(self) -> float:
        return (
            0
            if self.num_true_positives == 0
            else self.num_true_positives
            / (self.num_true_positives + self.num_false_positives)
        )

    @property
    def recall(self) -> float:
        return (
            0
            if self.num_true_positives == 0
            else self.num_true_positives
            / (self.num_true_positives + self.num_false_negatives)
        )

    @property
    def f1(self) -> float:
        return (
            0
            if self.num_true_positives == 0
            else 2 * self.precision * self.recall / (self.precision + self.recall)
        )


def evaluate_social_chemistry_rots(
    test_samples: list[SocialChemistrySample],
    min_similarity_threshold: float,
    max_collapsed_per_node: int,
    roberta_model_name: str,
    use_last_n_hidden_states: int,
    max_internal_merge_depth: Optional[int] = None,
    allow_collapsing_coreferences: bool = False,
    verbose: bool = False,
    print_final_results: bool = False,
) -> tuple[EvalResult, RotReasoner]:
    reasoner = RotReasoner(
        max_proof_depth=13,
        max_resolvent_width=8,
        min_similarity_threshold=min_similarity_threshold,
        max_collapsed_per_node=max_collapsed_per_node,
        roberta_model_name=roberta_model_name,
        use_last_n_hidden_states=use_last_n_hidden_states,
        max_internal_merge_depth=max_internal_merge_depth,
        allow_collapsing_coreferences=allow_collapsing_coreferences,
    )

    true_positives = []
    false_positives = []
    true_negatives = []
    false_negatives = []

    for sample in tqdm(test_samples):
        try:
            negative_sample = pick_random_negative(test_samples, sample)
            situation_amr = sample.situation_amr
            pos_rot_amr = sample.rot_amr
            neg_rot_amr = negative_sample.rot_amr

            pos_verdicts = get_verdicts(situation_amr, pos_rot_amr, reasoner)
            neg_verdicts = get_verdicts(situation_amr, neg_rot_amr, reasoner)

            if has_verdict(pos_verdicts):
                true_positives.append(
                    SampleResult(sample, pos_verdicts, "TP", pos_rot_amr, situation_amr)
                )
            else:
                false_negatives.append(
                    SampleResult(sample, pos_verdicts, "FN", pos_rot_amr, situation_amr)
                )

            if has_verdict(neg_verdicts):
                false_positives.append(
                    SampleResult(sample, pos_verdicts, "FP", neg_rot_amr, situation_amr)
                )
                if verbose:
                    print("FALSE POSITIVE")
                    print("SITUATION:", sample.situation_text)
                    print("NEG_ROT:", negative_sample.rot_text)
            else:
                true_negatives.append(
                    SampleResult(sample, pos_verdicts, "TN", neg_rot_amr, situation_amr)
                )
        except AssertionError as e:
            print(f"Skipping sample due to assertion error: {e}")
            print(sample.situation_text)
            print(sample.rot_text)

    result = EvalResult(
        true_positives=true_positives,
        false_positives=false_positives,
        true_negatives=true_negatives,
        false_negatives=false_negatives,
    )

    if verbose or print_final_results:
        print("EVALUATION RESULTS")
        print(f"min_similarity_threshold: {min_similarity_threshold}")
        print(f"max_collapsed_per_node: {max_collapsed_per_node}")
        print(f"roberta_model_name: {roberta_model_name}")
        print(f"use_last_n_hidden_states: {use_last_n_hidden_states}")
        print(f"max_internal_merge_depth: {max_internal_merge_depth}")
        print(f"allow_collapsing_coreferences: {allow_collapsing_coreferences}")

        print(f"Precision: {result.precision}")
        print(f"Recall: {result.recall}")
        print(f"F1: {result.f1}")

        print(f"num_true_positives: {result.num_true_positives}")
        print(f"num_false_positives: {result.num_false_positives}")
        print(f"num_true_negatives: {result.num_true_negatives}")
        print(f"num_false_negatives: {result.num_false_negatives}")
        print(f"total: {result.total_samples}")

    return result, reasoner
