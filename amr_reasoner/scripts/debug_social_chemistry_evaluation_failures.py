from __future__ import annotations

import argparse
import json
import random
from typing import Callable, Optional

import penman

from amr_reasoner.amr_util import count_nodes_in_amr_tree
from amr_reasoner.datasets.social_chemistry.evaluate_social_chemistry_rots import (
    EvalResult,
    SampleResult,
    evaluate_social_chemistry_rots,
)
from amr_reasoner.datasets.social_chemistry.load_social_chemistry_data import (
    SocialChemistrySample,
)
from amr_reasoner.datasets.social_chemistry.RotReasoner import RotReasoner
from amr_reasoner.util import flatten


def single_eval(
    social_chemistry_data_json: str,
    max_samples: Optional[int],
    verbose: bool,
    min_similarity_threshold: float = 0.8,
    max_collapsed_per_node: int = 3,
    use_last_n_hidden_states: int = 4,
    max_internal_merge_depth: Optional[int] = None,
    roberta_model_name: str = "roberta-base",
    allow_collapsing_coreferences: bool = False,
) -> tuple[EvalResult, RotReasoner]:
    with open(social_chemistry_data_json) as f:
        test_samples = [SocialChemistrySample(**s) for s in json.load(f)]

    random.seed(42)
    random.shuffle(test_samples)
    if max_samples is not None:
        test_samples = test_samples[:max_samples]

    return evaluate_social_chemistry_rots(
        test_samples=test_samples,
        min_similarity_threshold=min_similarity_threshold,
        max_collapsed_per_node=max_collapsed_per_node,
        roberta_model_name=roberta_model_name,
        use_last_n_hidden_states=use_last_n_hidden_states,
        max_internal_merge_depth=max_internal_merge_depth,
        allow_collapsing_coreferences=allow_collapsing_coreferences,
        verbose=verbose,
        print_final_results=True,
    )


FAILURE_TYPES = {
    "FP": "False Positive",
    "FN": "False Negative",
    "TP": "True Positive",
    "TN": "True Negative",
}


def debug_results(results: EvalResult, reasoner: RotReasoner) -> None:
    failures: list[SampleResult] = [
        *results.false_positives,
        *results.false_negatives,
    ]

    random.seed(42)
    random.shuffle(failures)

    for sample_res in failures:
        show_original(sample_res, reasoner)
        print()

        instructions = """
Enter one of the following:
 - "merge" or "m" to print the largest merge for both ROT and situation
 - "merge_all" or "ma" to print all merges for both ROT and situation
 - "proof" or "p" to print the proof, if this is a false positive
 - "logic" or "l" to print the base logic for both ROT and situation
 - "original" or "o" to print the original AMR for both ROT and situation
 - "quit" or "q" to quit
 - "cont" or "c" to continue
        """

        cmd_mapping: dict[str, Callable[[SampleResult, RotReasoner], None]] = {
            "merge": show_merges,
            "m": show_merges,
            "merge_all": show_all_merges_cmd,
            "ma": show_all_merges_cmd,
            "proof": show_proof,
            "p": show_proof,
            "logic": show_logic,
            "l": show_logic,
            "original": show_original,
            "o": show_original,
            "quit": quit_cmd,
            "q": quit_cmd,
        }

        while True:
            print(instructions)
            cmd = input()
            if cmd in ["cont", "c"]:
                break
            if cmd not in cmd_mapping:
                print("Invalid command")
            else:
                cmd_mapping[cmd](sample_res, reasoner)
            print(" -------------------")


def show_all_merges_cmd(sample_res: SampleResult, reasoner: RotReasoner) -> None:
    show_merges(sample_res, reasoner, True)


def show_merges(
    sample_res: SampleResult, reasoner: RotReasoner, show_all: bool = False
) -> None:
    rot_merges = reasoner.processor.merge_generator.generate_merged_amrs(
        penman.parse(sample_res.rot_amr)
    )
    situation_merges = reasoner.processor.merge_generator.generate_merged_amrs(
        penman.parse(sample_res.situation_amr)
    )
    rot_merges.sort(key=count_nodes_in_amr_tree)
    situation_merges.sort(key=count_nodes_in_amr_tree)

    if not show_all:
        rot_merges = rot_merges[:1]
        situation_merges = situation_merges[:1]

    print("ROT MERGES")
    for merge in rot_merges:
        print(penman.format(merge))
        print()

    print("SITUATION MERGES")
    for merge in situation_merges:
        print(penman.format(merge))
        print()


def show_proof(sample_res: SampleResult, _reasoner: RotReasoner) -> None:
    proofs = flatten(list(sample_res.verdicts.values()))
    if len(proofs) == 0:
        print("No proofs")
        return
    print("PROOFS")
    for proof in proofs:
        print(proof)


def show_logic(sample_res: SampleResult, reasoner: RotReasoner) -> None:
    print("ROT LOGIC")
    print(reasoner.processor.rot_logic_converter.convert(sample_res.rot_amr))

    print("SITUATION LOGIC")
    print(
        reasoner.processor.statement_logic_converter.convert(sample_res.situation_amr)
    )


def show_original(sample_res: SampleResult, _reasoner: RotReasoner) -> None:
    print(f"FAILURE TYPE: {FAILURE_TYPES[sample_res.type]}\n")

    print("ROT")
    print(sample_res.rot_amr)

    print("SITUATION")
    print(sample_res.situation_amr)


def quit_cmd(_sample_res: SampleResult, _reasoner: RotReasoner) -> None:
    quit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--social-chemistry-data-json",
        type=str,
        default="data/social_chemistry_101_data_enhanced.json",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--min-similarity-threshold",
        type=float,
        default=0.925,
    )
    parser.add_argument(
        "--max-collapsed-per-node",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--use-last-n-hidden-states",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--roberta-model-name",
        type=str,
        default="roberta-base",
    )
    parser.add_argument(
        "--allow-collapsing-coreferences",
        action="store_true",
    )
    parser.add_argument(
        "--max-internal-merge-depth",
        type=int,
        default=None,
    )
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()
    results, reasoner = single_eval(
        social_chemistry_data_json=args.social_chemistry_data_json,
        max_samples=args.max_samples,
        verbose=args.verbose,
        min_similarity_threshold=args.min_similarity_threshold,
        max_collapsed_per_node=args.max_collapsed_per_node,
        use_last_n_hidden_states=args.use_last_n_hidden_states,
        roberta_model_name=args.roberta_model_name,
        max_internal_merge_depth=args.max_internal_merge_depth,
        allow_collapsing_coreferences=args.allow_collapsing_coreferences,
    )
    debug_results(results, reasoner)
