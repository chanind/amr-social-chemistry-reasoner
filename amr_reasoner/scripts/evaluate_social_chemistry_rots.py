from __future__ import annotations

import argparse
import json
import pickle
import random
from typing import Optional, cast

import optuna
from optuna import Study, Trial

from amr_reasoner.datasets.social_chemistry.evaluate_social_chemistry_rots import (
    evaluate_social_chemistry_rots,
)
from amr_reasoner.datasets.social_chemistry.load_social_chemistry_data import (
    SocialChemistrySample,
)


def grid_eval(
    social_chemistry_data_json: str,
    max_samples: Optional[int],
    grid_search_trials: int,
    verbose: bool,
) -> Study:
    with open(social_chemistry_data_json) as f:
        test_samples = [SocialChemistrySample(**s) for s in json.load(f)]

    if max_samples is not None:
        test_samples = test_samples[:max_samples]

    def objective(trial: Trial) -> float:
        roberta_model_name = trial.suggest_categorical(
            "roberta_model_name", ["roberta-base", "roberta-large"]
        )
        min_similarity_threshold = trial.suggest_float(
            "min_similarity_threshold", 0.85, 0.975, step=0.025
        )
        max_collapsed_per_node = trial.suggest_int("max_collapsed_per_node", 4, 6)
        use_last_n_hidden_states = trial.suggest_int("use_last_n_hidden_states", 2, 6)
        max_internal_merge_depth = trial.suggest_categorical(
            "max_internal_merge_depth", [None, 1]
        )
        allow_collapsing_coreferences = trial.suggest_categorical(
            "allow_collapsing_coreferences", [True, False]
        )

        # reset random seed before every run, otherwise negative samples will be different between runs
        random.seed(42)
        eval_result, _reasoner = evaluate_social_chemistry_rots(
            test_samples=test_samples,
            min_similarity_threshold=min_similarity_threshold,
            max_collapsed_per_node=max_collapsed_per_node,
            roberta_model_name=roberta_model_name,
            use_last_n_hidden_states=use_last_n_hidden_states,
            max_internal_merge_depth=cast(Optional[int], max_internal_merge_depth),
            allow_collapsing_coreferences=allow_collapsing_coreferences,
            verbose=verbose,
            print_final_results=True,
        )

        # want to weight precision a bit more highly than recall
        return eval_result.f1 * 0.5 + eval_result.precision * 0.5

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=grid_search_trials)
    return study


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
) -> None:
    with open(social_chemistry_data_json) as f:
        test_samples = [SocialChemistrySample(**s) for s in json.load(f)]

    if max_samples is not None:
        test_samples = test_samples[:max_samples]

    random.seed(42)
    evaluate_social_chemistry_rots(
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
        "--max-internal-merge-depth",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--allow-collapsing-coreferences",
        action="store_true",
    )
    parser.add_argument(
        "--grid-search-trials",
        type=int,
        default=50,
    )
    parser.add_argument(
        "--save-study-path",
        type=str,
        default=None,
    )
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()
    if args.grid:
        study = grid_eval(
            social_chemistry_data_json=args.social_chemistry_data_json,
            max_samples=args.max_samples,
            verbose=args.verbose,
            grid_search_trials=args.grid_search_trials,
        )
        if args.save_study_path is not None:
            with open(args.save_study_path, "wb") as file:
                pickle.dump(study, file)
    else:
        single_eval(
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
