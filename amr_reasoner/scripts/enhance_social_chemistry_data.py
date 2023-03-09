from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from typing import Any
from amr_reasoner.parse.AmrParseCache import AmrParseCache
from amr_reasoner.parse.IbmAmrParser import IbmAmrParser

from amr_reasoner.datasets.social_chemistry.load_social_chemistry_data import (
    BaseSocialChemistrySample,
    SocialChemistrySample,
    load_base_social_chemistry_samples,
)
from amr_reasoner.parse.populate_amr_cache import populate_amr_cache

"""
Helper script to build up a AMR cache for the social chemistry dataset.
"""


def extract_sentences_to_parse(
    samples: list[BaseSocialChemistrySample],
) -> set[str]:
    sentences: set[str] = set()
    for sample in samples:
        sentences.add(sample.action_text)
        sentences.add(sample.rot_text)
        sentences.add(sample.situation_text)
    return sentences


def enhance_samples(
    samples: list[BaseSocialChemistrySample],
    cache: AmrParseCache,
) -> list[dict[str, Any]]:
    enhanced_samples = []
    for sample in samples:
        enhanced_sample = SocialChemistrySample(
            rot_judgment=sample.rot_judgment,
            split=sample.split,
            source=sample.source,
            action_text=sample.action_text,
            action_amr=cache.get_or_throw(sample.action_text),
            rot_text=sample.rot_text,
            rot_amr=cache.get_or_throw(sample.rot_text),
            situation_text=sample.situation_text,
            situation_amr=cache.get_or_throw(sample.situation_text),
        )
        enhanced_samples.append(asdict(enhanced_sample))
    return enhanced_samples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--social-chemistry-data-dir",
        type=str,
        default="data",
        help="Path to the Social Chemistry dataset directory",
    )
    parser.add_argument(
        "--amr-parser-checkpoint-path",
        type=str,
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
    )
    parser.add_argument(
        "--cache-file",
        type=str,
        default="cache.json",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="social_chemistry_101_data_enhanced.json",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    amr_parser = IbmAmrParser(args.amr_parser_checkpoint_path)
    cache = AmrParseCache(args.cache_file)

    samples = load_base_social_chemistry_samples(args.social_chemistry_data_dir)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    sentences = extract_sentences_to_parse(samples)
    populate_amr_cache(
        sentences,
        amr_parser,
        cache,
        verbose=args.verbose,
        batch_size=args.batch_size,
    )
    cache.save()
    enhanced_samples = enhance_samples(samples, cache)
    with open(args.output_file, "w") as f:
        json.dump(enhanced_samples, f, indent=2, ensure_ascii=False)
