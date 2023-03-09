from __future__ import annotations

import json
from dataclasses import dataclass
from os import path
from typing import Literal, Optional

import pandas as pd

SOCIAL_CHEMISTRY_TSV_FILE = "social-chem-101.v1.0.tsv"

Split = Literal["train", "dev", "test"]
Source = Literal["amitheasshole", "confessions", "dearabby", "rocstories"]


@dataclass(frozen=True)
class BaseSocialChemistrySample:
    rot_text: str
    rot_judgment: str
    situation_text: str
    action_text: str
    source: Source
    split: Split


@dataclass(frozen=True)
class SocialChemistrySample:
    rot_text: str
    rot_amr: str
    rot_judgment: str
    situation_text: str
    situation_amr: str
    action_text: str
    action_amr: str
    source: Source
    split: Split


def load_raw_social_chemistry_dataframe(dataset_path: str) -> pd.DataFrame:
    file_path = (
        dataset_path
        if dataset_path.lower().endswith(".tsv")
        else path.join(dataset_path, SOCIAL_CHEMISTRY_TSV_FILE)
    )
    df = pd.read_csv(file_path, sep="\t").convert_dtypes()
    return df


def load_base_social_chemistry_samples(
    dataset_path: str,
    split: Optional[Split] = None,
    dedupe_by_rot: bool = True,
) -> list[BaseSocialChemistrySample]:
    df = load_raw_social_chemistry_dataframe(dataset_path)
    if split is not None:
        df = df[df["split"] == split]
    samples = []
    seen_rots: set[str] = set()
    for _, row in df.iterrows():
        rot = row["rot"]
        if dedupe_by_rot and rot in seen_rots:
            continue
        samples.append(
            BaseSocialChemistrySample(
                rot_text=row["rot"],
                rot_judgment=row["rot-judgment"],
                situation_text=row["situation"],
                action_text=row["action"],
                split=row["split"],
                source=row["area"],
            )
        )
    return samples


def load_social_chemistry_samples(
    enhanced_data_json_path: str,
) -> list[SocialChemistrySample]:
    samples = []
    with open(enhanced_data_json_path, "r") as f:
        enhanced_data = json.load(f)
        for sample_dict in enhanced_data:
            samples.append(SocialChemistrySample(**sample_dict))
    return samples
