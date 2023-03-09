from __future__ import annotations

from typing import Iterable, TypeVar

import torch

T = TypeVar("T")


def chunk_list(items: list[T], chunk_size: int) -> Iterable[list[T]]:
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


def flatten(lst: list[list[T]]) -> list[T]:
    """Only flattens one level of nesting, but should be good enough."""
    return [item for sublist in lst for item in sublist]
