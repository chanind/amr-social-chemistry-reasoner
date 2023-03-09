from __future__ import annotations

from abc import ABC
from typing import Iterable


class AmrParser(ABC):
    def __init__(self, model_path: str) -> None:
        pass

    def generate_amr_annotations(self, sentences: Iterable[str]) -> list[str]:
        pass
