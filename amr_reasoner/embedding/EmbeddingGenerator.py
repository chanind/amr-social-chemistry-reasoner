from __future__ import annotations

from abc import ABC
from typing import List
import numpy as np
import numpy.typing as npt


WordEmbeddings = List[npt.NDArray[np.float64]]


class EmbeddingGenerator(ABC):
    def generate_word_embeddings(self, sentences: list[str]) -> list[WordEmbeddings]:
        pass
