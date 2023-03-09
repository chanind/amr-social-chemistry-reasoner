from __future__ import annotations

from typing import Iterable


from .AmrParser import AmrParser


class IbmAmrParser(AmrParser):
    def __init__(self, model_path: str) -> None:
        # import inside init since this isn't on PyPI and is a huge pain to get working
        from transition_amr_parser.parse import AMRParser

        self.parser = AMRParser.from_checkpoint(model_path)

    def generate_amr_annotations(self, sentences: Iterable[str]) -> list[str]:
        tokenized_sentences = [
            self.parser.tokenize(sentence)[0] for sentence in sentences
        ]
        sentences_annotations, _ = self.parser.parse_sentences(tokenized_sentences)
        return sentences_annotations
