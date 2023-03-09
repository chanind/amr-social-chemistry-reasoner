from __future__ import annotations
from typing import Iterable

from amr_reasoner.parse.AmrParseCache import AmrParseCache
from amr_reasoner.parse.AmrParser import AmrParser
from amr_reasoner.parse.populate_amr_cache import populate_amr_cache


class FakeParser(AmrParser):
    def generate_amr_annotations(self, sentences: Iterable[str]) -> list[str]:
        return [f"AMR: {sentence}" for sentence in sentences]


fake_parser = FakeParser("/fake/path")


def test_populate_amr_cache() -> None:
    cache = AmrParseCache()
    populate_amr_cache(
        ["This is a test", "This is another test"],
        fake_parser,
        cache,
        save_cache_every_n_batches=None,
    )
    assert cache.get("This is a test") == "AMR: This is a test"
    assert cache.get("This is another test") == "AMR: This is another test"
