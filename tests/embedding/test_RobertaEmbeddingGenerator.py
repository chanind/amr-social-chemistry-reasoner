from __future__ import annotations

from amr_reasoner.embedding.RobertaEmbeddingGenerator import RobertaEmbeddingGenerator


def test_RobertaEmbeddingGenerator() -> None:
    texts = [
        "People are expected to participate in the big events in their friends ' lives if asked .",
        "It 's reasonable for someone to call themselves an American if they 're from an American country .",
        "afsdhfkjsdhfjdshfjsd ok then",
    ]
    generator = RobertaEmbeddingGenerator()
    embeddings = generator.generate_word_embeddings(texts)
    assert len(embeddings) == len(texts)

    assert len(embeddings[0]) == len(texts[0].split())
    assert len(embeddings[1]) == len(texts[1].split())
    assert len(embeddings[2]) == len(texts[2].split())
