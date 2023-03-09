from __future__ import annotations

import numpy as np
import torch
from transformers import BatchEncoding, RobertaModel, RobertaTokenizerFast

from amr_reasoner.util import default_device

from .EmbeddingGenerator import EmbeddingGenerator, WordEmbeddings

MAX_LENGTH = 512


class RobertaEmbeddingGenerator(EmbeddingGenerator):
    tokenizer: RobertaTokenizerFast
    model: RobertaModel
    device: torch.device

    def __init__(
        self,
        model_name: str = "roberta-base",
        device: torch.device = default_device(),
        use_last_n_hidden_states: int = 1,
    ) -> None:
        self.tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
        self.model = RobertaModel.from_pretrained(model_name).to(device)
        self.device = device
        self.use_last_n_hidden_states = use_last_n_hidden_states

    def generate_word_embeddings(self, sentences: list[str]) -> list[WordEmbeddings]:
        tokenized_sentences = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            model_outputs = self.model(**tokenized_sentences, output_hidden_states=True)

            # average the embeddings over the last n hidden layers
            last_hidden_states = [
                layer.detach().cpu().numpy()
                for layer in model_outputs.hidden_states[
                    -1 * self.use_last_n_hidden_states :
                ]
            ]
            token_embeddings = np.mean(np.stack(last_hidden_states), axis=0)

            batch_word_embeddings: list[WordEmbeddings] = []
            for sent_index, sentence in enumerate(sentences):
                sent_word_embeddings: WordEmbeddings = []
                sent_token_embeddings = token_embeddings[sent_index, :, :]
                tokens_in_words = find_tokens_in_words(
                    tokenized_sentences, sent_index, len(sentence)
                )
                for word_tokens in tokens_in_words:
                    start_token = min(word_tokens)
                    end_token = max(word_tokens)
                    word_token_embeddings = sent_token_embeddings[
                        start_token : end_token + 1, :
                    ]
                    if end_token - start_token > 0:
                        word_embedding = np.mean(word_token_embeddings, axis=0)
                    else:
                        word_embedding = word_token_embeddings[0]
                    sent_word_embeddings.append(word_embedding)
                batch_word_embeddings.append(sent_word_embeddings)
        return batch_word_embeddings


def find_tokens_in_words(
    batch_encoding: BatchEncoding, sentence_index: int, sentence_len: int
) -> list[set[int]]:
    word_tokens: list[set[int]] = []
    cur_word_tokens: set[int] = set()
    for char_idx in range(sentence_len):
        token = batch_encoding.char_to_token(sentence_index, char_idx)
        if token is None and cur_word_tokens:
            word_tokens.append(cur_word_tokens)
            cur_word_tokens = set()
        elif token is not None:
            cur_word_tokens.add(token)
    if cur_word_tokens:
        word_tokens.append(cur_word_tokens)
    return word_tokens
