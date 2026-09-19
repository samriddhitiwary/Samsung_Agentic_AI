"""BM25 lexical retrieval baselines."""

from __future__ import annotations

import re
from collections.abc import Iterable

import bm25s
import numpy as np
from tqdm import tqdm


_BASIC_RE = re.compile(r"[A-Za-z0-9_]+")
_CODE_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+|==|!=|<=|>=|[-+*/%]=?|[(){}\[\].,:]")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def tokenize_basic(text: str) -> list[str]:
    return [token.lower() for token in _BASIC_RE.findall(text)]


def tokenize_code(text: str) -> list[str]:
    return [token.lower() for token in _CODE_RE.findall(text)]


def tokenize_code_split(text: str) -> list[str]:
    tokens: list[str] = []
    for token in tokenize_code(text):
        parts = []
        for snake_part in token.split("_"):
            parts.extend(part for part in _CAMEL_BOUNDARY_RE.split(snake_part) if part)
        tokens.append(token)
        tokens.extend(part.lower() for part in parts if part and part.lower() != token)
    return tokens


TOKENIZERS = {
    "basic": tokenize_basic,
    "code": tokenize_code,
    "code_split": tokenize_code_split,
}


def retrieve_bm25(
    *,
    queries: Iterable[dict[str, str]],
    corpus: Iterable[dict[str, str]],
    tokenizer_name: str = "code_split",
    top_k: int = 100,
) -> dict[str, dict[str, float]]:
    """Return an MTEB-compatible top-k ranking dict using BM25Okapi."""

    if tokenizer_name not in TOKENIZERS:
        raise ValueError(f"Unknown BM25 tokenizer {tokenizer_name!r}; expected one of {sorted(TOKENIZERS)}")

    tokenizer = TOKENIZERS[tokenizer_name]
    corpus_rows = list(corpus)
    query_rows = list(queries)
    doc_ids = [row["id"] for row in corpus_rows]
    tokenized_corpus = [
        tokenizer(f"{row.get('title') or ''}\n{row.get('text') or ''}") for row in tqdm(corpus_rows, desc="BM25 tokenize corpus")
    ]
    bm25 = bm25s.BM25(method="lucene", k1=1.5, b=0.75)
    bm25.index(tokenized_corpus, show_progress=True)

    tokenized_queries = [
        tokenizer(f"{row.get('title') or ''}\n{row.get('text') or ''}") for row in tqdm(query_rows, desc="BM25 tokenize queries")
    ]
    top_k = min(top_k, len(doc_ids))
    retrieved = bm25.retrieve(
        tokenized_queries,
        k=top_k,
        sorted=True,
        show_progress=True,
        n_threads=0,
    )

    results: dict[str, dict[str, float]] = {}
    for row, indices, scores in zip(
        query_rows,
        retrieved.documents,
        np.asarray(retrieved.scores, dtype=np.float32),
        strict=True,
    ):
        results[row["id"]] = {
            doc_ids[int(index)]: float(score)
            for index, score in zip(indices, scores, strict=True)
        }
    return results
