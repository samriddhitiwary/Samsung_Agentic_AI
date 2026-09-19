"""Dense embedding retrieval with cached corpus embeddings."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")


def _row_text(row: dict[str, Any]) -> str:
    return f"{row.get('title') or ''}\n{row.get('text') or ''}".strip()


def _corpus_fingerprint(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(_row_text(row).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def _load_model(model_name: str, max_seq_length: int | None) -> SentenceTransformer:
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    if max_seq_length is not None:
        model.max_seq_length = max_seq_length
    return model


def _cache_paths(cache_dir: Path, model_name: str, fingerprint: str) -> tuple[Path, Path]:
    stem = f"{_model_slug(model_name)}_{fingerprint}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.meta.json"


def encode_corpus_cached(
    *,
    model: SentenceTransformer,
    model_name: str,
    corpus: Iterable[dict[str, Any]],
    cache_dir: Path,
    batch_size: int,
    max_seq_length: int | None,
) -> tuple[list[str], np.ndarray, bool, Path]:
    """Encode corpus texts once and reuse normalized embeddings from disk."""

    rows = list(corpus)
    doc_ids = [row["id"] for row in rows]
    fingerprint = _corpus_fingerprint(rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = f"{model_name}_maxseq-{max_seq_length or 'model-default'}"
    embeddings_path, meta_path = _cache_paths(cache_dir, cache_key, fingerprint)

    if embeddings_path.exists() and meta_path.exists():
        data = np.load(embeddings_path)
        cached_doc_ids = data["doc_ids"].astype(str).tolist()
        embeddings = data["embeddings"].astype(np.float32)
        if cached_doc_ids == doc_ids:
            return doc_ids, embeddings, True, embeddings_path

    embeddings = model.encode(
        [_row_text(row) for row in rows],
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    np.savez_compressed(
        embeddings_path,
        doc_ids=np.asarray(doc_ids),
        embeddings=embeddings,
    )
    meta_path.write_text(
        json.dumps(
            {
                "model_name": model_name,
                "max_seq_length": max_seq_length,
                "fingerprint": fingerprint,
                "embedding_shape": list(embeddings.shape),
                "normalized": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return doc_ids, embeddings, False, embeddings_path


def retrieve_dense(
    *,
    queries: Iterable[dict[str, Any]],
    corpus: Iterable[dict[str, Any]],
    model_name: str = DEFAULT_DENSE_MODEL,
    cache_dir: Path = Path("data/cache"),
    top_k: int = 100,
    corpus_batch_size: int = 8,
    query_batch_size: int = 16,
    similarity_batch_size: int = 64,
    max_seq_length: int | None = 128,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Return MTEB-compatible dense retrieval rankings and run metadata."""

    query_rows = list(queries)
    model = _load_model(model_name, max_seq_length=max_seq_length)
    doc_ids, corpus_embeddings, cache_hit, cache_path = encode_corpus_cached(
        model=model,
        model_name=model_name,
        corpus=corpus,
        cache_dir=cache_dir,
        batch_size=corpus_batch_size,
        max_seq_length=max_seq_length,
    )
    query_embeddings = model.encode(
        [_row_text(row) for row in query_rows],
        batch_size=query_batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    top_k = min(top_k, len(doc_ids))
    results: dict[str, dict[str, float]] = {}
    corpus_matrix = corpus_embeddings.T
    for start in tqdm(range(0, len(query_rows), similarity_batch_size), desc="Dense top-k search"):
        end = min(start + similarity_batch_size, len(query_rows))
        scores = query_embeddings[start:end] @ corpus_matrix
        candidate_idx = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
        candidate_scores = np.take_along_axis(scores, candidate_idx, axis=1)
        order = np.argsort(candidate_scores, axis=1)[:, ::-1]
        sorted_idx = np.take_along_axis(candidate_idx, order, axis=1)
        sorted_scores = np.take_along_axis(candidate_scores, order, axis=1)
        for offset, row in enumerate(query_rows[start:end]):
            results[row["id"]] = {
                doc_ids[int(index)]: float(sorted_scores[offset, rank])
                for rank, index in enumerate(sorted_idx[offset])
            }

    metadata = {
        "model_name": model_name,
        "normalized_embeddings": True,
        "max_seq_length": max_seq_length,
        "corpus_embedding_cache_hit": cache_hit,
        "corpus_embedding_cache_path": str(cache_path),
        "corpus_embedding_shape": list(corpus_embeddings.shape),
        "query_embedding_shape": list(query_embeddings.shape),
    }
    return results, metadata
