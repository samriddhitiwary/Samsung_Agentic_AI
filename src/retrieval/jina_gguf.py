"""HTTP client retrieval for Jina Code Embeddings 0.5B GGUF via llama.cpp."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from src.retrieval.jina_code import (
    DOCUMENT_INSTRUCTION,
    MODEL_NAME,
    NORMALIZE_EMBEDDINGS,
    POOLING_STRATEGY,
    QUERY_INSTRUCTION,
    corpus_hash,
    row_text,
)


GGUF_REPO = "jinaai/jina-code-embeddings-0.5b-GGUF"
GGUF_FILE = "jina-code-embeddings-0.5b-Q8_0.gguf"


def model_slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def truncate_with_tokenizer(
    *,
    tokenizer: Any,
    text: str,
    instruction: str,
    max_length: int,
) -> str:
    encoded = tokenizer(
        f"{instruction}{text}",
        add_special_tokens=True,
        truncation=True,
        max_length=max_length,
    )["input_ids"]
    return tokenizer.decode(encoded, skip_special_tokens=True)


def parse_embedding_response(payload: Any) -> np.ndarray:
    if isinstance(payload, list):
        values = payload
    elif isinstance(payload, dict) and "embedding" in payload:
        values = [payload]
    elif isinstance(payload, dict):
        values = payload.get("data") or payload.get("value")
    else:
        values = None
    if values is None:
        keys = sorted(payload) if isinstance(payload, dict) else type(payload).__name__
        raise ValueError(f"Unexpected embedding response shape: {keys}")

    embeddings = []
    for item in values:
        vector = item["embedding"] if isinstance(item, dict) else item
        if vector and isinstance(vector[0], list):
            vector = vector[0]
        embeddings.append(vector)
    return np.asarray(embeddings, dtype=np.float32)


def embed_http(
    *,
    texts: list[str],
    server_url: str,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    endpoint = server_url.rstrip("/") + "/embedding"
    session = requests.Session()
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[start : start + batch_size]
        response = session.post(endpoint, json={"content": batch}, timeout=600)
        response.raise_for_status()
        embeddings.append(parse_embedding_response(response.json()))
    result = np.vstack(embeddings).astype(np.float32)
    if NORMALIZE_EMBEDDINGS:
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / np.maximum(norms, 1e-12)
    return result


def prompt_hash(ids: list[str], prompts: list[str]) -> str:
    hasher = hashlib.sha256()
    for item_id, prompt in zip(ids, prompts, strict=True):
        hasher.update(item_id.encode("utf-8", errors="replace"))
        hasher.update(b"\0")
        hasher.update(prompt.encode("utf-8", errors="replace"))
        hasher.update(b"\0")
    return hasher.hexdigest()[:16]


def embed_http_cached(
    *,
    ids: list[str],
    texts: list[str],
    server_url: str,
    batch_size: int,
    desc: str,
    cache_dir: Path,
    cache_stem: str,
) -> tuple[np.ndarray, bool, Path]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = cache_dir / f"{cache_stem}.npy"
    meta_path = cache_dir / f"{cache_stem}.meta.json"

    metadata: dict[str, Any] | None = None
    if embeddings_path.exists() and meta_path.exists():
        candidate_metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            candidate_metadata.get("ids") == ids
            and candidate_metadata.get("prompt_hash") == prompt_hash(ids, texts)
            and candidate_metadata.get("normalization") == NORMALIZE_EMBEDDINGS
        ):
            done_count = int(candidate_metadata.get("done_count", 0))
            shape = tuple(candidate_metadata.get("shape", ()))
            if len(shape) == 2 and shape[0] == len(texts) and done_count >= len(texts):
                return np.load(embeddings_path).astype(np.float32), True, embeddings_path
            if len(shape) == 2 and shape[0] == len(texts) and 0 < done_count < len(texts):
                metadata = candidate_metadata

    if metadata is None:
        done_count = 0
        embedding_dim = None
    else:
        done_count = int(metadata.get("done_count", 0))
        embedding_dim = int(metadata["shape"][1])

    endpoint = server_url.rstrip("/") + "/embedding"
    session = requests.Session()
    progress = tqdm(range(done_count, len(texts), batch_size), desc=desc)
    mmap_array: np.memmap | None = None
    if embedding_dim is not None and embeddings_path.exists():
        mmap_array = np.lib.format.open_memmap(
            embeddings_path, mode="r+", dtype=np.float32, shape=(len(texts), embedding_dim)
        )

    for start in progress:
        batch = texts[start : start + batch_size]
        response = session.post(endpoint, json={"content": batch}, timeout=600)
        response.raise_for_status()
        embeddings = parse_embedding_response(response.json())
        if NORMALIZE_EMBEDDINGS:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-12)
        if mmap_array is None:
            embedding_dim = int(embeddings.shape[1])
            mmap_array = np.lib.format.open_memmap(
                embeddings_path,
                mode="w+",
                dtype=np.float32,
                shape=(len(texts), embedding_dim),
            )
            mmap_array[:] = np.nan
        end = start + len(batch)
        mmap_array[start:end] = embeddings.astype(np.float32)
        mmap_array.flush()
        done_count = end
        meta_path.write_text(
            json.dumps(
                {
                    "ids": ids,
                    "prompt_hash": prompt_hash(ids, texts),
                    "shape": [len(texts), int(embedding_dim)],
                    "done_count": done_count,
                    "normalization": NORMALIZE_EMBEDDINGS,
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    result = np.load(embeddings_path).astype(np.float32)
    return result, False, embeddings_path


def cache_paths(
    *,
    cache_dir: Path,
    gguf_file: str,
    max_length: int,
    fingerprint: str,
) -> tuple[Path, Path]:
    stem = f"{model_slug(gguf_file)}_nl2code_last_maxseq-{max_length}_{fingerprint}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.meta.json"


def encode_corpus_cached_gguf(
    *,
    corpus_rows: list[dict[str, Any]],
    tokenizer: Any,
    server_url: str,
    cache_dir: Path,
    max_length: int,
    batch_size: int,
    gguf_file: str = GGUF_FILE,
) -> tuple[list[str], np.ndarray, bool, Path, dict[str, Any]]:
    doc_ids = [row["id"] for row in corpus_rows]
    fingerprint = corpus_hash(corpus_rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path, meta_path = cache_paths(
        cache_dir=cache_dir,
        gguf_file=gguf_file,
        max_length=max_length,
        fingerprint=fingerprint,
    )
    if embeddings_path.exists() and meta_path.exists():
        data = np.load(embeddings_path)
        cached_doc_ids = data["doc_ids"].astype(str).tolist()
        embeddings = data["embeddings"].astype(np.float32)
        if cached_doc_ids == doc_ids:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            return doc_ids, embeddings, True, embeddings_path, metadata

    prompted = [
        truncate_with_tokenizer(
            tokenizer=tokenizer,
            text=row_text(row),
            instruction=DOCUMENT_INSTRUCTION,
            max_length=max_length,
        )
        for row in tqdm(corpus_rows, desc="GGUF prepare corpus prompts")
    ]
    embeddings = embed_http(
        texts=prompted,
        server_url=server_url,
        batch_size=batch_size,
        desc="GGUF encode corpus",
    )
    metadata = {
        "model_name": MODEL_NAME,
        "gguf_repo": GGUF_REPO,
        "gguf_file": gguf_file,
        "model_revision": None,
        "max_sequence_length": max_length,
        "embedding_dimension": int(embeddings.shape[1]),
        "query_instruction": QUERY_INSTRUCTION,
        "document_instruction": DOCUMENT_INSTRUCTION,
        "pooling_strategy": POOLING_STRATEGY,
        "normalization": NORMALIZE_EMBEDDINGS,
        "corpus_hash": fingerprint,
        "embedding_shape": list(embeddings.shape),
        "transport": "llama.cpp HTTP /embedding",
    }
    np.savez_compressed(
        embeddings_path,
        doc_ids=np.asarray(doc_ids),
        embeddings=embeddings,
    )
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return doc_ids, embeddings, False, embeddings_path, metadata


def retrieve_jina_code_gguf(
    *,
    queries: Iterable[dict[str, Any]],
    corpus: Iterable[dict[str, Any]],
    cache_dir: Path,
    server_url: str,
    max_length: int = 512,
    corpus_batch_size: int = 8,
    query_batch_size: int = 8,
    similarity_batch_size: int = 64,
    top_k: int = 100,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    query_rows = list(queries)
    corpus_rows = list(corpus)
    doc_ids, corpus_embeddings, cache_hit, cache_path, cache_metadata = (
        encode_corpus_cached_gguf(
            corpus_rows=corpus_rows,
            tokenizer=tokenizer,
            server_url=server_url,
            cache_dir=cache_dir,
            max_length=max_length,
            batch_size=corpus_batch_size,
        )
    )
    query_prompts = [
        truncate_with_tokenizer(
            tokenizer=tokenizer,
            text=row_text(row),
            instruction=QUERY_INSTRUCTION,
            max_length=max_length,
        )
        for row in tqdm(query_rows, desc="GGUF prepare query prompts")
    ]
    query_ids = [row["id"] for row in query_rows]
    query_cache_stem = (
        f"{model_slug(GGUF_FILE)}_query_nl2code_last_maxseq-{max_length}_"
        f"{prompt_hash(query_ids, query_prompts)}"
    )
    query_embeddings, query_cache_hit, query_cache_path = embed_http_cached(
        ids=query_ids,
        texts=query_prompts,
        server_url=server_url,
        batch_size=query_batch_size,
        desc="GGUF encode queries",
        cache_dir=cache_dir,
        cache_stem=query_cache_stem,
    )

    top_k = min(top_k, len(doc_ids))
    results: dict[str, dict[str, float]] = {}
    corpus_matrix = corpus_embeddings.T
    for start in tqdm(range(0, len(query_rows), similarity_batch_size), desc="GGUF top-k search"):
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
        **cache_metadata,
        "server_url": server_url,
        "corpus_embedding_cache_hit": cache_hit,
        "corpus_embedding_cache_path": str(cache_path),
        "query_embedding_cache_hit": query_cache_hit,
        "query_embedding_cache_path": str(query_cache_path),
        "query_embedding_shape": list(query_embeddings.shape),
    }
    return results, metadata
