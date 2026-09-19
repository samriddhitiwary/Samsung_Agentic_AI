"""Jina Code Embeddings 0.5B retrieval with official NL-to-code prompts."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "jinaai/jina-code-embeddings-0.5b"
QUERY_INSTRUCTION = "Find the most relevant code snippet given the following query:\n"
DOCUMENT_INSTRUCTION = "Candidate code snippet:\n"
POOLING_STRATEGY = "last_token"
NORMALIZE_EMBEDDINGS = True


def model_slug(model_name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", model_name).strip("_")


def row_text(row: dict[str, Any]) -> str:
    return f"{row.get('title') or ''}\n{row.get('text') or ''}".strip()


def corpus_hash(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(str(row["id"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(row_text(row).encode("utf-8", errors="replace"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def add_query_instruction(text: str) -> str:
    return f"{QUERY_INSTRUCTION}{text}"


def add_document_instruction(text: str) -> str:
    return f"{DOCUMENT_INSTRUCTION}{text}"


def load_jina_model(
    *,
    model_name: str = MODEL_NAME,
    local_files_only: bool = False,
) -> tuple[Any, Any, torch.device, str | None]:
    """Load tokenizer/model for CPU or CUDA inference."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    revision = getattr(getattr(model, "config", None), "_commit_hash", None)
    return tokenizer, model, device, revision


def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    left_padding = bool((attention_mask[:, -1].sum() == attention_mask.shape[0]).item())
    if left_padding:
        return last_hidden_states[:, -1]
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_states.shape[0]
    return last_hidden_states[
        torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
    ]


def token_length_stats(
    *,
    tokenizer: Any,
    texts: Iterable[str],
    instruction: str,
) -> dict[str, int | float]:
    lengths = [
        len(tokenizer(f"{instruction}{text}", add_special_tokens=True)["input_ids"])
        for text in tqdm(list(texts), desc="Token length scan")
    ]
    values = np.asarray(lengths, dtype=np.int32)
    return {
        "count": int(values.size),
        "median": float(np.percentile(values, 50)),
        "p75": float(np.percentile(values, 75)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "max": int(values.max()),
    }


def encode_texts(
    *,
    texts: list[str],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    instruction: str,
    max_length: int,
    batch_size: int,
    desc: str,
) -> np.ndarray:
    embeddings: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = [f"{instruction}{text}" for text in texts[start : start + batch_size]]
        batch = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            outputs = model(**batch)
            pooled = last_token_pool(outputs.last_hidden_state, batch["attention_mask"])
            if NORMALIZE_EMBEDDINGS:
                pooled = F.normalize(pooled.float(), p=2, dim=1)
        embeddings.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.vstack(embeddings)


def cache_paths(
    *,
    cache_dir: Path,
    model_name: str,
    max_length: int,
    fingerprint: str,
) -> tuple[Path, Path]:
    stem = f"{model_slug(model_name)}_nl2code_last_maxseq-{max_length}_{fingerprint}"
    return cache_dir / f"{stem}.npz", cache_dir / f"{stem}.meta.json"


def encode_corpus_cached(
    *,
    corpus_rows: list[dict[str, Any]],
    tokenizer: Any,
    model: Any,
    device: torch.device,
    model_name: str,
    model_revision: str | None,
    max_length: int,
    batch_size: int,
    cache_dir: Path,
) -> tuple[list[str], np.ndarray, bool, Path, dict[str, Any]]:
    doc_ids = [row["id"] for row in corpus_rows]
    fingerprint = corpus_hash(corpus_rows)
    cache_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path, meta_path = cache_paths(
        cache_dir=cache_dir,
        model_name=model_name,
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

    embeddings = encode_texts(
        texts=[row_text(row) for row in corpus_rows],
        tokenizer=tokenizer,
        model=model,
        device=device,
        instruction=DOCUMENT_INSTRUCTION,
        max_length=max_length,
        batch_size=batch_size,
        desc="Jina encode corpus",
    )
    metadata = {
        "model_name": model_name,
        "model_revision": model_revision,
        "max_sequence_length": max_length,
        "embedding_dimension": int(embeddings.shape[1]),
        "query_instruction": QUERY_INSTRUCTION,
        "document_instruction": DOCUMENT_INSTRUCTION,
        "pooling_strategy": POOLING_STRATEGY,
        "normalization": NORMALIZE_EMBEDDINGS,
        "corpus_hash": fingerprint,
        "embedding_shape": list(embeddings.shape),
    }
    np.savez_compressed(
        embeddings_path,
        doc_ids=np.asarray(doc_ids),
        embeddings=embeddings,
    )
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return doc_ids, embeddings, False, embeddings_path, metadata


def retrieve_jina_code(
    *,
    queries: Iterable[dict[str, Any]],
    corpus: Iterable[dict[str, Any]],
    cache_dir: Path,
    model_name: str = MODEL_NAME,
    max_length: int = 512,
    corpus_batch_size: int = 2,
    query_batch_size: int = 4,
    similarity_batch_size: int = 64,
    top_k: int = 100,
    local_files_only: bool = False,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    query_rows = list(queries)
    corpus_rows = list(corpus)
    tokenizer, model, device, revision = load_jina_model(
        model_name=model_name,
        local_files_only=local_files_only,
    )
    doc_ids, corpus_embeddings, cache_hit, cache_path, cache_metadata = encode_corpus_cached(
        corpus_rows=corpus_rows,
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_name=model_name,
        model_revision=revision,
        max_length=max_length,
        batch_size=corpus_batch_size,
        cache_dir=cache_dir,
    )
    query_embeddings = encode_texts(
        texts=[row_text(row) for row in query_rows],
        tokenizer=tokenizer,
        model=model,
        device=device,
        instruction=QUERY_INSTRUCTION,
        max_length=max_length,
        batch_size=query_batch_size,
        desc="Jina encode queries",
    )

    top_k = min(top_k, len(doc_ids))
    results: dict[str, dict[str, float]] = {}
    corpus_matrix = corpus_embeddings.T
    for start in tqdm(range(0, len(query_rows), similarity_batch_size), desc="Jina top-k search"):
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
        "device": str(device),
        "torch_dtype": str(next(model.parameters()).dtype),
        "corpus_embedding_cache_hit": cache_hit,
        "corpus_embedding_cache_path": str(cache_path),
        "query_embedding_shape": list(query_embeddings.shape),
    }
    return results, metadata
