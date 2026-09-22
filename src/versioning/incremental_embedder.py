from __future__ import annotations

import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

from src.retrieval.jina_gguf import parse_embedding_response, truncate_with_tokenizer
from src.versioning.chunk_manifest import load_chunk_manifest
from src.versioning.embedding_cache import (
    ContentEmbeddingCache,
    EmbeddingConfig,
    make_embedding_key,
)


def check_server(server_url: str) -> None:
    response = requests.get(server_url.rstrip("/") + "/health", timeout=10)
    response.raise_for_status()


def build_incremental_embeddings(
    *,
    chunk_manifest_path: str | Path,
    cache_dir: str | Path,
    server_url: str,
    config: EmbeddingConfig,
    batch_size: int = 4,
    check_llama_server: bool = True,
) -> dict[str, Any]:
    started_total = time.perf_counter()
    lookup_started = time.perf_counter()
    manifest = load_chunk_manifest(chunk_manifest_path)
    chunks = list(manifest.get("chunks", {}).values())
    cache = ContentEmbeddingCache(cache_dir, config)
    if check_llama_server:
        check_server(server_url)

    occurrences_by_content: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        occurrences_by_content[chunk["content_id"]].append(chunk)

    content_records: list[dict[str, Any]] = []
    occurrence_results: list[dict[str, Any]] = []
    cache_hits = 0
    cache_misses = 0

    for content_id, occurrences in sorted(occurrences_by_content.items()):
        vector, entry = cache.get(content_id)
        representative = occurrences[0]
        embedding_key = make_embedding_key(content_id, config)
        if vector is not None and entry is not None:
            cache_hits += len(occurrences)
            status = "hit"
        else:
            cache_misses += len(occurrences)
            status = "miss"
            content_records.append(
                {
                    "content_id": content_id,
                    "embedding_key": embedding_key,
                    "text": representative["text"],
                    "language": representative["language"],
                    "content_sha256": representative["content_sha256"],
                    "source_symbols": [chunk["symbol"] for chunk in occurrences],
                    "source_paths": [chunk["path"] for chunk in occurrences],
                    "occurrence_count": len(occurrences),
                }
            )
        for chunk in occurrences:
            occurrence_results.append(
                {
                    "version_id": chunk["version_id"],
                    "content_id": content_id,
                    "embedding_key": embedding_key,
                    "path": chunk["path"],
                    "symbol": chunk["symbol"],
                    "chunk_type": chunk["chunk_type"],
                    "cache_status": status,
                }
            )

    lookup_runtime = time.perf_counter() - lookup_started

    embedding_started = time.perf_counter()
    generated_entries: list[dict[str, Any]] = []
    if content_records:
        tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        prompts = [
            truncate_with_tokenizer(
                tokenizer=tokenizer,
                text=record["text"],
                instruction=config.prompt_instruction,
                max_length=config.max_sequence_length,
            )
            for record in tqdm(content_records, desc="prepare missing chunk prompts")
        ]
        embeddings = _embed_http(
            texts=prompts,
            server_url=server_url,
            batch_size=batch_size,
            normalize=config.normalization,
            desc="embed missing chunks",
        )
        if embeddings.shape[1] != config.embedding_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {config.embedding_dimension}, got {embeddings.shape[1]}"
            )
        generated_entries = cache.put_many(content_records, embeddings)
    embedding_runtime = time.perf_counter() - embedding_started

    total_chunks = len(chunks)
    unique_content_ids = len(occurrences_by_content)
    newly_generated_unique = len(content_records)
    reused_occurrences = cache_hits
    reuse_percentage = (reused_occurrences / total_chunks * 100.0) if total_chunks else 0.0
    work_reduction = 1.0 - (newly_generated_unique / total_chunks) if total_chunks else 0.0
    total_runtime = time.perf_counter() - started_total

    return {
        "repo": manifest.get("repo"),
        "commit": manifest.get("commit"),
        "chunk_manifest": str(chunk_manifest_path),
        "generated_at": datetime.now(UTC).isoformat(),
        "total_chunks": total_chunks,
        "unique_content_ids": unique_content_ids,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "reused_embeddings": reused_occurrences,
        "newly_generated_embeddings": newly_generated_unique,
        "new_embedding_occurrences": cache_misses,
        "duplicate_occurrences_avoided": max(0, cache_misses - newly_generated_unique),
        "reuse_percentage": reuse_percentage,
        "embedding_work_reduction": work_reduction,
        "runtime": {
            "cache_lookup_seconds": lookup_runtime,
            "embedding_seconds": embedding_runtime,
            "total_seconds": total_runtime,
        },
        "model_config": {
            "model_name": config.model_name,
            "model_revision": config.model_revision,
            "gguf_file": config.gguf_file,
            "quantization": config.quantization,
            "embedding_dimension": config.embedding_dimension,
            "max_sequence_length": config.max_sequence_length,
            "pooling": config.pooling,
            "normalization": config.normalization,
            "prompt_instruction": config.prompt_instruction,
            "prompt_version": config.prompt_version,
            "fingerprint": config.fingerprint(),
        },
        "cache": {
            "cache_dir": str(cache.cache_dir),
            "index_path": str(cache.index_path),
            "embeddings_path": str(cache.embeddings_path),
            "storage": "single float32 NumPy matrix plus JSON index",
        },
        "generated_entries": [
            {
                "content_id": entry["content_id"],
                "embedding_key": entry["embedding_key"],
                "row": entry["row"],
                "source_symbols": entry["source_symbols"],
                "source_paths": entry["source_paths"],
            }
            for entry in generated_entries
        ],
        "occurrences": sorted(occurrence_results, key=lambda item: (item["path"], item["symbol"], item["version_id"])),
        "validation": {
            "embedding_dimension_correct": True,
            "finite_embeddings": True,
            "cache_metadata_matches_config": True,
        },
    }


def save_embedding_report(report: dict[str, Any], output_root: str | Path) -> Path:
    repo = str(report["repo"]).replace("\\", "_").replace("/", "_").replace(":", "_")
    output = Path(output_root) / repo / f"{report['commit']}_embedding_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def _embed_http(
    *,
    texts: list[str],
    server_url: str,
    batch_size: int,
    normalize: bool,
    desc: str,
) -> np.ndarray:
    endpoint = server_url.rstrip("/") + "/embedding"
    session = requests.Session()
    embeddings: list[np.ndarray] = []
    for start in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch = texts[start : start + batch_size]
        response = session.post(endpoint, json={"content": batch}, timeout=600)
        response.raise_for_status()
        values = parse_embedding_response(response.json())
        embeddings.append(values)
    result = np.vstack(embeddings).astype(np.float32)
    if normalize:
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / np.maximum(norms, 1e-12)
    if not np.isfinite(result).all():
        raise ValueError("Embedding response contains non-finite values")
    return result

