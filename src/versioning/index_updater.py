from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from src.versioning.chunk_manifest import load_chunk_manifest
from src.versioning.embedding_cache import ContentEmbeddingCache, EmbeddingConfig
from src.versioning.vector_index import VersionedVectorIndex


def build_version_index(
    *,
    chunk_manifest_path: str | Path,
    embedding_cache_dir: str | Path,
    index_dir: str | Path,
    embedding_config: EmbeddingConfig,
) -> dict[str, Any]:
    manifest = load_chunk_manifest(chunk_manifest_path)
    cache = ContentEmbeddingCache(embedding_cache_dir, embedding_config)
    index = VersionedVectorIndex(repo=manifest["repo"], index_dir=index_dir, embedding_config=embedding_config)
    report = index.build_from_chunk_manifest(chunk_manifest_path=chunk_manifest_path, embedding_cache=cache)
    load_started = time.perf_counter()
    reloaded = VersionedVectorIndex.load(index_dir, embedding_config)
    report["load_reload_time_seconds"] = time.perf_counter() - load_started
    report["reload_unique_vectors"] = reloaded.unique_vector_count()
    return report


def update_version_index(
    *,
    existing_index_dir: str | Path,
    target_chunk_manifest_path: str | Path,
    embedding_cache_dir: str | Path,
    embedding_config: EmbeddingConfig,
) -> dict[str, Any]:
    cache = ContentEmbeddingCache(embedding_cache_dir, embedding_config)
    index = VersionedVectorIndex.load(existing_index_dir, embedding_config)
    return index.update_to_chunk_manifest(chunk_manifest_path=target_chunk_manifest_path, embedding_cache=cache)


def save_index_report(report: dict[str, Any], output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output

