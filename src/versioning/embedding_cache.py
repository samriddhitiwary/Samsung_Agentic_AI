from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EmbeddingConfig:
    model_name: str = "jinaai/jina-code-embeddings-1.5b"
    model_revision: str | None = None
    gguf_file: str = "jina-code-embeddings-1.5b-Q8_0.gguf"
    quantization: str = "Q8_0"
    embedding_dimension: int = 1536
    max_sequence_length: int = 1024
    pooling: str = "last_token"
    normalization: bool = True
    prompt_instruction: str = "Candidate code snippet:\n"
    prompt_version: str = "jina_code_1.5b_passage_v1"

    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_embedding_key(content_id: str, config: EmbeddingConfig) -> str:
    payload = {
        "content_id": content_id,
        "model_config_fingerprint": config.fingerprint(),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ContentEmbeddingCache:
    """Append-friendly local embedding cache keyed by model-compatible content IDs.

    Storage:
      - embeddings.npy: dense float32 matrix, one row per embedding key
      - index.json: embedding_key -> row + metadata

    This avoids thousands of tiny vector files while keeping lookup simple.
    """

    def __init__(self, cache_dir: str | Path, config: EmbeddingConfig):
        self.cache_dir = Path(cache_dir)
        self.config = config
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_path = self.cache_dir / "embeddings.npy"
        self.index_path = self.cache_dir / "index.json"
        self.index = self._load_index()

    def get(self, content_id: str) -> tuple[np.ndarray | None, dict[str, Any] | None]:
        embedding_key = make_embedding_key(content_id, self.config)
        entry = self.index["entries"].get(embedding_key)
        if entry is None:
            return None, None
        if not self._entry_compatible(entry, content_id):
            return None, None
        embeddings = self._load_embeddings()
        row = int(entry["row"])
        if row >= embeddings.shape[0]:
            return None, None
        vector = embeddings[row].astype(np.float32)
        if not self.validate_vector(vector):
            return None, None
        return vector, entry

    def put_many(self, records: list[dict[str, Any]], embeddings: np.ndarray) -> list[dict[str, Any]]:
        if not records:
            return []
        if embeddings.shape != (len(records), self.config.embedding_dimension):
            raise ValueError(
                f"Embedding shape mismatch: got {embeddings.shape}, "
                f"expected {(len(records), self.config.embedding_dimension)}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("Cannot cache non-finite embeddings")

        current = self._load_embeddings()
        start_row = int(current.shape[0])
        combined = np.vstack([current, embeddings.astype(np.float32)])
        np.save(self.embeddings_path, combined)

        now = datetime.now(UTC).isoformat()
        written: list[dict[str, Any]] = []
        for offset, record in enumerate(records):
            content_id = record["content_id"]
            embedding_key = make_embedding_key(content_id, self.config)
            entry = {
                "row": start_row + offset,
                "content_id": content_id,
                "embedding_key": embedding_key,
                "model_config_fingerprint": self.config.fingerprint(),
                "model_name": self.config.model_name,
                "model_revision": self.config.model_revision,
                "gguf_file": self.config.gguf_file,
                "quantization": self.config.quantization,
                "embedding_dimension": self.config.embedding_dimension,
                "max_sequence_length": self.config.max_sequence_length,
                "pooling": self.config.pooling,
                "normalization": self.config.normalization,
                "prompt_instruction": self.config.prompt_instruction,
                "prompt_version": self.config.prompt_version,
                "created_at": now,
                "chunk_language": record.get("language"),
                "content_sha256": record.get("content_sha256"),
                "source_symbols": sorted(set(record.get("source_symbols", []))),
                "source_paths": sorted(set(record.get("source_paths", []))),
            }
            self.index["entries"][embedding_key] = entry
            written.append(entry)
        self.index["updated_at"] = now
        self.index["embedding_count"] = len(self.index["entries"])
        self.index["matrix_shape"] = [int(combined.shape[0]), int(combined.shape[1])]
        self._save_index()
        return written

    def validate_vector(self, vector: np.ndarray) -> bool:
        return (
            vector.ndim == 1
            and vector.shape[0] == self.config.embedding_dimension
            and bool(np.isfinite(vector).all())
        )

    def _entry_compatible(self, entry: dict[str, Any], content_id: str) -> bool:
        return (
            entry.get("content_id") == content_id
            and entry.get("model_config_fingerprint") == self.config.fingerprint()
            and entry.get("embedding_dimension") == self.config.embedding_dimension
            and entry.get("max_sequence_length") == self.config.max_sequence_length
            and entry.get("pooling") == self.config.pooling
            and entry.get("normalization") == self.config.normalization
        )

    def _load_index(self) -> dict[str, Any]:
        if self.index_path.exists():
            return json.loads(self.index_path.read_text(encoding="utf-8"))
        return {
            "cache_version": 1,
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": None,
            "storage": {
                "format": "numpy_matrix_plus_json_index",
                "embeddings_file": self.embeddings_path.name,
                "index_file": self.index_path.name,
            },
            "config": asdict(self.config),
            "config_fingerprint": self.config.fingerprint(),
            "embedding_count": 0,
            "matrix_shape": [0, self.config.embedding_dimension],
            "entries": {},
        }

    def _save_index(self) -> None:
        self.index_path.write_text(json.dumps(self.index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _load_embeddings(self) -> np.ndarray:
        if self.embeddings_path.exists():
            values = np.load(self.embeddings_path).astype(np.float32)
            if values.ndim != 2 or values.shape[1] != self.config.embedding_dimension:
                raise ValueError(f"Cache matrix has incompatible shape: {values.shape}")
            return values
        return np.empty((0, self.config.embedding_dimension), dtype=np.float32)

    def compatibility_probe(self, content_id: str, incompatible_config: EmbeddingConfig) -> bool:
        key = make_embedding_key(content_id, incompatible_config)
        entry = self.index["entries"].get(key)
        if entry is None:
            return False
        return entry.get("model_config_fingerprint") == incompatible_config.fingerprint()


def runtime_seconds(start: float) -> float:
    return time.perf_counter() - start

