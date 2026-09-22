from __future__ import annotations

import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.versioning.chunk_manifest import load_chunk_manifest
from src.versioning.embedding_cache import ContentEmbeddingCache, EmbeddingConfig


INDEX_VERSION = "p1_exact_numpy_index_v1"
SIMILARITY = "cosine/dot-product on normalized vectors"


class VersionedVectorIndex:
    """Persistent exact cosine index with version-aware occurrence filtering.

    Unique vectors are stored by content_id. Repository-version occurrences are
    stored separately by version_id and commit. Deletions are logical: an older
    occurrence can remain in metadata while no longer appearing in the target
    commit's active occurrence set.
    """

    def __init__(
        self,
        *,
        repo: str,
        index_dir: str | Path,
        embedding_config: EmbeddingConfig,
        vectors: np.ndarray | None = None,
        state: dict[str, Any] | None = None,
    ):
        self.repo = repo
        self.index_dir = Path(index_dir)
        self.embedding_config = embedding_config
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.vectors_path = self.index_dir / "vectors.npy"
        self.state_path = self.index_dir / "state.json"
        self.vectors = (
            vectors.astype(np.float32)
            if vectors is not None
            else np.empty((0, embedding_config.embedding_dimension), dtype=np.float32)
        )
        self.state = state if state is not None else self._empty_state()

    @classmethod
    def load(cls, index_dir: str | Path, embedding_config: EmbeddingConfig) -> "VersionedVectorIndex":
        started = time.perf_counter()
        index_dir = Path(index_dir)
        state = json.loads((index_dir / "state.json").read_text(encoding="utf-8"))
        vectors = np.load(index_dir / "vectors.npy").astype(np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != embedding_config.embedding_dimension:
            raise ValueError(f"Incompatible vector matrix shape: {vectors.shape}")
        if state["model_config_fingerprint"] != embedding_config.fingerprint():
            raise ValueError("Index model/config fingerprint does not match current embedding config")
        loaded = cls(
            repo=state["repo"],
            index_dir=index_dir,
            embedding_config=embedding_config,
            vectors=vectors,
            state=state,
        )
        loaded.state["last_load_time_seconds"] = time.perf_counter() - started
        return loaded

    def save(self) -> None:
        self._refresh_metadata()
        np.save(self.vectors_path, self.vectors.astype(np.float32))
        self.state.write_text if False else None
        self.state_path.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def build_from_chunk_manifest(
        self,
        *,
        chunk_manifest_path: str | Path,
        embedding_cache: ContentEmbeddingCache,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        manifest = load_chunk_manifest(chunk_manifest_path)
        self.state["active_commit"] = manifest["commit"]
        inserted = self._add_manifest_chunks(manifest=manifest, embedding_cache=embedding_cache)
        self.save()
        runtime = time.perf_counter() - started
        return {
            "repo": manifest["repo"],
            "commit": manifest["commit"],
            "total_chunk_occurrences": len(manifest.get("chunks", {})),
            "unique_content_ids": len({c["content_id"] for c in manifest.get("chunks", {}).values()}),
            "vectors_inserted": inserted,
            "active_occurrences": self.active_occurrence_count(manifest["commit"]),
            "unique_vectors": self.unique_vector_count(),
            "index_build_time_seconds": runtime,
            "persisted_index_size_bytes": self.persisted_size_bytes(),
            "index_dir": self.index_dir.as_posix(),
            "index_fingerprint": self.index_fingerprint(),
        }

    def update_to_chunk_manifest(
        self,
        *,
        chunk_manifest_path: str | Path,
        embedding_cache: ContentEmbeddingCache,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        source_commit = self.state.get("active_commit")
        manifest = load_chunk_manifest(chunk_manifest_path)
        target_commit = manifest["commit"]

        old_active = self.active_occurrences(source_commit) if source_commit else {}
        new_chunks = manifest.get("chunks", {})
        old_content_ids = {item["content_id"] for item in old_active.values()}
        new_content_ids = {item["content_id"] for item in new_chunks.values()}
        old_by_occurrence = {_occurrence_key(item): item for item in old_active.values()}
        new_by_occurrence = {_occurrence_key(item): item for item in new_chunks.values()}
        shared_occurrence_keys = set(old_by_occurrence) & set(new_by_occurrence)
        unchanged_occurrences = sum(
            1
            for key in shared_occurrence_keys
            if old_by_occurrence[key]["content_id"] == new_by_occurrence[key]["content_id"]
        )
        modified_replaced_occurrences = sum(
            1
            for key in shared_occurrence_keys
            if old_by_occurrence[key]["content_id"] != new_by_occurrence[key]["content_id"]
        )
        added_occurrences = len(set(new_by_occurrence) - set(old_by_occurrence)) + modified_replaced_occurrences
        removed_occurrences = len(set(old_by_occurrence) - set(new_by_occurrence)) + modified_replaced_occurrences

        reused_vectors = len(new_content_ids & set(self.state["content_id_to_row"]))
        inserted = self._add_manifest_chunks(manifest=manifest, embedding_cache=embedding_cache)

        self.state["active_commit"] = target_commit
        self.save()
        runtime = time.perf_counter() - started
        tombstoned = sum(
            1
            for content_id in old_content_ids - new_content_ids
            if not self._content_active_in_commit(content_id, target_commit)
        )
        return {
            "source_commit": source_commit,
            "target_commit": target_commit,
            "unchanged_occurrences": unchanged_occurrences,
            "modified_replaced_occurrences": modified_replaced_occurrences,
            "added_occurrences": added_occurrences,
            "removed_occurrences": removed_occurrences,
            "reused_vectors": reused_vectors,
            "vectors_newly_inserted": inserted,
            "vectors_tombstoned": tombstoned,
            "active_vectors_after_update": self.active_vector_count(target_commit),
            "active_occurrences_after_update": self.active_occurrence_count(target_commit),
            "unique_vectors_total": self.unique_vector_count(),
            "update_time_seconds": runtime,
            "persisted_index_size_bytes": self.persisted_size_bytes(),
            "index_fingerprint": self.index_fingerprint(),
        }

    def search(
        self,
        *,
        query_embedding: np.ndarray,
        commit: str,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        if query_embedding.ndim != 1:
            raise ValueError("query_embedding must be a 1D vector")
        active = list(self.active_occurrences(commit).values())
        if not active:
            return []
        active_content_ids = sorted({chunk["content_id"] for chunk in active})
        rows = [self.state["content_id_to_row"][content_id] for content_id in active_content_ids]
        matrix = self.vectors[np.asarray(rows, dtype=np.int64)]
        scores = matrix @ query_embedding.astype(np.float32)
        score_by_content = {content_id: float(score) for content_id, score in zip(active_content_ids, scores)}

        results: list[dict[str, Any]] = []
        for occurrence in active:
            results.append(
                {
                    "repo": occurrence["repo"],
                    "commit": commit,
                    "path": occurrence["path"],
                    "symbol": occurrence["symbol"],
                    "chunk_type": occurrence["chunk_type"],
                    "content_id": occurrence["content_id"],
                    "version_id": occurrence["version_id"],
                    "score": score_by_content[occurrence["content_id"]],
                }
            )
        results.sort(key=lambda item: (-item["score"], item["path"], item["symbol"], item["version_id"]))
        return results[:top_k]

    def active_occurrences(self, commit: str | None) -> dict[str, dict[str, Any]]:
        if commit is None:
            return {}
        version_ids = self.state["commit_to_version_ids"].get(commit, [])
        return {
            version_id: self.state["version_id_to_occurrence"][version_id]
            for version_id in version_ids
            if version_id in self.state["version_id_to_occurrence"]
        }

    def active_occurrence_count(self, commit: str) -> int:
        return len(self.active_occurrences(commit))

    def active_vector_count(self, commit: str) -> int:
        return len({item["content_id"] for item in self.active_occurrences(commit).values()})

    def unique_vector_count(self) -> int:
        return len(self.state["content_id_to_row"])

    def persisted_size_bytes(self) -> int:
        total = 0
        for path in (self.vectors_path, self.state_path):
            if path.exists():
                total += path.stat().st_size
        return total

    def index_fingerprint(self) -> str:
        payload = {
            "repo": self.state["repo"],
            "index_version": self.state["index_version"],
            "model_config_fingerprint": self.state["model_config_fingerprint"],
            "content_id_to_row": self.state["content_id_to_row"],
            "commit_to_version_ids": self.state["commit_to_version_ids"],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    def _add_manifest_chunks(self, *, manifest: dict[str, Any], embedding_cache: ContentEmbeddingCache) -> int:
        inserted = 0
        commit = manifest["commit"]
        commit_version_ids: list[str] = []
        for chunk in sorted(manifest.get("chunks", {}).values(), key=lambda c: (c["path"], c["start_line"], c["version_id"])):
            content_id = chunk["content_id"]
            if content_id not in self.state["content_id_to_row"]:
                vector, cache_entry = embedding_cache.get(content_id)
                if vector is None or cache_entry is None:
                    raise KeyError(f"Missing compatible cached embedding for content_id={content_id}")
                row = int(self.vectors.shape[0])
                self.vectors = np.vstack([self.vectors, vector.reshape(1, -1).astype(np.float32)])
                self.state["content_id_to_row"][content_id] = row
                self.state["content_metadata"][content_id] = {
                    "content_id": content_id,
                    "content_sha256": chunk["content_sha256"],
                    "embedding_key": cache_entry["embedding_key"],
                    "row": row,
                    "language": chunk["language"],
                }
                inserted += 1

            occurrence = {
                "repo": chunk["repo"],
                "commit": chunk["commit"],
                "path": chunk["path"],
                "symbol": chunk["symbol"],
                "chunk_type": chunk["chunk_type"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "content_id": content_id,
                "version_id": chunk["version_id"],
            }
            self.state["version_id_to_occurrence"][chunk["version_id"]] = occurrence
            commit_version_ids.append(chunk["version_id"])
        self.state["commit_to_version_ids"][commit] = commit_version_ids
        return inserted

    def _content_active_in_commit(self, content_id: str, commit: str) -> bool:
        return any(item["content_id"] == content_id for item in self.active_occurrences(commit).values())

    def _refresh_metadata(self) -> None:
        now = datetime.now(UTC).isoformat()
        self.state["updated_at"] = now
        self.state["vector_dimension"] = self.embedding_config.embedding_dimension
        self.state["number_of_unique_vectors"] = self.unique_vector_count()
        active_commit = self.state.get("active_commit")
        self.state["number_of_active_occurrences"] = (
            self.active_occurrence_count(active_commit) if active_commit else 0
        )
        self.state["number_of_active_vectors"] = self.active_vector_count(active_commit) if active_commit else 0
        self.state["index_fingerprint"] = self.index_fingerprint()

    def _empty_state(self) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat()
        return {
            "repo": self.repo,
            "active_commit": None,
            "index_version": INDEX_VERSION,
            "created_at": now,
            "updated_at": now,
            "model_config_fingerprint": self.embedding_config.fingerprint(),
            "vector_dimension": self.embedding_config.embedding_dimension,
            "similarity_metric": SIMILARITY,
            "storage": {
                "vectors_file": self.vectors_path.name,
                "state_file": self.state_path.name,
                "format": "exact_numpy_matrix_with_json_metadata",
            },
            "content_id_to_row": {},
            "content_metadata": {},
            "version_id_to_occurrence": {},
            "commit_to_version_ids": {},
            "number_of_unique_vectors": 0,
            "number_of_active_occurrences": 0,
            "number_of_active_vectors": 0,
        }


def repo_index_dir(root: str | Path, repo: str, suffix: str | None = None) -> Path:
    safe_repo = repo.replace("\\", "_").replace("/", "_").replace(":", "_")
    name = safe_repo if suffix is None else f"{safe_repo}_{suffix}"
    return Path(root) / name


def _occurrence_key(chunk: dict[str, Any]) -> str:
    return f"{chunk['path']}:{chunk['chunk_type']}:{chunk['symbol']}"
