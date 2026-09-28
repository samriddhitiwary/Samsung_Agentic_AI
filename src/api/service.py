from __future__ import annotations

import json
import os
import subprocess
import time
from collections import OrderedDict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from fastapi import HTTPException
from transformers import AutoTokenizer

from src.agentic.controller import run_agentic_query
from src.retrieval.jina_code import QUERY_INSTRUCTION
from src.retrieval.jina_gguf import truncate_with_tokenizer
from src.structure.graph import build_structural_graph, load_structural_graph, structural_graph_output_path
from src.versioning.chunk_manifest import (
    build_chunk_manifest,
    chunk_manifest_output_path,
    compare_chunk_manifests,
    load_chunk_manifest,
    save_chunk_manifest,
)
from src.versioning.embedding_cache import EmbeddingConfig
from src.versioning.evolution_search import (
    build_evolution_metadata,
    load_evolution_metadata,
    search_across_versions,
)
from src.versioning.git_scanner import GitScannerError, ScannerConfig, scan_git_repository
from src.versioning.incremental_embedder import _embed_http, build_incremental_embeddings
from src.versioning.index_updater import build_version_index, update_version_index
from src.versioning.manifest import compare_manifests, load_manifest, manifest_output_path, save_manifest
from src.versioning.vector_index import VersionedVectorIndex


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SERVER_URL = os.environ.get("JINA_EMBEDDING_SERVER_URL", "http://127.0.0.1:8081")


class ApiService:
    """Thin orchestration over the existing versioning/retrieval modules."""

    def __init__(
        self,
        *,
        registry_path: str | Path | None = None,
        server_url: str = DEFAULT_SERVER_URL,
    ) -> None:
        self.registry_path = Path(registry_path or PROJECT_ROOT / "data/api/registry.json")
        self.server_url = server_url
        self.embedding_config = EmbeddingConfig()
        self.artifact_root = PROJECT_ROOT / "data/api/repos"
        self._tokenizer: AutoTokenizer | None = None
        self._http = requests.Session()
        self._query_embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._query_embedding_cache_limit = int(os.environ.get("QUERY_EMBEDDING_CACHE_SIZE", "128"))
        self._index_cache: dict[tuple[str, str], VersionedVectorIndex] = {}
        self._chunk_manifest_cache: dict[str, dict[str, Any]] = {}
        self._graph_cache: dict[str, dict[str, Any]] = {}

    def health(self) -> dict[str, Any]:
        registry = self._load_registry()
        model_file = PROJECT_ROOT / "data/cache/model_files" / self.embedding_config.gguf_file
        llama_exe = Path(
            os.environ.get(
                "LLAMA_SERVER_EXE",
                r"C:\Users\samri\AppData\Local\Microsoft\WinGet\Packages\ggml.llamacpp_Microsoft.Winget.Source_8wekyb3d8bbwe\llama-server.exe",
            )
        )
        return {
            "status": "ok",
            "model": {
                "name": self.embedding_config.model_name,
                "gguf_file": self.embedding_config.gguf_file,
                "local_file_available": model_file.exists(),
            },
            "llama_cpp": {
                "server_url": self.server_url,
                "server_healthy": self._server_healthy(timeout=1.0),
                "executable_available": llama_exe.exists(),
            },
            "cache": {
                "artifact_root_available": self.artifact_root.exists(),
            },
            "registered_repository_count": len(registry.get("repos", {})),
        }

    def register_repo(self, *, repo_path: str, repo_id: str | None, commit: str) -> dict[str, Any]:
        started = time.perf_counter()
        self._require_server()
        repo = self._validate_repo_path(repo_path)
        resolved_repo_id = repo_id or repo.name
        resolved_commit = self._resolve_commit(repo, commit)
        paths = self._paths(resolved_repo_id)

        manifest_path, file_manifest = self._build_or_load_manifest(repo, resolved_repo_id, resolved_commit, paths)
        chunk_path, chunk_manifest = self._build_or_load_chunk_manifest(repo, resolved_repo_id, resolved_commit, manifest_path, paths)

        embedding_report = build_incremental_embeddings(
            chunk_manifest_path=chunk_path,
            cache_dir=paths["embedding_cache"],
            server_url=self.server_url,
            config=self.embedding_config,
            batch_size=4,
        )

        index_report = build_version_index(
            chunk_manifest_path=chunk_path,
            embedding_cache_dir=paths["embedding_cache"],
            index_dir=paths["index"],
            embedding_config=self.embedding_config,
        )

        evolution = build_evolution_metadata(
            repo=resolved_repo_id,
            commit_order=[resolved_commit],
            chunk_manifest_paths=[chunk_path],
            output_dir=paths["evolution"],
        )
        graph_path, graph = self._build_or_load_structural_graph(resolved_repo_id, resolved_commit, chunk_path, paths)

        registry = self._load_registry()
        registry["repos"][resolved_repo_id] = {
            "repo_id": resolved_repo_id,
            "repo_path": str(repo),
            "commits": [resolved_commit],
            "active_commit": resolved_commit,
            "artifact_root": self._rel(paths["root"]),
            "manifest_paths": {resolved_commit: self._rel(manifest_path)},
            "chunk_manifest_paths": {resolved_commit: self._rel(chunk_path)},
            "structural_graph_paths": {resolved_commit: self._rel(graph_path)},
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._save_registry(registry)

        return {
            "repo": resolved_repo_id,
            "commit": resolved_commit,
            "source_files": len(file_manifest.get("files", {})),
            "chunks": len(chunk_manifest.get("chunks", {})),
            "symbols": graph["stats"]["symbols"],
            "call_edges": graph["stats"]["call_edges"],
            "reference_edges": graph["stats"]["reference_edges"],
            "loc": self._count_loc(chunk_manifest),
            "reused_embeddings": embedding_report["reused_embeddings"],
            "newly_generated_embeddings": embedding_report["newly_generated_embeddings"],
            "index": {
                "active_vectors": index_report["unique_vectors"],
                "active_occurrences": index_report["active_occurrences"],
            },
            "evolution_chains": evolution["metrics"]["evolution_chains"],
            "indexing_stages": {
                "git_file_scan_seconds": file_manifest.get("stats", {}).get("scan_time_seconds"),
                "chunk_generation_and_js_parse_seconds": chunk_manifest.get("stats", {}).get("build_time_seconds"),
                "structural_graph_seconds": graph.get("stats", {}).get("build_time_seconds"),
                "embedding_seconds": embedding_report.get("runtime", {}).get("embedding_seconds"),
                "embedding_cache_lookup_seconds": embedding_report.get("runtime", {}).get("cache_lookup_seconds"),
                "vector_index_persistence_seconds": index_report.get("index_build_time_seconds"),
                "total_seconds": time.perf_counter() - started,
            },
            "indexing_runtime_seconds": time.perf_counter() - started,
        }

    def update_repo(self, *, repo_id: str, commit: str) -> dict[str, Any]:
        started = time.perf_counter()
        self._require_server()
        repo_record = self._repo_record(repo_id)
        repo = self._validate_repo_path(repo_record["repo_path"])
        previous_commit = repo_record["active_commit"]
        new_commit = self._resolve_commit(repo, commit)
        if new_commit == previous_commit:
            return {
                "repo": repo_id,
                "previous_commit": previous_commit,
                "new_commit": new_commit,
                "message": "Repository is already indexed at this commit.",
                "update_runtime_seconds": 0.0,
            }

        paths = self._paths(repo_id)
        manifest_path, file_manifest = self._build_or_load_manifest(repo, repo_id, new_commit, paths)
        chunk_path, chunk_manifest = self._build_or_load_chunk_manifest(repo, repo_id, new_commit, manifest_path, paths)

        old_manifest = load_manifest(self._abs(repo_record["manifest_paths"][previous_commit]))
        old_chunk_manifest = load_chunk_manifest(self._abs(repo_record["chunk_manifest_paths"][previous_commit]))
        file_diff = compare_manifests(old_manifest, file_manifest)
        chunk_diff = compare_chunk_manifests(
            old_chunk_manifest,
            chunk_manifest,
            old_file_manifest=old_manifest,
            new_file_manifest=file_manifest,
        )

        embedding_report = build_incremental_embeddings(
            chunk_manifest_path=chunk_path,
            cache_dir=paths["embedding_cache"],
            server_url=self.server_url,
            config=self.embedding_config,
            batch_size=4,
        )
        index_report = update_version_index(
            existing_index_dir=paths["index"],
            target_chunk_manifest_path=chunk_path,
            embedding_cache_dir=paths["embedding_cache"],
            embedding_config=self.embedding_config,
        )
        graph_path, graph = self._build_or_load_structural_graph(repo_id, new_commit, chunk_path, paths)

        commits = list(repo_record["commits"])
        if new_commit not in commits:
            commits.append(new_commit)
        repo_record["commits"] = commits
        repo_record["active_commit"] = new_commit
        repo_record["manifest_paths"][new_commit] = self._rel(manifest_path)
        repo_record["chunk_manifest_paths"][new_commit] = self._rel(chunk_path)
        repo_record.setdefault("structural_graph_paths", {})[new_commit] = self._rel(graph_path)
        repo_record["updated_at"] = datetime.now(UTC).isoformat()
        self._write_repo_record(repo_id, repo_record)

        build_evolution_metadata(
            repo=repo_id,
            commit_order=commits,
            chunk_manifest_paths=[self._abs(repo_record["chunk_manifest_paths"][c]) for c in commits],
            output_dir=paths["evolution"],
        )

        return {
            "repo": repo_id,
            "previous_commit": previous_commit,
            "new_commit": new_commit,
            "changed_files": file_diff["counts"],
            "reusable_chunks": chunk_diff["counts"]["reused_chunks"],
            "chunk_reuse_percent": chunk_diff["reuse_percentage"],
            "embeddings_reused": embedding_report["reused_embeddings"],
            "embeddings_generated": embedding_report["newly_generated_embeddings"],
            "embedding_reuse_percent": embedding_report["reuse_percentage"],
            "vectors_added": index_report["vectors_newly_inserted"],
            "vectors_removed_tombstoned": index_report["vectors_tombstoned"],
            "active_vectors": index_report["active_vectors_after_update"],
            "active_occurrences": index_report["active_occurrences_after_update"],
            "symbols": graph["stats"]["symbols"],
            "call_edges": graph["stats"]["call_edges"],
            "reference_edges": graph["stats"]["reference_edges"],
            "indexing_stages": {
                "git_file_scan_seconds": file_manifest.get("stats", {}).get("scan_time_seconds"),
                "chunk_generation_and_js_parse_seconds": chunk_manifest.get("stats", {}).get("build_time_seconds"),
                "structural_graph_seconds": graph.get("stats", {}).get("build_time_seconds"),
                "embedding_seconds": embedding_report.get("runtime", {}).get("embedding_seconds"),
                "embedding_cache_lookup_seconds": embedding_report.get("runtime", {}).get("cache_lookup_seconds"),
                "vector_index_update_seconds": index_report.get("update_time_seconds"),
                "total_seconds": time.perf_counter() - started,
            },
            "update_runtime_seconds": time.perf_counter() - started,
        }

    def search(self, *, repo_id: str, query: str, commit: str, top_k: int) -> dict[str, Any]:
        self._require_server()
        record = self._repo_record(repo_id)
        resolved_commit = self._resolve_commit(Path(record["repo_path"]), commit)
        if resolved_commit not in record["commits"]:
            raise HTTPException(status_code=404, detail=f"Commit is not indexed for repo '{repo_id}': {resolved_commit}")
        paths = self._paths(repo_id)
        try:
            index = self._load_index(repo_id, resolved_commit, paths["index"])
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Could not load persisted index: {exc}") from exc
        metadata = load_evolution_metadata(paths["evolution"])
        vector, _ = self._embed_query_profile(query)
        raw = index.search(query_embedding=vector, commit=resolved_commit, top_k=top_k)
        results = []
        for rank, item in enumerate(raw, start=1):
            state = metadata["content_states"].get(item["content_id"], {})
            results.append(
                {
                    "rank": rank,
                    "score": item["score"],
                    "path": item["path"],
                    "symbol": item["symbol"],
                    "chunk_type": item["chunk_type"],
                    "content_preview": state.get("preview"),
                    "commit": resolved_commit,
                    "content_id": item["content_id"],
                    "version_id": item["version_id"],
                }
            )
        return {"repo": repo_id, "commit": resolved_commit, "query": query, "results": results}

    def query(self, *, repo_id: str, query: str, commit: str | None, top_k: int) -> dict[str, Any]:
        self._require_server()
        record = self._repo_record(repo_id)
        resolved_commit = self._resolve_commit(Path(record["repo_path"]), commit or record.get("active_commit") or "HEAD")
        if resolved_commit not in record["commits"]:
            raise HTTPException(status_code=404, detail=f"Commit is not indexed for repo '{repo_id}': {resolved_commit}")
        chunk_path = self._abs(record["chunk_manifest_paths"][resolved_commit])
        chunk_manifest = self._load_chunk_manifest_cached(chunk_path)
        graph = self._load_graph_for_record(record, resolved_commit)

        def semantic_search(search_query: str, search_commit: str, broad_top_k: int) -> tuple[list[dict[str, Any]], dict[str, float | str | bool | None]]:
            return self._semantic_search_profile(repo_id=repo_id, query=search_query, commit=search_commit, top_k=broad_top_k)

        payload = run_agentic_query(
            repo_id=repo_id,
            commit=resolved_commit,
            query=query,
            top_k=top_k,
            chunk_manifest_path=chunk_path,
            chunk_manifest=chunk_manifest,
            graph=graph,
            semantic_search=semantic_search,
        )
        return payload

    def evolution_search(
        self,
        *,
        repo_id: str,
        query: str,
        start_commit: str | None,
        end_commit: str | None,
        top_k: int,
        include_evolution_context: bool = False,
    ) -> dict[str, Any]:
        self._require_server()
        record = self._repo_record(repo_id)
        repo = Path(record["repo_path"])
        resolved_start = self._resolve_commit(repo, start_commit) if start_commit else None
        resolved_end = self._resolve_commit(repo, end_commit) if end_commit else None
        vector = self._embed_query(query)
        try:
            payload = search_across_versions(
                index_dir=self._paths(repo_id)["index"],
                evolution_dir=self._paths(repo_id)["evolution"],
                query_embedding=vector,
                start_commit=resolved_start,
                end_commit=resolved_end,
                top_k=top_k,
                raw=False,
                include_evolution_context=include_evolution_context,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        results = []
        for rank, item in enumerate(payload["results"], start=1):
            occurrences = item.get("occurrences", [])
            commits = [occ["commit"] for occ in occurrences]
            results.append(
                {
                    "rank": rank,
                    "content_id": item["content_id"],
                    "score": item["score"],
                    "symbol": item["best_match"]["symbol"],
                    "path": item["best_match"]["path"],
                    "preview": item.get("preview"),
                    "occurrences": occurrences,
                    "first_seen_commit": commits[0] if commits else None,
                    "last_seen_commit": commits[-1] if commits else None,
                    "changed_state_count": len({occ["version_id"] for occ in occurrences}),
                    "lineage": item.get("lineage", []),
                    "evolution_context": item.get("evolution_context"),
                }
            )
        return {
            "repo": repo_id,
            "query": query,
            "commits": payload["commits"],
            "results": results,
            "runtime": payload["runtime"],
        }

    def symbol_evolution(self, *, repo_id: str, symbol: str, path: str | None) -> dict[str, Any]:
        self._repo_record(repo_id)
        metadata = load_evolution_metadata(self._paths(repo_id)["evolution"])
        matches = []
        for key, chain in metadata["chains"].items():
            present_states = [s for s in chain["states"] if s.get("symbol") == symbol]
            if not present_states:
                continue
            if path and not any(s.get("path") == path for s in present_states):
                continue
            matches.append(chain)
        if not matches:
            raise HTTPException(status_code=404, detail=f"No evolution chain found for symbol '{symbol}'.")
        chain = matches[0]
        return {
            "repo": repo_id,
            "symbol": symbol,
            "path": path,
            "occurrence_key": chain["occurrence_key"],
            "states": chain["states"],
            "transitions": chain["transitions"],
        }

    def metrics_summary(self) -> dict[str, Any]:
        p1_report_path = PROJECT_ROOT / "data/versioning/benchmarks/itsdangerous/p1_real_repo_benchmark.json"
        p0_metrics_path = PROJECT_ROOT / "results/jina_code_1.5b_full_1024/metrics.json"
        p0_values = {
            "NDCG@10": 0.86950,
            "MRR@10": 0.84141,
            "HitRate@10": 0.95564,
            "Recall@100": 0.99097,
        }
        p0_source = "frozen verified values fallback"
        if p0_metrics_path.exists():
            p0_payload = json.loads(p0_metrics_path.read_text(encoding="utf-8"))
            metrics = p0_payload.get("metrics", {})
            p0_values = {
                "NDCG@10": metrics.get("ndcg_at_10", p0_values["NDCG@10"]),
                "MRR@10": metrics.get("mrr_at_10", p0_values["MRR@10"]),
                "HitRate@10": metrics.get("hitrate_at_10", p0_values["HitRate@10"]),
                "Recall@100": metrics.get("recall_at_100", p0_values["Recall@100"]),
            }
            p0_source = self._rel(p0_metrics_path)
        p1 = None
        if p1_report_path.exists():
            report = json.loads(p1_report_path.read_text(encoding="utf-8"))
            p1 = {
                "repository": report["repository"]["name"],
                "average_chunk_reuse": report["aggregate_metrics"]["average_chunk_reuse_percent"],
                "average_embedding_reuse": report["aggregate_metrics"]["average_embedding_reuse_percent"],
                "measured_incremental_speedups": report["aggregate_metrics"]["speedups"],
                "max_speedup": report["aggregate_metrics"]["maximum_speedup"],
                "embedding_work_saved": report["aggregate_metrics"]["embedding_work_saved_clean_comparison"],
                "retrieval_parity": report["retrieval_parity"]["passed"],
                "restart_resume_parity": report["restart_resume"]["passed"],
                "source": self._rel(p1_report_path),
            }
        return {
            "p0": {
                "source": p0_source,
                "jina_code_1_5b_q8_full_retrieval": p0_values,
            },
            "p1": p1,
        }

    def _embed_query(self, query: str) -> np.ndarray:
        vector, _ = self._embed_query_profile(query)
        return vector

    def _embed_query_profile(self, query: str) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        cache_key = self._query_cache_key(query)
        cached = self._query_embedding_cache.get(cache_key)
        if cached is not None:
            self._query_embedding_cache.move_to_end(cache_key)
            return cached.copy(), {
                "query_embedding_ms": (time.perf_counter() - started) * 1000.0,
                "query_embedding_cache": "hit",
            }
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.embedding_config.model_name)
        prompt = truncate_with_tokenizer(
            tokenizer=self._tokenizer,
            text=query,
            instruction=QUERY_INSTRUCTION,
            max_length=self.embedding_config.max_sequence_length,
        )
        endpoint = self.server_url.rstrip("/") + "/embedding"
        response = self._http.post(endpoint, json={"content": [prompt]}, timeout=600)
        response.raise_for_status()
        from src.retrieval.jina_gguf import parse_embedding_response

        result = parse_embedding_response(response.json()).astype(np.float32)
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        result = result / np.maximum(norms, 1e-12)
        if result.shape[1] != self.embedding_config.embedding_dimension:
            raise HTTPException(status_code=500, detail=f"Embedding dimension mismatch: {result.shape}")
        if not np.isfinite(result).all():
            raise HTTPException(status_code=500, detail="Embedding response contains non-finite values")
        vector = result[0]
        self._query_embedding_cache[cache_key] = vector.copy()
        self._query_embedding_cache.move_to_end(cache_key)
        while len(self._query_embedding_cache) > self._query_embedding_cache_limit:
            self._query_embedding_cache.popitem(last=False)
        return vector, {
            "query_embedding_ms": (time.perf_counter() - started) * 1000.0,
            "query_embedding_cache": "miss",
        }

    def _semantic_search_profile(
        self,
        *,
        repo_id: str,
        query: str,
        commit: str,
        top_k: int,
    ) -> tuple[list[dict[str, Any]], dict[str, float | str | bool | None]]:
        started_total = time.perf_counter()
        paths = self._paths(repo_id)
        index_started = time.perf_counter()
        try:
            index = self._load_index(repo_id, commit, paths["index"])
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Could not load persisted index: {exc}") from exc
        index_ms = (time.perf_counter() - index_started) * 1000.0

        vector, embed_profile = self._embed_query_profile(query)
        search_started = time.perf_counter()
        raw = index.search(query_embedding=vector, commit=commit, top_k=top_k)
        search_ms = (time.perf_counter() - search_started) * 1000.0
        results = [
            {
                "rank": rank,
                "score": item["score"],
                "path": item["path"],
                "symbol": item["symbol"],
                "chunk_type": item["chunk_type"],
                "commit": commit,
                "content_id": item["content_id"],
                "version_id": item["version_id"],
            }
            for rank, item in enumerate(raw, start=1)
        ]
        total_ms = (time.perf_counter() - started_total) * 1000.0
        return results, {
            "semantic_search_ms": total_ms,
            "query_embedding_ms": embed_profile["query_embedding_ms"],
            "query_embedding_cache": embed_profile["query_embedding_cache"],
            "vector_index_access_ms": index_ms,
            "vector_search_ms": search_ms,
        }

    def _build_or_load_manifest(self, repo: Path, repo_id: str, commit: str, paths: dict[str, Path]) -> tuple[Path, dict[str, Any]]:
        manifest_path = manifest_output_path(paths["manifests"], repo_id, commit)
        if manifest_path.exists():
            return manifest_path, load_manifest(manifest_path)
        try:
            manifest = scan_git_repository(repo_path=repo, commit_ref=commit, repo_id=repo_id, config=ScannerConfig())
        except GitScannerError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        save_manifest(manifest, manifest_path)
        if not manifest.get("files"):
            raise HTTPException(status_code=400, detail="No supported source files were found for indexing.")
        return manifest_path, manifest

    def _build_or_load_chunk_manifest(
        self,
        repo: Path,
        repo_id: str,
        commit: str,
        manifest_path: Path,
        paths: dict[str, Path],
    ) -> tuple[Path, dict[str, Any]]:
        chunk_path = chunk_manifest_output_path(paths["chunks"], repo_id, commit)
        if chunk_path.exists():
            return chunk_path, load_chunk_manifest(chunk_path)
        chunk_manifest = build_chunk_manifest(repo_path=repo, file_manifest_path=manifest_path)
        save_chunk_manifest(chunk_manifest, chunk_path)
        if not chunk_manifest.get("chunks"):
            raise HTTPException(status_code=400, detail="No indexable code chunks were produced.")
        return chunk_path, chunk_manifest

    def _build_or_load_structural_graph(
        self,
        repo_id: str,
        commit: str,
        chunk_path: Path,
        paths: dict[str, Path],
    ) -> tuple[Path, dict[str, Any]]:
        graph_path = structural_graph_output_path(paths["graphs"], repo_id, commit)
        if graph_path.exists():
            return graph_path, self._load_graph(graph_path)
        graph = build_structural_graph(chunk_manifest_path=chunk_path, output_dir=paths["graphs"])
        self._graph_cache[str(graph_path.resolve())] = graph
        return graph_path, graph

    def _load_graph_for_record(self, record: dict[str, Any], commit: str) -> dict[str, Any] | None:
        graph_rel = record.get("structural_graph_paths", {}).get(commit)
        if graph_rel:
            path = self._abs(graph_rel)
            if path.exists():
                return self._load_graph(path)
        fallback = structural_graph_output_path(self._paths(record["repo_id"])["graphs"], record["repo_id"], commit)
        if fallback.exists():
            return self._load_graph(fallback)
        return None

    def _load_graph(self, path: Path) -> dict[str, Any]:
        key = str(path.resolve())
        graph = self._graph_cache.get(key)
        if graph is None:
            graph = load_structural_graph(path)
            self._graph_cache[key] = graph
        return graph

    def _load_chunk_manifest_cached(self, path: Path) -> dict[str, Any]:
        key = str(path.resolve())
        manifest = self._chunk_manifest_cache.get(key)
        if manifest is None:
            manifest = load_chunk_manifest(path)
            self._chunk_manifest_cache[key] = manifest
        return manifest

    def _load_index(self, repo_id: str, commit: str, index_dir: Path) -> VersionedVectorIndex:
        key = (repo_id, commit)
        index = self._index_cache.get(key)
        if index is None:
            index = VersionedVectorIndex.load(index_dir, self.embedding_config)
            self._index_cache[key] = index
        return index

    def _query_cache_key(self, query: str) -> str:
        normalized = " ".join(query.strip().split()).lower()
        return json.dumps(
            {
                "query": normalized,
                "model": self.embedding_config.fingerprint(),
                "prompt_version": self.embedding_config.prompt_version,
                "instruction": QUERY_INSTRUCTION,
            },
            sort_keys=True,
        )

    def _validate_repo_path(self, repo_path: str) -> Path:
        repo = Path(repo_path)
        if not repo.is_absolute():
            repo = PROJECT_ROOT / repo
        repo = repo.resolve()
        if not repo.exists():
            raise HTTPException(status_code=400, detail=f"Repository path does not exist: {repo_path}")
        if not (repo / ".git").exists():
            raise HTTPException(status_code=400, detail=f"Path is not a Git repository: {repo_path}")
        return repo

    def _resolve_commit(self, repo: Path, ref: str | None) -> str:
        try:
            result = subprocess.run(
                ["git", "-C", str(repo), "rev-parse", ref or "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            )
            return result.stdout.strip()
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() or f"Invalid Git ref: {ref}"
            raise HTTPException(status_code=400, detail=detail) from exc

    def _paths(self, repo_id: str) -> dict[str, Path]:
        root = self.artifact_root / self._safe(repo_id)
        return {
            "root": root,
            "manifests": root / "manifests",
            "chunks": root / "chunks",
            "embedding_cache": root / "embedding_cache",
            "index": root / "index",
            "evolution": root / "evolution",
            "graphs": root / "graphs",
        }

    def _repo_record(self, repo_id: str) -> dict[str, Any]:
        registry = self._load_registry()
        record = registry.get("repos", {}).get(repo_id)
        if record is None:
            raise HTTPException(status_code=404, detail=f"Repository is not registered: {repo_id}")
        return record

    def _write_repo_record(self, repo_id: str, record: dict[str, Any]) -> None:
        registry = self._load_registry()
        registry.setdefault("repos", {})[repo_id] = record
        self._save_registry(registry)

    def _load_registry(self) -> dict[str, Any]:
        if self.registry_path.exists():
            return json.loads(self.registry_path.read_text(encoding="utf-8"))
        return {"version": 1, "repos": {}}

    def _save_registry(self, registry: dict[str, Any]) -> None:
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _require_server(self) -> None:
        if not self._server_healthy(timeout=5.0):
            raise HTTPException(
                status_code=503,
                detail=f"Embedding model server is unavailable at {self.server_url}. Start llama.cpp embedding server first.",
            )

    def _server_healthy(self, timeout: float) -> bool:
        try:
            response = requests.get(self.server_url.rstrip("/") + "/health", timeout=timeout)
            return response.ok
        except requests.RequestException:
            return False

    def _rel(self, path: Path) -> str:
        try:
            return path.resolve().relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            return path.as_posix()

    def _abs(self, path_text: str) -> Path:
        path = Path(path_text)
        return path if path.is_absolute() else PROJECT_ROOT / path

    @staticmethod
    def _safe(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "repo"

    @staticmethod
    def _count_loc(chunk_manifest: dict[str, Any]) -> int:
        by_path: dict[str, int] = {}
        for chunk in chunk_manifest.get("chunks", {}).values():
            by_path[chunk["path"]] = max(by_path.get(chunk["path"], 0), int(chunk.get("end_line", 0)))
        return sum(by_path.values())
