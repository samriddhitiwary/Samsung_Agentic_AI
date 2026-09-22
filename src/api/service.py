from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import requests
from fastapi import HTTPException
from transformers import AutoTokenizer

from src.retrieval.jina_code import QUERY_INSTRUCTION
from src.retrieval.jina_gguf import truncate_with_tokenizer
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

        registry = self._load_registry()
        registry["repos"][resolved_repo_id] = {
            "repo_id": resolved_repo_id,
            "repo_path": str(repo),
            "commits": [resolved_commit],
            "active_commit": resolved_commit,
            "artifact_root": self._rel(paths["root"]),
            "manifest_paths": {resolved_commit: self._rel(manifest_path)},
            "chunk_manifest_paths": {resolved_commit: self._rel(chunk_path)},
            "created_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._save_registry(registry)

        return {
            "repo": resolved_repo_id,
            "commit": resolved_commit,
            "source_files": len(file_manifest.get("files", {})),
            "chunks": len(chunk_manifest.get("chunks", {})),
            "reused_embeddings": embedding_report["reused_embeddings"],
            "newly_generated_embeddings": embedding_report["newly_generated_embeddings"],
            "index": {
                "active_vectors": index_report["unique_vectors"],
                "active_occurrences": index_report["active_occurrences"],
            },
            "evolution_chains": evolution["metrics"]["evolution_chains"],
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

        commits = list(repo_record["commits"])
        if new_commit not in commits:
            commits.append(new_commit)
        repo_record["commits"] = commits
        repo_record["active_commit"] = new_commit
        repo_record["manifest_paths"][new_commit] = self._rel(manifest_path)
        repo_record["chunk_manifest_paths"][new_commit] = self._rel(chunk_path)
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
            index = VersionedVectorIndex.load(paths["index"], self.embedding_config)
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"Could not load persisted index: {exc}") from exc
        metadata = load_evolution_metadata(paths["evolution"])
        vector = self._embed_query(query)
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
        if self._tokenizer is None:
            self._tokenizer = AutoTokenizer.from_pretrained(self.embedding_config.model_name)
        prompt = truncate_with_tokenizer(
            tokenizer=self._tokenizer,
            text=query,
            instruction=QUERY_INSTRUCTION,
            max_length=self.embedding_config.max_sequence_length,
        )
        return _embed_http(
            texts=[prompt],
            server_url=self.server_url,
            batch_size=1,
            normalize=True,
            desc="api query embedding",
        )[0]

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
