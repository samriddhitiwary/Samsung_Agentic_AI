from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
from src.versioning.evolution_search import build_evolution_metadata, search_across_versions
from src.versioning.git_scanner import ScannerConfig, scan_git_repository
from src.versioning.incremental_embedder import _embed_http, build_incremental_embeddings, check_server
from src.versioning.index_updater import build_version_index, update_version_index
from src.versioning.manifest import compare_manifests, load_manifest, manifest_output_path, save_manifest
from src.versioning.vector_index import VersionedVectorIndex


DEFAULT_QUERIES = [
    "request routing",
    "configuration loading",
    "error handling",
    "HTTP header parsing",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real-repository P1 versioning benchmark.")
    parser.add_argument("--repo-path", required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--commits", nargs="+", required=True, help="Chronological commit refs.")
    parser.add_argument("--server-url", default="http://127.0.0.1:8081")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--clean-rebuild-count", type=int, default=2)
    parser.add_argument("--benchmark-root", default="data/versioning/benchmarks")
    parser.add_argument("--real-output-root", default="data/versioning/real_benchmark_artifacts")
    parser.add_argument("--queries", nargs="*", default=DEFAULT_QUERIES)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_path = Path(args.repo_path).resolve()
    repo_id = args.repo_id or repo_path.name
    safe_repo = safe_name(repo_id)
    benchmark_dir = Path(args.benchmark_root) / safe_repo
    artifacts_root = Path(args.real_output_root) / safe_repo
    manifest_root = artifacts_root / "manifests"
    chunk_root = artifacts_root / "chunks"
    reports_root = artifacts_root / "reports"
    incremental_cache_dir = artifacts_root / "embedding_cache_incremental"
    incremental_index_dir = artifacts_root / "index_incremental"
    evolution_dir = artifacts_root / "evolution"

    check_server(args.server_url)
    config = EmbeddingConfig()
    commits = [git_text(repo_path, ["rev-parse", ref]).strip() for ref in args.commits]
    validate_ancestry(repo_path, commits)

    benchmark_dir.mkdir(parents=True, exist_ok=True)
    artifacts_root.mkdir(parents=True, exist_ok=True)
    reports_root.mkdir(parents=True, exist_ok=True)

    start_all = time.perf_counter()
    per_commit: list[dict[str, Any]] = []
    manifest_paths: list[Path] = []
    chunk_paths: list[Path] = []
    incremental_total_times: dict[str, float] = {}

    for index, commit in enumerate(commits):
        stage_started = time.perf_counter()

        file_manifest = scan_git_repository(
            repo_path=repo_path,
            commit_ref=commit,
            repo_id=repo_id,
            config=ScannerConfig(),
        )
        manifest_path = manifest_output_path(manifest_root, repo_id, file_manifest["commit"])
        save_manifest(file_manifest, manifest_path)
        manifest_paths.append(manifest_path)

        chunk_manifest = build_chunk_manifest(repo_path=repo_path, file_manifest_path=manifest_path)
        chunk_path = chunk_manifest_output_path(chunk_root, repo_id, chunk_manifest["commit"])
        save_chunk_manifest(chunk_manifest, chunk_path)
        chunk_paths.append(chunk_path)

        file_diff: dict[str, Any] | None = None
        chunk_diff: dict[str, Any] | None = None
        if index > 0:
            old_manifest = load_manifest(manifest_paths[index - 1])
            file_diff = compare_manifests(old_manifest, file_manifest)
            old_chunk_manifest = load_chunk_manifest(chunk_paths[index - 1])
            chunk_diff = compare_chunk_manifests(
                old_chunk_manifest,
                chunk_manifest,
                old_file_manifest=old_manifest,
                new_file_manifest=file_manifest,
            )

        embedding_report = build_incremental_embeddings(
            chunk_manifest_path=chunk_path,
            cache_dir=incremental_cache_dir,
            server_url=args.server_url,
            config=config,
            batch_size=args.batch_size,
        )
        save_json(reports_root / f"{commit}_embedding_report.json", embedding_report)

        if index == 0:
            index_report = build_version_index(
                chunk_manifest_path=chunk_path,
                embedding_cache_dir=incremental_cache_dir,
                index_dir=incremental_index_dir,
                embedding_config=config,
            )
        else:
            index_report = update_version_index(
                existing_index_dir=incremental_index_dir,
                target_chunk_manifest_path=chunk_path,
                embedding_cache_dir=incremental_cache_dir,
                embedding_config=config,
            )
        save_json(reports_root / f"{commit}_index_report.json", index_report)

        total_seconds = time.perf_counter() - stage_started
        incremental_total_times[commit] = total_seconds
        per_commit.append(
            per_commit_record(
                index=index,
                commit=commit,
                file_manifest=file_manifest,
                chunk_manifest=chunk_manifest,
                file_diff=file_diff,
                chunk_diff=chunk_diff,
                embedding_report=embedding_report,
                index_report=index_report,
                total_seconds=total_seconds,
            )
        )

        if index == 1:
            restart_ok = restart_probe(incremental_index_dir, config, commit)
            per_commit[-1]["restart_probe_after_commit"] = restart_ok

    evolution_summary = build_evolution_metadata(
        repo=repo_id,
        commit_order=commits,
        chunk_manifest_paths=chunk_paths,
        output_dir=evolution_dir,
    )

    clean_commits = commits[-max(0, args.clean_rebuild_count) :]
    clean_rebuilds = []
    for commit in clean_commits:
        clean_rebuilds.append(
            run_clean_rebuild(
                repo_path=repo_path,
                repo_id=repo_id,
                commit=commit,
                server_url=args.server_url,
                batch_size=args.batch_size,
                config=config,
                output_root=artifacts_root / "clean_rebuilds" / commit,
            )
        )

    query_embeddings = embed_queries(args.queries, args.server_url, config, args.batch_size)
    parity = retrieval_parity_checks(
        queries=args.queries,
        query_embeddings=query_embeddings,
        commits=clean_commits,
        incremental_index_dir=incremental_index_dir,
        clean_rebuilds=clean_rebuilds,
        config=config,
        top_k=args.top_k,
    )
    evolution_queries = run_evolution_queries(
        queries=args.queries,
        query_embeddings=query_embeddings,
        index_dir=incremental_index_dir,
        evolution_dir=evolution_dir,
        top_k=args.top_k,
        output_dir=reports_root / "evolution_queries",
    )
    chains = load_json(evolution_dir / "chains.json")["chains"]
    selected_chains = select_interesting_chains(chains)

    aggregate = aggregate_metrics(
        per_commit=per_commit,
        clean_rebuilds=clean_rebuilds,
        incremental_total_times=incremental_total_times,
    )
    storage = storage_metrics([artifacts_root, benchmark_dir])
    total_runtime = time.perf_counter() - start_all

    report = {
        "benchmark_version": "p1_real_repo_benchmark_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "repository": {
            "name": repo_id,
            "path": None,
            "url": args.repo_url,
            "languages": sorted(
                {
                    language
                    for record in per_commit
                    for language in record["repository"]["languages"]
                }
            ),
            "selected_commit_range": {
                "first": commits[0],
                "last": commits[-1],
                "count": len(commits),
            },
        },
        "selected_commits": commits,
        "environment_model_config": {
            "python": sys.version,
            "embedding_config": config.__dict__,
            "server_url": args.server_url,
            "batch_size": args.batch_size,
        },
        "per_commit": per_commit,
        "clean_rebuilds": clean_rebuilds,
        "retrieval_parity": parity,
        "restart_resume": {
            "passed": bool(per_commit[1].get("restart_probe_after_commit", False)) if len(per_commit) > 1 else None,
            "method": "index was persisted after the second commit, reloaded from disk, then later updates continued from disk",
        },
        "evolution_summary": evolution_summary["metrics"],
        "selected_evolution_chains": selected_chains,
        "evolution_queries": evolution_queries,
        "aggregate_metrics": aggregate,
        "storage": storage,
        "total_benchmark_runtime_seconds": total_runtime,
        "artifacts": {
            "manifest_root": manifest_root.as_posix(),
            "chunk_root": chunk_root.as_posix(),
            "incremental_cache_dir": incremental_cache_dir.as_posix(),
            "incremental_index_dir": incremental_index_dir.as_posix(),
            "evolution_dir": evolution_dir.as_posix(),
            "reports_root": reports_root.as_posix(),
        },
    }

    report_path = benchmark_dir / "p1_real_repo_benchmark.json"
    csv_path = benchmark_dir / "p1_real_repo_benchmark.csv"
    save_json(report_path, report)
    save_csv(csv_path, per_commit, clean_rebuilds)

    print(json.dumps(summary_for_console(report, report_path, csv_path), indent=2, sort_keys=True))
    return 0


def per_commit_record(
    *,
    index: int,
    commit: str,
    file_manifest: dict[str, Any],
    chunk_manifest: dict[str, Any],
    file_diff: dict[str, Any] | None,
    chunk_diff: dict[str, Any] | None,
    embedding_report: dict[str, Any],
    index_report: dict[str, Any],
    total_seconds: float,
) -> dict[str, Any]:
    total_files = len(file_manifest.get("files", {}))
    total_chunks = len(chunk_manifest.get("chunks", {}))
    if file_diff is None:
        file_counts = {"unchanged": 0, "modified": 0, "added": total_files, "deleted": 0}
        file_reuse = 0.0
    else:
        file_counts = file_diff["counts"]
        file_reuse = file_diff["percentages"]["unchanged"]

    if chunk_diff is None:
        chunk_counts = {
            "reused_chunks": 0,
            "modified_replaced_chunks": 0,
            "new_chunks": total_chunks,
            "deleted_chunks": 0,
            "new_chunks_requiring_embeddings": total_chunks,
        }
        chunk_reuse = 0.0
    else:
        chunk_counts = chunk_diff["counts"]
        chunk_reuse = chunk_diff["reuse_percentage"]

    return {
        "commit_index": index,
        "commit": commit,
        "repository": {
            "source_files": total_files,
            "languages": sorted({item["language"] for item in file_manifest.get("files", {}).values()}),
            "file_changes": file_counts,
            "file_reuse_percent": file_reuse,
        },
        "chunks": {
            "total_chunks": total_chunks,
            "reusable_chunks": chunk_counts["reused_chunks"],
            "modified_replaced_chunks": chunk_counts["modified_replaced_chunks"],
            "new_chunks": chunk_counts["new_chunks"],
            "deleted_chunks": chunk_counts["deleted_chunks"],
            "new_chunks_requiring_embeddings": chunk_counts["new_chunks_requiring_embeddings"],
            "reuse_percent": chunk_reuse,
        },
        "embeddings": {
            "cache_hits": embedding_report["cache_hits"],
            "cache_misses": embedding_report["cache_misses"],
            "new_embeddings_generated": embedding_report["newly_generated_embeddings"],
            "embeddings_reused": embedding_report["reused_embeddings"],
            "embedding_reuse_percent": embedding_report["reuse_percentage"],
            "duplicate_occurrences_avoided": embedding_report["duplicate_occurrences_avoided"],
        },
        "index": {
            "reused_vectors": index_report.get("reused_vectors", 0),
            "newly_inserted_vectors": index_report.get("vectors_newly_inserted", index_report.get("vectors_inserted", 0)),
            "removed_tombstoned_vectors": index_report.get("vectors_tombstoned", 0),
            "active_vectors": index_report.get("active_vectors_after_update", index_report.get("unique_vectors")),
            "active_occurrences": index_report.get("active_occurrences_after_update", index_report.get("active_occurrences")),
            "unique_vectors_total": index_report.get("unique_vectors_total", index_report.get("unique_vectors")),
            "persisted_index_size_bytes": index_report.get("persisted_index_size_bytes"),
        },
        "runtime": {
            "git_file_scan_seconds": file_manifest["stats"]["scan_time_seconds"],
            "chunk_manifest_build_seconds": chunk_manifest["stats"]["build_time_seconds"],
            "embedding_stage_seconds": embedding_report["runtime"]["total_seconds"],
            "index_update_seconds": index_report.get("update_time_seconds", index_report.get("index_build_time_seconds")),
            "total_incremental_update_seconds": total_seconds,
        },
    }


def run_clean_rebuild(
    *,
    repo_path: Path,
    repo_id: str,
    commit: str,
    server_url: str,
    batch_size: int,
    config: EmbeddingConfig,
    output_root: Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    manifest_root = output_root / "manifest"
    chunk_root = output_root / "chunks"
    cache_dir = output_root / "embedding_cache_clean"
    index_dir = output_root / "index_clean"

    file_manifest = scan_git_repository(repo_path=repo_path, commit_ref=commit, repo_id=repo_id, config=ScannerConfig())
    manifest_path = manifest_output_path(manifest_root, repo_id, file_manifest["commit"])
    save_manifest(file_manifest, manifest_path)
    chunk_manifest = build_chunk_manifest(repo_path=repo_path, file_manifest_path=manifest_path)
    chunk_path = chunk_manifest_output_path(chunk_root, repo_id, chunk_manifest["commit"])
    save_chunk_manifest(chunk_manifest, chunk_path)
    embedding_report = build_incremental_embeddings(
        chunk_manifest_path=chunk_path,
        cache_dir=cache_dir,
        server_url=server_url,
        config=config,
        batch_size=batch_size,
    )
    index_report = build_version_index(
        chunk_manifest_path=chunk_path,
        embedding_cache_dir=cache_dir,
        index_dir=index_dir,
        embedding_config=config,
    )
    total = time.perf_counter() - started
    return {
        "commit": commit,
        "manifest_path": manifest_path.as_posix(),
        "chunk_manifest_path": chunk_path.as_posix(),
        "cache_dir": cache_dir.as_posix(),
        "index_dir": index_dir.as_posix(),
        "source_files": len(file_manifest.get("files", {})),
        "chunks": len(chunk_manifest.get("chunks", {})),
        "unique_content_ids": embedding_report["unique_content_ids"],
        "new_embeddings_generated": embedding_report["newly_generated_embeddings"],
        "runtime": {
            "full_manifest_time_seconds": file_manifest["stats"]["scan_time_seconds"],
            "full_chunking_time_seconds": chunk_manifest["stats"]["build_time_seconds"],
            "full_embedding_time_seconds": embedding_report["runtime"]["total_seconds"],
            "full_index_build_time_seconds": index_report["index_build_time_seconds"],
            "total_rebuild_time_seconds": total,
        },
        "index": {
            "active_vectors": index_report["unique_vectors"],
            "active_occurrences": index_report["active_occurrences"],
            "persisted_index_size_bytes": index_report["persisted_index_size_bytes"],
        },
    }


def embed_queries(
    queries: list[str],
    server_url: str,
    config: EmbeddingConfig,
    batch_size: int,
) -> dict[str, list[float]]:
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    prompts = [
        truncate_with_tokenizer(
            tokenizer=tokenizer,
            text=query,
            instruction=QUERY_INSTRUCTION,
            max_length=config.max_sequence_length,
        )
        for query in queries
    ]
    query_config = EmbeddingConfig(prompt_instruction=QUERY_INSTRUCTION, prompt_version="jina_code_1.5b_query_v1")
    vectors = _embed_http(
        texts=prompts,
        server_url=server_url,
        batch_size=batch_size,
        normalize=query_config.normalization,
        desc="embed benchmark queries",
    )
    return {query: vectors[i].astype(np.float32).tolist() for i, query in enumerate(queries)}


def retrieval_parity_checks(
    *,
    queries: list[str],
    query_embeddings: dict[str, list[float]],
    commits: list[str],
    incremental_index_dir: Path,
    clean_rebuilds: list[dict[str, Any]],
    config: EmbeddingConfig,
    top_k: int,
) -> dict[str, Any]:
    incremental = VersionedVectorIndex.load(incremental_index_dir, config)
    clean_by_commit = {
        item["commit"]: VersionedVectorIndex.load(item["index_dir"], config)
        for item in clean_rebuilds
    }
    checks = []
    for commit in commits:
        clean = clean_by_commit[commit]
        for query in queries:
            vector = np.asarray(query_embeddings[query], dtype=np.float32)
            inc_results = incremental.search(query_embedding=vector, commit=commit, top_k=top_k)
            clean_results = clean.search(query_embedding=vector, commit=commit, top_k=top_k)
            inc_ids = [r["version_id"] for r in inc_results]
            clean_ids = [r["version_id"] for r in clean_results]
            score_deltas = [
                abs(float(inc["score"]) - float(clean["score"]))
                for inc, clean in zip(inc_results, clean_results)
            ]
            max_score_delta = max(score_deltas) if score_deltas else 0.0
            checks.append(
                {
                    "commit": commit,
                    "query": query,
                    "passed": inc_ids == clean_ids,
                    "equivalence_rule": "exact same top-k version_id ordering; score drift is reported separately",
                    "max_score_delta": max_score_delta,
                    "incremental_top": [
                        {"version_id": r["version_id"], "score": float(r["score"])}
                        for r in inc_results
                    ],
                    "clean_top": [
                        {"version_id": r["version_id"], "score": float(r["score"])}
                        for r in clean_results
                    ],
                }
            )
    return {
        "passed": all(item["passed"] for item in checks),
        "top_k": top_k,
        "checks": checks,
    }


def run_evolution_queries(
    *,
    queries: list[str],
    query_embeddings: dict[str, list[float]],
    index_dir: Path,
    evolution_dir: Path,
    top_k: int,
    output_dir: Path,
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for query in queries:
        vector = np.asarray(query_embeddings[query], dtype=np.float32)
        payload = search_across_versions(
            index_dir=index_dir,
            evolution_dir=evolution_dir,
            query_embedding=vector,
            top_k=top_k,
            raw=False,
        )
        payload["query"] = query
        safe_query = safe_name(query)[:60]
        save_json(output_dir / f"{safe_query}.json", payload)
        results.append(
            {
                "query": query,
                "top_results": [
                    {
                        "rank": rank,
                        "score": result["score"],
                        "symbol": result["best_match"]["symbol"],
                        "path": result["best_match"]["path"],
                        "best_commit": result["best_match"]["commit"],
                        "occurrence_count": len(result.get("occurrences", [])),
                        "preview": result.get("preview"),
                    }
                    for rank, result in enumerate(payload["results"], start=1)
                ],
                "runtime": payload["runtime"],
            }
        )
    return results


def select_interesting_chains(chains: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    scored = []
    for key, chain in chains.items():
        transitions = [item["transition"] for item in chain.get("transitions", [])]
        score = (
            transitions.count("modified") * 4
            + transitions.count("deleted") * 3
            + transitions.count("reintroduced") * 3
            + transitions.count("added")
        )
        if score > 1:
            scored.append((score, key, chain))
    scored.sort(key=lambda item: (-item[0], item[1]))
    selected = []
    for _, key, chain in scored[:limit]:
        selected.append(
            {
                "occurrence_key": key,
                "transitions": chain.get("transitions", []),
                "states": [
                    {
                        "commit": state["commit"],
                        "status": state["status"],
                        "content_id": state.get("content_id"),
                        "path": state.get("path"),
                        "symbol": state.get("symbol"),
                    }
                    for state in chain.get("states", [])
                ],
            }
        )
    return selected


def aggregate_metrics(
    *,
    per_commit: list[dict[str, Any]],
    clean_rebuilds: list[dict[str, Any]],
    incremental_total_times: dict[str, float],
) -> dict[str, Any]:
    later = per_commit[1:] or per_commit
    clean_by_commit = {item["commit"]: item for item in clean_rebuilds}
    speedups = []
    for commit, clean in clean_by_commit.items():
        inc = incremental_total_times[commit]
        speedups.append(clean["runtime"]["total_rebuild_time_seconds"] / inc if inc else math.inf)
    total_if_full = sum(item["unique_content_ids"] for item in clean_rebuilds)
    total_new_embeddings = sum(
        record["embeddings"]["new_embeddings_generated"]
        for record in per_commit
        if record["commit"] in clean_by_commit
    )
    avoided_in_clean_compared_commits = max(0, total_if_full - total_new_embeddings)
    total_embeddings_avoided_all_incremental = sum(
        record["embeddings"]["cache_hits"]
        for record in per_commit[1:]
    )
    total_duplicate_vector_insertions_avoided = sum(
        max(0, record["chunks"]["total_chunks"] - record["index"]["newly_inserted_vectors"])
        for record in per_commit
    )
    return {
        "average_file_reuse_percent": mean([r["repository"]["file_reuse_percent"] for r in later]),
        "average_chunk_reuse_percent": mean([r["chunks"]["reuse_percent"] for r in later]),
        "average_embedding_reuse_percent": mean([r["embeddings"]["embedding_reuse_percent"] for r in later]),
        "average_incremental_update_time_seconds": mean([r["runtime"]["total_incremental_update_seconds"] for r in later]),
        "average_full_rebuild_time_seconds": mean([r["runtime"]["total_rebuild_time_seconds"] for r in clean_rebuilds]),
        "median_speedup": statistics.median(speedups) if speedups else None,
        "maximum_speedup": max(speedups) if speedups else None,
        "speedups": speedups,
        "total_embeddings_avoided": total_embeddings_avoided_all_incremental,
        "total_duplicate_vector_insertions_avoided": total_duplicate_vector_insertions_avoided,
        "embedding_work_saved_clean_comparison": (
            avoided_in_clean_compared_commits / total_if_full if total_if_full else None
        ),
        "clean_comparison_total_embedding_work_if_full_rebuild": total_if_full,
        "clean_comparison_embeddings_avoided": avoided_in_clean_compared_commits,
    }


def restart_probe(index_dir: Path, config: EmbeddingConfig, commit: str) -> bool:
    loaded = VersionedVectorIndex.load(index_dir, config)
    return loaded.state.get("active_commit") == commit and loaded.active_occurrence_count(commit) > 0


def storage_metrics(paths: list[Path]) -> dict[str, Any]:
    by_path = {}
    total = 0
    for path in paths:
        size = dir_size(path)
        by_path[path.as_posix()] = size
        total += size
    return {"total_bytes": total, "by_path_bytes": by_path}


def save_csv(path: Path, per_commit: list[dict[str, Any]], clean_rebuilds: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clean_by_commit = {item["commit"]: item for item in clean_rebuilds}
    fields = [
        "commit",
        "source_files",
        "total_chunks",
        "file_reuse_percent",
        "chunk_reuse_percent",
        "embedding_reuse_percent",
        "new_embeddings_generated",
        "active_vectors",
        "active_occurrences",
        "incremental_update_seconds",
        "clean_rebuild_seconds",
        "speedup",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in per_commit:
            clean = clean_by_commit.get(record["commit"])
            inc = record["runtime"]["total_incremental_update_seconds"]
            clean_time = clean["runtime"]["total_rebuild_time_seconds"] if clean else None
            writer.writerow(
                {
                    "commit": record["commit"],
                    "source_files": record["repository"]["source_files"],
                    "total_chunks": record["chunks"]["total_chunks"],
                    "file_reuse_percent": record["repository"]["file_reuse_percent"],
                    "chunk_reuse_percent": record["chunks"]["reuse_percent"],
                    "embedding_reuse_percent": record["embeddings"]["embedding_reuse_percent"],
                    "new_embeddings_generated": record["embeddings"]["new_embeddings_generated"],
                    "active_vectors": record["index"]["active_vectors"],
                    "active_occurrences": record["index"]["active_occurrences"],
                    "incremental_update_seconds": inc,
                    "clean_rebuild_seconds": clean_time,
                    "speedup": clean_time / inc if clean_time is not None and inc else None,
                }
            )


def summary_for_console(report: dict[str, Any], report_path: Path, csv_path: Path) -> dict[str, Any]:
    return {
        "repository": report["repository"],
        "selected_commits": report["selected_commits"],
        "per_commit_brief": [
            {
                "commit": item["commit"][:12],
                "files": item["repository"]["source_files"],
                "chunks": item["chunks"]["total_chunks"],
                "file_reuse_percent": item["repository"]["file_reuse_percent"],
                "chunk_reuse_percent": item["chunks"]["reuse_percent"],
                "embedding_reuse_percent": item["embeddings"]["embedding_reuse_percent"],
                "incremental_seconds": item["runtime"]["total_incremental_update_seconds"],
            }
            for item in report["per_commit"]
        ],
        "aggregate_metrics": report["aggregate_metrics"],
        "retrieval_parity_passed": report["retrieval_parity"]["passed"],
        "restart_resume_passed": report["restart_resume"]["passed"],
        "report_path": report_path.as_posix(),
        "csv_path": csv_path.as_posix(),
    }


def validate_ancestry(repo: Path, commits: list[str]) -> None:
    for parent, child in zip(commits, commits[1:]):
        subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", parent, child], check=True)


def git_text(repo: Path, args: list[str]) -> str:
    result = subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True)
    return result.stdout


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value).strip("_") or "repo"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def mean(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return statistics.fmean(clean) if clean else None


if __name__ == "__main__":
    raise SystemExit(main())
