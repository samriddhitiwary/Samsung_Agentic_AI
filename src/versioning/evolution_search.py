from __future__ import annotations

import difflib
import json
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.versioning.chunk_manifest import load_chunk_manifest
from src.versioning.embedding_cache import EmbeddingConfig
from src.versioning.vector_index import VersionedVectorIndex


EVOLUTION_VERSION = "p1_evolution_v1"


def build_evolution_metadata(
    *,
    repo: str,
    commit_order: list[str],
    chunk_manifest_paths: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    started = time.perf_counter()
    output_dir = Path(output_dir)
    manifests = [load_chunk_manifest(path) for path in chunk_manifest_paths]
    by_commit = {manifest["commit"]: manifest for manifest in manifests}
    if set(commit_order) != set(by_commit):
        raise ValueError("commit_order must match chunk manifests exactly")

    content_occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    content_states: dict[str, dict[str, Any]] = {}
    version_occurrences: dict[str, dict[str, Any]] = {}
    commit_graph = {"repo": repo, "order": commit_order, "parents": {}}

    for index, commit in enumerate(commit_order):
        commit_graph["parents"][commit] = [commit_order[index - 1]] if index else []
        manifest = by_commit[commit]
        for chunk in manifest.get("chunks", {}).values():
            occurrence_key = occurrence_key_for_chunk(chunk)
            occurrence = {
                "repo": repo,
                "commit": commit,
                "path": chunk["path"],
                "symbol": chunk["symbol"],
                "chunk_type": chunk["chunk_type"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "content_id": chunk["content_id"],
                "version_id": chunk["version_id"],
                "occurrence_key": occurrence_key,
            }
            version_occurrences[chunk["version_id"]] = occurrence
            content_occurrences[chunk["content_id"]].append(occurrence)
            content_states.setdefault(
                chunk["content_id"],
                {
                    "content_id": chunk["content_id"],
                    "language": chunk["language"],
                    "content_sha256": chunk["content_sha256"],
                    "text": chunk["text"],
                    "preview": preview(chunk["text"]),
                },
            )

    chains = build_chains(commit_order=commit_order, manifests=by_commit, content_states=content_states)
    metrics = evolution_metrics(commit_order, chains, content_occurrences)
    metrics["build_time_seconds"] = time.perf_counter() - started

    payload = {
        "evolution_version": EVOLUTION_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "repo": repo,
        "commit_graph": commit_graph,
        "chains": chains,
        "content_occurrences": dict(content_occurrences),
        "content_states": content_states,
        "version_occurrences": version_occurrences,
        "metrics": metrics,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chains.json").write_text(
        json.dumps({"repo": repo, "commit_order": commit_order, "chains": chains}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "content_occurrences.json").write_text(
        json.dumps({"repo": repo, "content_occurrences": dict(content_occurrences)}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "commit_graph.json").write_text(
        json.dumps(commit_graph, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "evolution_metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def load_evolution_metadata(evolution_dir: str | Path) -> dict[str, Any]:
    return json.loads((Path(evolution_dir) / "evolution_metadata.json").read_text(encoding="utf-8"))


def build_chains(
    *,
    commit_order: list[str],
    manifests: dict[str, dict[str, Any]],
    content_states: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_key_by_commit: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for commit in commit_order:
        for chunk in manifests[commit].get("chunks", {}).values():
            by_key_by_commit[occurrence_key_for_chunk(chunk)][commit] = chunk

    chains: dict[str, dict[str, Any]] = {}
    for key, chunks_by_commit in sorted(by_key_by_commit.items()):
        states: list[dict[str, Any]] = []
        transitions: list[dict[str, Any]] = []
        previous_chunk: dict[str, Any] | None = None
        seen_content_ids: set[str] = set()
        for commit in commit_order:
            current = chunks_by_commit.get(commit)
            if current is None:
                states.append({"commit": commit, "status": "absent"})
                if previous_chunk is not None:
                    transitions.append(
                        {
                            "from_commit": previous_chunk["commit"],
                            "to_commit": commit,
                            "transition": "deleted",
                            "old_content_id": previous_chunk["content_id"],
                            "new_content_id": None,
                            "lines_added": 0,
                            "lines_removed": line_count(content_states[previous_chunk["content_id"]]["text"]),
                        }
                    )
                previous_chunk = None
                continue

            current_ref = {
                "commit": commit,
                "status": "present",
                "path": current["path"],
                "symbol": current["symbol"],
                "chunk_type": current["chunk_type"],
                "content_id": current["content_id"],
                "version_id": current["version_id"],
            }
            states.append(current_ref)
            if previous_chunk is None:
                transition = "reintroduced" if current["content_id"] in seen_content_ids else "added"
                transitions.append(
                    {
                        "from_commit": None,
                        "to_commit": commit,
                        "transition": transition,
                        "old_content_id": None,
                        "new_content_id": current["content_id"],
                        "lines_added": line_count(content_states[current["content_id"]]["text"]),
                        "lines_removed": 0,
                    }
                )
            elif previous_chunk["content_id"] == current["content_id"]:
                transitions.append(
                    {
                        "from_commit": previous_chunk["commit"],
                        "to_commit": commit,
                        "transition": "unchanged",
                        "old_content_id": previous_chunk["content_id"],
                        "new_content_id": current["content_id"],
                        "lines_added": 0,
                        "lines_removed": 0,
                    }
                )
            else:
                old_text = content_states[previous_chunk["content_id"]]["text"]
                new_text = content_states[current["content_id"]]["text"]
                diff = deterministic_diff(old_text, new_text)
                transitions.append(
                    {
                        "from_commit": previous_chunk["commit"],
                        "to_commit": commit,
                        "transition": "modified",
                        "old_content_id": previous_chunk["content_id"],
                        "new_content_id": current["content_id"],
                        **diff,
                    }
                )
            seen_content_ids.add(current["content_id"])
            previous_chunk = {**current, "commit": commit}
        chains[key] = {"occurrence_key": key, "states": states, "transitions": transitions}
    return chains


def search_across_versions(
    *,
    index_dir: str | Path,
    evolution_dir: str | Path,
    query_embedding: np.ndarray,
    start_commit: str | None = None,
    end_commit: str | None = None,
    top_k: int = 5,
    raw: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    metadata = load_evolution_metadata(evolution_dir)
    commits = select_commits(metadata["commit_graph"]["order"], start_commit, end_commit)
    index = VersionedVectorIndex.load(index_dir, EmbeddingConfig())

    raw_results: list[dict[str, Any]] = []
    for commit in commits:
        raw_results.extend(index.search(query_embedding=query_embedding, commit=commit, top_k=max(top_k, 20)))

    grouping_started = time.perf_counter()
    if raw:
        results = attach_text_to_raw(raw_results, metadata)
    else:
        results = group_results_by_content(raw_results, metadata, commits, top_k=top_k)
    grouping_time = time.perf_counter() - grouping_started
    return {
        "repo": metadata["repo"],
        "commits": commits,
        "top_k": top_k,
        "raw": raw,
        "results": results[:top_k] if raw else results,
        "runtime": {
            "search_seconds": time.perf_counter() - started,
            "grouping_seconds": grouping_time,
        },
    }


def group_results_by_content(
    raw_results: list[dict[str, Any]],
    metadata: dict[str, Any],
    commits: list[str],
    top_k: int,
) -> list[dict[str, Any]]:
    best_by_content: dict[str, dict[str, Any]] = {}
    for result in raw_results:
        content_id = result["content_id"]
        if content_id not in best_by_content or result["score"] > best_by_content[content_id]["score"]:
            best_by_content[content_id] = result

    selected_commits = set(commits)
    grouped: list[dict[str, Any]] = []
    for content_id, best in best_by_content.items():
        occurrences = [
            occ
            for occ in metadata["content_occurrences"].get(content_id, [])
            if occ["commit"] in selected_commits
        ]
        state = metadata["content_states"][content_id]
        grouped.append(
            {
                "content_id": content_id,
                "score": best["score"],
                "best_match": best,
                "preview": state["preview"],
                "text": state["text"],
                "occurrences": sorted(occurrences, key=lambda item: (commits.index(item["commit"]), item["path"], item["symbol"])),
                "lineage": lineage_for_content(metadata, content_id),
            }
        )
    grouped.sort(key=lambda item: (-item["score"], item["content_id"]))
    return grouped[:top_k]


def attach_text_to_raw(raw_results: list[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    values = []
    for result in sorted(raw_results, key=lambda item: (-item["score"], item["commit"], item["path"], item["symbol"])):
        state = metadata["content_states"][result["content_id"]]
        values.append({**result, "preview": state["preview"], "text": state["text"], "lineage": lineage_for_content(metadata, result["content_id"])})
    return values


def select_commits(order: list[str], start_commit: str | None, end_commit: str | None) -> list[str]:
    start = order.index(start_commit) if start_commit else 0
    end = order.index(end_commit) if end_commit else len(order) - 1
    if start > end:
        raise ValueError("start_commit must not come after end_commit")
    return order[start : end + 1]


def deterministic_diff(old_text: str, new_text: str) -> dict[str, int]:
    lines = list(difflib.ndiff(old_text.splitlines(), new_text.splitlines()))
    return {
        "lines_added": sum(1 for line in lines if line.startswith("+ ")),
        "lines_removed": sum(1 for line in lines if line.startswith("- ")),
    }


def evolution_metrics(
    commit_order: list[str],
    chains: dict[str, dict[str, Any]],
    content_occurrences: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    transition_counts = Counter(
        transition["transition"]
        for chain in chains.values()
        for transition in chain["transitions"]
    )
    total_occurrences = sum(len(value) for value in content_occurrences.values())
    unique_content_ids = len(content_occurrences)
    return {
        "commits_indexed": len(commit_order),
        "total_chunk_occurrences": total_occurrences,
        "unique_content_ids": unique_content_ids,
        "duplicate_occurrences_collapsed": total_occurrences - unique_content_ids,
        "evolution_chains": len(chains),
        "unchanged_transitions": transition_counts["unchanged"],
        "modified_transitions": transition_counts["modified"],
        "additions": transition_counts["added"],
        "deletions": transition_counts["deleted"],
        "reintroductions": transition_counts["reintroduced"],
    }


def lineage_for_content(metadata: dict[str, Any], content_id: str) -> list[dict[str, Any]]:
    values: list[dict[str, Any]] = []
    for chain in metadata["chains"].values():
        for state in chain["states"]:
            if state.get("content_id") == content_id:
                values.append({"occurrence_key": chain["occurrence_key"], **state})
    return values


def occurrence_key_for_chunk(chunk: dict[str, Any]) -> str:
    return f"{chunk['path']}:{chunk['chunk_type']}:{chunk['symbol']}"


def preview(text: str, limit: int = 240) -> str:
    compact = " ".join(text.strip().split())
    return compact[:limit] + ("..." if len(compact) > limit else "")


def line_count(text: str) -> int:
    return len(text.splitlines())

