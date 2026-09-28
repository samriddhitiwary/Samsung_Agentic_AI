from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

from src.agentic.router import classify_query
from src.structure.graph import search_call_order, search_references
from src.versioning.chunk_manifest import load_chunk_manifest


SemanticSearchFn = Callable[[str, str, int], tuple[list[dict[str, Any]], dict[str, float | str | bool | None]]]


def run_agentic_query(
    *,
    repo_id: str,
    commit: str,
    query: str,
    top_k: int,
    chunk_manifest_path: str | Path,
    chunk_manifest: dict[str, Any] | None = None,
    graph: dict[str, Any] | None,
    semantic_search: SemanticSearchFn,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    timings: dict[str, float | str | None] = {
        "request_parsing_ms": None,
        "classification_ms": 0.0,
        "query_embedding_ms": None,
        "query_embedding_cache": None,
        "vector_index_access_ms": None,
        "semantic_search_ms": 0.0,
        "structural_search_ms": 0.0,
        "candidate_read_ms": 0.0,
        "refinement_ms": 0.0,
        "ranking_ms": 0.0,
        "serialization_ms": None,
        "total_ms": 0.0,
    }
    trace: list[dict[str, Any]] = []

    started = time.perf_counter()
    plan = classify_query(query, top_k=top_k)
    timings["classification_ms"] = _ms(started)
    trace.append({"stage": "classify", "result": plan.query_type, "terms": plan.terms})

    started = time.perf_counter()
    structural: list[dict[str, Any]] = []
    if graph is not None and plan.order_pair:
        structural.extend(search_call_order(graph, plan.order_pair[0], plan.order_pair[1], limit=plan.structural_top_k))
    if graph is not None and plan.terms and plan.query_type in {"usage", "structural", "mixed"}:
        structural.extend(search_references(graph, plan.terms, limit=plan.structural_top_k))
    timings["structural_search_ms"] = _ms(started)
    trace.append({"stage": "structural_search", "matches": len(structural)})

    semantic_raw: list[dict[str, Any]] = []
    exact_structural = [item for item in structural if item.get("evidence_type") == "structural"]
    if plan.query_type == "usage" and structural:
        should_run_semantic = False
    elif plan.query_type == "structural" and (exact_structural or structural):
        should_run_semantic = False
    else:
        should_run_semantic = plan.query_type in {"semantic", "mixed"} or len(_dedupe_structural(structural)) < top_k
    if should_run_semantic:
        started = time.perf_counter()
        semantic_raw, semantic_profile = semantic_search(query, commit, plan.semantic_top_k)
        timings["semantic_search_ms"] = float(semantic_profile.get("semantic_search_ms") or _ms(started))
        timings["query_embedding_ms"] = _float_or_none(semantic_profile.get("query_embedding_ms"))
        timings["query_embedding_cache"] = str(semantic_profile.get("query_embedding_cache") or "unknown")
        timings["vector_index_access_ms"] = _float_or_none(semantic_profile.get("vector_index_access_ms"))
        trace.append({"stage": "semantic_search", "candidates": len(semantic_raw), "cache": timings["query_embedding_cache"]})
    else:
        trace.append({"stage": "semantic_search", "skipped": True, "reason": "sufficient exact structural/reference evidence"})

    started = time.perf_counter()
    manifest_payload = chunk_manifest if chunk_manifest is not None else load_chunk_manifest(chunk_manifest_path)
    chunks = manifest_payload.get("chunks", {})
    by_version = {chunk["version_id"]: chunk for chunk in chunks.values()}
    timings["candidate_read_ms"] = _ms(started)
    trace.append({"stage": "read", "regions": len(set([r.get("version_id") for r in semantic_raw + structural if r.get("version_id")] ))})

    started = time.perf_counter()
    rank_started = time.perf_counter()
    merged = _merge_and_score(
        semantic_raw=semantic_raw,
        structural_raw=structural,
        chunk_by_version=by_version,
        query_type=plan.query_type,
        top_k=top_k,
    )
    timings["ranking_ms"] = _ms(rank_started)
    timings["refinement_ms"] = _ms(started)
    trace.append({"stage": "refine", "remaining_candidates": len(merged)})
    trace.append({"stage": "rank", "returned": min(top_k, len(merged))})

    timings["total_ms"] = (time.perf_counter() - total_start) * 1000.0
    return {
        "query": query,
        "query_type": plan.query_type,
        "repo_id": repo_id,
        "commit": commit,
        "timing": timings,
        "trace": trace,
        "results": _rank_results(merged[:top_k]),
        "scoring": {
            "semantic": "base normalized Jina vector score from the persisted repository index",
            "reference": "+0.35 exact reference/call/symbol evidence",
            "structural": "+0.45 static source-order/call-order evidence",
            "path_symbol": "+0.05 symbol/path term overlap",
            "note": "Deterministic retrieval scoring; qrels are never used.",
        },
    }


def _merge_and_score(
    *,
    semantic_raw: list[dict[str, Any]],
    structural_raw: list[dict[str, Any]],
    chunk_by_version: dict[str, dict[str, Any]],
    query_type: str,
    top_k: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    semantic_scores = [float(item.get("score", 0.0)) for item in semantic_raw]
    min_score = min(semantic_scores) if semantic_scores else 0.0
    max_score = max(semantic_scores) if semantic_scores else 1.0
    denom = max(max_score - min_score, 1e-9)

    for item in semantic_raw:
        version_id = item["version_id"]
        chunk = chunk_by_version.get(version_id, {})
        normalized = (float(item.get("score", 0.0)) - min_score) / denom
        merged[version_id] = _result_from_chunk(chunk, item, base_score=normalized, evidence_type="semantic")

    for item in structural_raw:
        version_id = item["version_id"]
        chunk = chunk_by_version.get(version_id, item)
        current = merged.get(version_id) or _result_from_chunk(chunk, item, base_score=0.0, evidence_type=item.get("evidence_type", "reference"))
        evidence = item.get("evidence_type", "reference")
        if evidence == "structural":
            boost = 0.90
        elif query_type == "structural":
            boost = 0.15
        else:
            boost = 0.35
        if query_type == "usage" and evidence in {"reference", "call", "definition"}:
            boost += 0.15
        current["score"] = float(current.get("score", 0.0)) + boost
        current["evidence_type"] = "mixed" if current.get("evidence_type") != evidence and current.get("evidence_type") != "semantic" else evidence
        current["referenced_symbol"] = item.get("referenced_symbol") or item.get("matched_term") or current.get("referenced_symbol")
        current["ordered_calls"] = item.get("ordered_calls") or current.get("ordered_calls")
        current["first_occurrence"] = item.get("first_occurrence") or current.get("first_occurrence")
        current["second_occurrence"] = item.get("second_occurrence") or current.get("second_occurrence")
        current["limitation"] = item.get("limitation") or current.get("limitation")
        merged[version_id] = current

    results = list(merged.values())
    for result in results:
        result["score"] += _symbol_path_bonus(result)
    results.sort(key=lambda r: (-float(r["score"]), r["path"], int(r["start_line"]), r["symbol"] or ""))
    return results


def _result_from_chunk(chunk: dict[str, Any], item: dict[str, Any], *, base_score: float, evidence_type: str) -> dict[str, Any]:
    return {
        "score": float(base_score),
        "path": chunk.get("path") or item.get("path"),
        "start_line": int(chunk.get("start_line") or item.get("start_line") or 1),
        "end_line": int(chunk.get("end_line") or item.get("end_line") or 1),
        "symbol": chunk.get("symbol") or item.get("symbol"),
        "symbol_type": chunk.get("symbol_type") or chunk.get("chunk_type") or item.get("symbol_type") or item.get("chunk_type"),
        "commit": item.get("commit") or chunk.get("commit"),
        "snippet": chunk.get("text") or item.get("snippet") or item.get("content_preview") or "",
        "evidence_type": evidence_type,
        "content_id": chunk.get("content_id") or item.get("content_id"),
        "version_id": chunk.get("version_id") or item.get("version_id"),
        "referenced_symbol": item.get("referenced_symbol"),
    }


def _rank_results(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ranked = []
    for rank, result in enumerate(results, start=1):
        ranked.append({"rank": rank, **result})
    return ranked


def _symbol_path_bonus(result: dict[str, Any]) -> float:
    if result.get("evidence_type") in {"reference", "structural", "mixed"}:
        return 0.05
    return 0.0


def _ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def _dedupe_structural(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, int, int]] = set()
    unique = []
    for item in items:
        key = (
            str(item.get("path")),
            str(item.get("symbol")),
            int(item.get("start_line", 0)),
            int(item.get("end_line", 0)),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
