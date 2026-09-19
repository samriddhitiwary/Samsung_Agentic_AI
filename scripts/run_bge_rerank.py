"""Rerank Jina Code 0.5B Q8 top-20 results with BAAI/bge-reranker-v2-m3."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import datasets
import mteb
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import DEFAULT_K_VALUES, evaluate_rankings, metrics_payload, positive_ranks_at_k
from src.reranking.bge import MODEL_NAME as BGE_MODEL_NAME
from src.reranking.bge import ONNX_MODEL_FILE, ONNX_MODEL_NAME
from src.reranking.bge import load_bge_onnx_reranker, load_bge_reranker, timed_score_pairs, timed_score_pairs_onnx
from src.retrieval.jina_code import row_text


TASK_NAME = "AppsRetrieval"
SUBSET = "default"
SPLIT = "test"
BASELINE_RUN = PROJECT_ROOT / "results" / "jina_code_0.5b_q8"
OUTPUT_RUN = PROJECT_ROOT / "results" / "jina_code_0.5b_q8_bge_rerank"
RERANK_DEPTH = 20
TOP_K = 100


def version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "datasets": version("datasets"),
        "mteb": version("mteb"),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "tqdm": version("tqdm"),
    }


def dataset_to_rows(dataset: datasets.Dataset) -> list[dict[str, Any]]:
    return [dict(row) for row in dataset]


def qrels_to_dict(qrels: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        str(query_id): {str(doc_id): int(label) for doc_id, label in docs.items()}
        for query_id, docs in qrels.items()
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def ranked_items(scores: dict[str, float], limit: int = TOP_K) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda item: (item[1], item[0]), reverse=True)[:limit]


def rank_buckets(positive_ranks: dict[str, int | None]) -> dict[str, dict[str, float]]:
    buckets = Counter()
    for rank in positive_ranks.values():
        if rank == 1:
            buckets["rank_1"] += 1
        elif rank is not None and rank <= 3:
            buckets["rank_2_3"] += 1
        elif rank is not None and rank <= 10:
            buckets["rank_4_10"] += 1
        elif rank is not None and rank <= 20:
            buckets["rank_11_20"] += 1
        elif rank is not None and rank <= 100:
            buckets["rank_21_100"] += 1
        else:
            buckets["not_in_top_100"] += 1
    total = len(positive_ranks)
    order = ["rank_1", "rank_2_3", "rank_4_10", "rank_11_20", "rank_21_100", "not_in_top_100"]
    return {
        name: {"count": int(buckets[name]), "percent": float(buckets[name] / total * 100)}
        for name in order
    }


def movement_counts(
    before_ranks: dict[str, int | None],
    after_ranks: dict[str, int | None],
) -> dict[str, int]:
    rank_2_10_to_1 = sum(
        before_ranks[qid] is not None
        and 2 <= int(before_ranks[qid]) <= 10
        and after_ranks[qid] == 1
        for qid in before_ranks
    )
    rank_1_moved_down = sum(before_ranks[qid] == 1 and after_ranks[qid] != 1 for qid in before_ranks)
    before_rank_1 = sum(rank == 1 for rank in before_ranks.values())
    after_rank_1 = sum(rank == 1 for rank in after_ranks.values())
    return {
        "rank_2_10_positives_moved_to_rank_1": int(rank_2_10_to_1),
        "rank_1_positives_incorrectly_moved_down": int(rank_1_moved_down),
        "baseline_rank_1_count": int(before_rank_1),
        "reranked_rank_1_count": int(after_rank_1),
        "net_change_in_rank_1_accuracy_count": int(after_rank_1 - before_rank_1),
    }


def build_pairs_for_query(
    *,
    query_id: str,
    baseline_rankings: dict[str, dict[str, float]],
    query_text_by_id: dict[str, str],
    doc_text_by_id: dict[str, str],
    depth: int,
) -> tuple[list[str], list[tuple[str, str]], list[tuple[str, float]]]:
    ordered = ranked_items(baseline_rankings[query_id], limit=TOP_K)
    top_items = ordered[:depth]
    top_doc_ids = [doc_id for doc_id, _ in top_items]
    pairs = [(query_text_by_id[query_id], doc_text_by_id[doc_id]) for doc_id in top_doc_ids]
    return top_doc_ids, pairs, ordered


def merge_reranked_top(
    *,
    original_ordered: list[tuple[str, float]],
    top_doc_ids: list[str],
    reranker_scores: list[float],
    depth: int,
) -> dict[str, float]:
    top_set = set(top_doc_ids)
    reranked_top = sorted(
        zip(top_doc_ids, reranker_scores, strict=True),
        key=lambda item: (float(item[1]), item[0]),
        reverse=True,
    )
    tail = [(doc_id, score) for doc_id, score in original_ordered if doc_id not in top_set]

    # Use monotonic synthetic ranking scores so top-20 remains above ranks 21-100.
    merged: dict[str, float] = {}
    next_score = float(TOP_K + depth)
    for rank, (doc_id, _score) in enumerate(reranked_top, start=1):
        merged[doc_id] = next_score - rank
    tail_start = next_score - depth - 1
    for offset, (doc_id, _score) in enumerate(tail[: TOP_K - depth]):
        merged[doc_id] = tail_start - offset
    return merged


def main() -> int:
    datasets.disable_progress_bars()

    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--smoke-queries", type=int, default=10)
    parser.add_argument("--depth", type=int, default=RERANK_DEPTH)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--backend", choices=("onnx", "transformers"), default="onnx")
    args = parser.parse_args()

    baseline_top100_path = BASELINE_RUN / "top100.json"
    baseline_metrics_path = BASELINE_RUN / "metrics.json"
    baseline_payload = json.loads(baseline_top100_path.read_text(encoding="utf-8"))
    baseline_metrics = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    baseline_rankings: dict[str, dict[str, float]] = baseline_payload["rankings"]

    print(f"Loading official {TASK_NAME} data...")
    task = mteb.get_task(TASK_NAME)
    task.load_data()
    split = task.dataset[SUBSET][SPLIT]
    corpus = dataset_to_rows(split["corpus"])
    queries = dataset_to_rows(split["queries"])
    qrels = qrels_to_dict(split["relevant_docs"])
    doc_text_by_id = {row["id"]: row_text(row) for row in corpus}
    query_text_by_id = {row["id"]: row_text(row) for row in queries}
    print(f"Loaded corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)}")

    if args.backend == "onnx":
        print(f"Loading reranker: {ONNX_MODEL_NAME} / {ONNX_MODEL_FILE}")
        reranker = load_bge_onnx_reranker(
            model_name=ONNX_MODEL_NAME,
            onnx_file=ONNX_MODEL_FILE,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        score_fn = timed_score_pairs_onnx
        reranker_model = ONNX_MODEL_NAME
        reranker_model_file = ONNX_MODEL_FILE
        reranker_implementation = "onnxruntime CPUExecutionProvider"
    else:
        print(f"Loading reranker: {BGE_MODEL_NAME}")
        reranker = load_bge_reranker(
            model_name=BGE_MODEL_NAME,
            max_length=args.max_length,
            batch_size=args.batch_size,
            device="cpu",
        )
        score_fn = timed_score_pairs
        reranker_model = BGE_MODEL_NAME
        reranker_model_file = None
        reranker_implementation = "transformers AutoModelForSequenceClassification"

    smoke_query_ids = list(baseline_rankings)[: args.smoke_queries]
    smoke_pairs: list[tuple[str, str]] = []
    for query_id in smoke_query_ids:
        _top_doc_ids, pairs, _ordered = build_pairs_for_query(
            query_id=query_id,
            baseline_rankings=baseline_rankings,
            query_text_by_id=query_text_by_id,
            doc_text_by_id=doc_text_by_id,
            depth=args.depth,
        )
        smoke_pairs.extend(pairs)

    smoke_scores, smoke_runtime = score_fn(
        reranker,
        smoke_pairs,
        desc=f"BGE smoke {len(smoke_query_ids)}x{args.depth}",
    )
    finite_scores = all(math.isfinite(score) for score in smoke_scores)
    smoke = {
        "model_loaded": True,
        "model_name": reranker_model,
        "model_file": reranker_model_file,
        "backend": args.backend,
        "device": "cpu",
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "smoke_queries": len(smoke_query_ids),
        "rerank_depth": args.depth,
        "pair_count": len(smoke_pairs),
        "finite_scores": finite_scores,
        "runtime_seconds": smoke_runtime,
        "seconds_per_pair": smoke_runtime / max(1, len(smoke_pairs)),
    }
    write_json(OUTPUT_RUN / "smoke_test.json", smoke)
    print("Smoke test:")
    print(json.dumps(smoke, indent=2, sort_keys=True))
    if not finite_scores:
        raise ValueError("Smoke test produced non-finite reranker scores")
    if args.smoke_only:
        print("Stopping after smoke test by request.")
        return 0

    print("Reranking top-20 for all queries...")
    start = time.perf_counter()
    reranked: dict[str, dict[str, float]] = {}
    all_query_ids = list(baseline_rankings)
    for query_id in tqdm(all_query_ids, desc="BGE rerank queries"):
        top_doc_ids, pairs, ordered = build_pairs_for_query(
            query_id=query_id,
            baseline_rankings=baseline_rankings,
            query_text_by_id=query_text_by_id,
            doc_text_by_id=doc_text_by_id,
            depth=args.depth,
        )
        scores = score_fn(reranker, pairs, desc=f"BGE score {query_id}")[0]
        reranked[query_id] = merge_reranked_top(
            original_ordered=ordered,
            top_doc_ids=top_doc_ids,
            reranker_scores=scores,
            depth=args.depth,
        )
    runtime = time.perf_counter() - start

    report = evaluate_rankings(reranked, qrels, k_values=DEFAULT_K_VALUES)
    baseline_ranks_100 = positive_ranks_at_k(baseline_rankings, qrels, k=TOP_K)
    reranked_ranks_100 = positive_ranks_at_k(reranked, qrels, k=TOP_K)
    baseline_buckets = rank_buckets(baseline_ranks_100)
    reranked_buckets = rank_buckets(reranked_ranks_100)
    movements = movement_counts(baseline_ranks_100, reranked_ranks_100)

    metadata = {
        "run_name": "jina_code_0.5b_q8_bge_rerank",
        "baseline_run": str(BASELINE_RUN),
        "baseline_top100": str(baseline_top100_path),
        "baseline_metrics": {
            "ndcg_at_10": baseline_metrics["metrics"]["ndcg_at_10"],
            "mrr_at_10": baseline_metrics["metrics"]["mrr_at_10"],
            "hitrate_at_10": baseline_metrics["metrics"]["hitrate_at_10"],
        },
        "task_name": TASK_NAME,
        "subset": SUBSET,
        "split": SPLIT,
        "top_k": TOP_K,
        "rerank_depth": args.depth,
        "candidate_source": "Jina Code 0.5B Q8 top100, top20 reranked only",
        "reranker_model": reranker_model,
        "reranker_model_file": reranker_model_file,
        "reranker_implementation": reranker_implementation,
        "reranker_device": "cpu",
        "max_sequence_length": args.max_length,
        "batch_size": args.batch_size,
        "query_count": len(queries),
        "corpus_size": len(corpus),
        "runtime_seconds": runtime,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_versions": package_versions(),
        "smoke_test": smoke,
    }

    write_json(OUTPUT_RUN / "top100.json", {"metadata": metadata, "rankings": reranked})
    payload = metrics_payload(report, metadata=metadata)
    payload["rank_buckets_before"] = baseline_buckets
    payload["rank_buckets_after"] = reranked_buckets
    payload["movement_counts"] = movements
    write_json(OUTPUT_RUN / "metrics.json", payload)

    metrics = payload["metrics"]
    print("\nJina + BGE reranker metrics:")
    for key in (
        "ndcg_at_1",
        "ndcg_at_3",
        "ndcg_at_5",
        "ndcg_at_10",
        "mrr_at_1",
        "mrr_at_3",
        "mrr_at_5",
        "mrr_at_10",
        "recall_at_10",
        "hitrate_at_10",
    ):
        print(f"{key}: {metrics[key]:.5f}")
    print(f"Sanity checks: {payload['sanity_checks']}")
    print("Rank buckets before:")
    print(json.dumps(baseline_buckets, indent=2, sort_keys=True))
    print("Rank buckets after:")
    print(json.dumps(reranked_buckets, indent=2, sort_keys=True))
    print("Movement counts:")
    print(json.dumps(movements, indent=2, sort_keys=True))
    print(f"Runtime seconds: {runtime:.2f}")
    print("Saved results/jina_code_0.5b_q8_bge_rerank/metrics.json and top100.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
