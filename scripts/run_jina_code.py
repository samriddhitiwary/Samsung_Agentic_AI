"""Run Jina Code Embeddings 0.5B on AppsRetrieval."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
import warnings
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import datasets
import mteb
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings(
    "ignore",
    message="`torch.jit.script` is deprecated.*",
    category=FutureWarning,
)

from src.evaluation.metrics import DEFAULT_K_VALUES, evaluate_rankings, metrics_payload
from src.retrieval.jina_code import (
    DOCUMENT_INSTRUCTION,
    MODEL_NAME,
    POOLING_STRATEGY,
    QUERY_INSTRUCTION,
    add_document_instruction,
    add_query_instruction,
    encode_corpus_cached,
    encode_texts,
    load_jina_model,
    retrieve_jina_code,
    row_text,
    token_length_stats,
)


TASK_NAME = "AppsRetrieval"
SUBSET = "default"
SPLIT = "test"
TOP_K = 100


def version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def package_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "mteb": version("mteb"),
        "datasets": version("datasets"),
        "numpy": version("numpy"),
        "bm25s": version("bm25s"),
        "sentence-transformers": version("sentence-transformers"),
        "transformers": version("transformers"),
        "torch": version("torch"),
        "protobuf": version("protobuf"),
        "safetensors": version("safetensors"),
        "huggingface-hub": version("huggingface-hub"),
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


def rank_buckets(positive_ranks_top100: dict[str, int | None]) -> dict[str, dict[str, float]]:
    buckets = Counter()
    for rank in positive_ranks_top100.values():
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

    total = len(positive_ranks_top100)
    order = [
        "rank_1",
        "rank_2_3",
        "rank_4_10",
        "rank_11_20",
        "rank_21_100",
        "not_in_top_100",
    ]
    return {
        name: {"count": int(buckets[name]), "percent": float(buckets[name] / total * 100)}
        for name in order
    }


def positive_ranks_at_100(
    rankings: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
) -> dict[str, int | None]:
    ranks: dict[str, int | None] = {}
    for query_id, rels in qrels.items():
        positive_doc = next(doc_id for doc_id, label in rels.items() if label > 0)
        ranked = sorted(rankings[query_id].items(), key=lambda item: (item[1], item[0]), reverse=True)
        doc_ids = [doc_id for doc_id, _ in ranked[:100]]
        ranks[query_id] = doc_ids.index(positive_doc) + 1 if positive_doc in doc_ids else None
    return ranks


def smoke_test(
    *,
    corpus: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    max_length: int,
    corpus_batch_size: int,
    query_batch_size: int,
    cache_dir: Path,
    local_files_only: bool,
) -> dict[str, Any]:
    tokenizer, model, device, revision = load_jina_model(local_files_only=local_files_only)
    smoke_corpus = corpus[:100]
    smoke_queries = queries[:5]
    doc_ids, doc_embeddings, cache_hit, cache_path, cache_metadata = encode_corpus_cached(
        corpus_rows=smoke_corpus,
        tokenizer=tokenizer,
        model=model,
        device=device,
        model_name=MODEL_NAME,
        model_revision=revision,
        max_length=max_length,
        batch_size=corpus_batch_size,
        cache_dir=cache_dir / "smoke",
    )
    query_embeddings = encode_texts(
        texts=[row_text(row) for row in smoke_queries],
        tokenizer=tokenizer,
        model=model,
        device=device,
        instruction=QUERY_INSTRUCTION,
        max_length=max_length,
        batch_size=query_batch_size,
        desc="Jina smoke encode queries",
    )
    scores = query_embeddings @ doc_embeddings.T
    top_idx = np.argsort(scores, axis=1)[:, ::-1][:, :5]
    return {
        "model_loaded": True,
        "device": str(device),
        "model_revision": revision,
        "embedding_dimension": int(doc_embeddings.shape[1]),
        "query_instruction": QUERY_INSTRUCTION,
        "document_instruction": DOCUMENT_INSTRUCTION,
        "instructions_differ": add_query_instruction("x") != add_document_instruction("x"),
        "pooling_strategy": POOLING_STRATEGY,
        "smoke_corpus_size": len(smoke_corpus),
        "smoke_query_count": len(smoke_queries),
        "document_embedding_shape": list(doc_embeddings.shape),
        "query_embedding_shape": list(query_embeddings.shape),
        "finite_embeddings": bool(np.isfinite(doc_embeddings).all() and np.isfinite(query_embeddings).all()),
        "finite_scores": bool(np.isfinite(scores).all()),
        "cache_written": cache_path.exists(),
        "cache_hit": cache_hit,
        "cache_path": str(cache_path),
        "sample_top5_doc_ids": {
            smoke_queries[i]["id"]: [doc_ids[int(index)] for index in top_idx[i]]
            for i in range(len(smoke_queries))
        },
        "cache_metadata": cache_metadata,
    }


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    datasets.disable_progress_bars()

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--corpus-batch-size", type=int, default=2)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    print("Current package versions:")
    print(json.dumps(package_versions(), indent=2, sort_keys=True))
    print(f"Selected model: {MODEL_NAME}")
    print(f"Query instruction: {QUERY_INSTRUCTION!r}")
    print(f"Document instruction: {DOCUMENT_INSTRUCTION!r}")

    task = mteb.get_task(TASK_NAME)
    task.load_data()
    split = task.dataset[SUBSET][SPLIT]
    corpus = dataset_to_rows(split["corpus"])
    queries = dataset_to_rows(split["queries"])
    qrels = qrels_to_dict(split["relevant_docs"])
    print(f"Loaded corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)}")

    tokenizer, _, _, _ = load_jina_model(local_files_only=args.local_files_only)
    stats = token_length_stats(
        tokenizer=tokenizer,
        texts=[row_text(row) for row in corpus],
        instruction=DOCUMENT_INSTRUCTION,
    )
    print("Corpus code/document token length stats:")
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Chosen max sequence length: {args.max_length}")

    smoke = smoke_test(
        corpus=corpus,
        queries=queries,
        max_length=args.max_length,
        corpus_batch_size=args.corpus_batch_size,
        query_batch_size=args.query_batch_size,
        cache_dir=PROJECT_ROOT / "data" / "cache" / "jina_code_0.5b",
        local_files_only=args.local_files_only,
    )
    write_json(PROJECT_ROOT / "results" / "jina_code_0.5b" / "smoke_test.json", smoke)
    print("Smoke test:")
    print(json.dumps(smoke, indent=2, sort_keys=True))

    if args.skip_full:
        print("Skipping full benchmark by request.")
        return 0

    start = time.perf_counter()
    rankings, retrieval_metadata = retrieve_jina_code(
        queries=queries,
        corpus=corpus,
        cache_dir=PROJECT_ROOT / "data" / "cache" / "jina_code_0.5b",
        max_length=args.max_length,
        corpus_batch_size=args.corpus_batch_size,
        query_batch_size=args.query_batch_size,
        top_k=TOP_K,
        local_files_only=args.local_files_only,
    )
    runtime = time.perf_counter() - start
    report = evaluate_rankings(rankings, qrels, k_values=DEFAULT_K_VALUES)
    positive_ranks_top100 = positive_ranks_at_100(rankings, qrels)
    buckets = rank_buckets(positive_ranks_top100)

    metadata = {
        "run_name": "jina_code_0.5b",
        "task_name": TASK_NAME,
        "dataset": task.metadata.dataset["path"],
        "dataset_revision": task.metadata.dataset["revision"],
        "subset": SUBSET,
        "split": SPLIT,
        "top_k": TOP_K,
        "corpus_size": len(corpus),
        "query_count": len(queries),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": runtime,
        "method": "jina_code_dense_cosine",
        "model": MODEL_NAME,
        "preprocessing": "title + text with official nl2code query/passage instructions",
        "sequence_length_stats": stats,
        "chosen_max_sequence_length": args.max_length,
        "package_versions": package_versions(),
        **retrieval_metadata,
    }
    run_dir = PROJECT_ROOT / "results" / "jina_code_0.5b"
    write_json(run_dir / "top100.json", {"metadata": metadata, "rankings": rankings})
    payload = metrics_payload(report, metadata=metadata)
    payload["rank_buckets"] = buckets
    write_json(run_dir / "metrics.json", payload)

    metrics = payload["metrics"]
    print("\nJina Code 0.5B metrics:")
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
    print("Rank buckets:")
    print(json.dumps(buckets, indent=2, sort_keys=True))
    print(f"Runtime seconds: {runtime:.2f}")
    print(f"Embeddings cached: {retrieval_metadata['corpus_embedding_cache_hit']}")
    print("Saved results/jina_code_0.5b/metrics.json and top100.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
