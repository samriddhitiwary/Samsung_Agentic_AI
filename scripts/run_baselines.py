"""Run sparse and dense AppsRetrieval baselines without using qrels during retrieval."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import random
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
import re
from typing import Any

import datasets
import mteb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings(
    "ignore",
    message="`torch.jit.script` is deprecated.*",
    category=FutureWarning,
)

from src.evaluation.metrics import DEFAULT_K_VALUES, evaluate_rankings, metrics_payload
from src.retrieval.bm25 import retrieve_bm25
from src.retrieval.dense import DEFAULT_DENSE_MODEL, retrieve_dense


TASK_NAME = "AppsRetrieval"
SUBSET = "default"
SPLIT = "test"
TOP_K = 100


def version(distribution: str) -> str:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def dataset_to_rows(dataset: datasets.Dataset) -> list[dict[str, Any]]:
    return [dict(row) for row in dataset]


def qrels_to_dict(qrels: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {str(query_id): {str(doc_id): int(label) for doc_id, label in docs.items()} for query_id, docs in qrels.items()}


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


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
    }


def safe_run_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def save_run(
    *,
    run_name: str,
    rankings: dict[str, dict[str, float]],
    qrels: dict[str, dict[str, int]],
    metadata: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, int | None]]:
    report = evaluate_rankings(rankings, qrels, k_values=DEFAULT_K_VALUES)
    run_dir = PROJECT_ROOT / "results" / run_name
    write_json(
        run_dir / "top100.json",
        {
            "metadata": metadata,
            "rankings": rankings,
        },
    )
    payload = metrics_payload(report, metadata=metadata)
    write_json(run_dir / "metrics.json", payload)
    return payload, report.positive_ranks


def print_metric_line(name: str, metrics: dict[str, float]) -> None:
    print(
        f"{name:24s} "
        f"NDCG@10={metrics['ndcg_at_10']:.5f} "
        f"MRR@10={metrics['mrr_at_10']:.5f} "
        f"Recall@10={metrics['recall_at_10']:.5f} "
        f"HitRate@10={metrics['hitrate_at_10']:.5f}"
    )


def print_random_examples(
    *,
    queries: list[dict[str, Any]],
    rankings: dict[str, dict[str, float]],
    positive_ranks: dict[str, int | None],
    seed: int,
) -> None:
    rng = random.Random(seed)
    query_by_id = {row["id"]: row for row in queries}
    sample_ids = rng.sample(sorted(rankings), 5)
    print("\nRandom query inspection (positive rank is looked up only after retrieval):")
    for query_id in sample_ids:
        query_text = str(query_by_id[query_id].get("text") or "").replace("\r", " ").replace("\n", " ")
        if len(query_text) > 180:
            query_text = query_text[:180] + "... <truncated>"
        top5 = list(rankings[query_id])[:5]
        print(f"- {query_id}: {query_text}")
        print(f"  ranks 1-5: {top5}")
        print(f"  positive rank in top 10: {positive_ranks[query_id]}")


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    datasets.disable_progress_bars()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dense-model", default=DEFAULT_DENSE_MODEL)
    parser.add_argument("--bm25-tokenizer", default="code_split", choices=("basic", "code", "code_split"))
    parser.add_argument("--skip-bm25", action="store_true")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--dense-corpus-batch-size", type=int, default=8)
    parser.add_argument("--dense-query-batch-size", type=int, default=16)
    parser.add_argument("--dense-max-seq-length", type=int, default=128)
    parser.add_argument("--sample-seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading official AppsRetrieval benchmark...")
    task = mteb.get_task(TASK_NAME)
    task.load_data()
    split = task.dataset[SUBSET][SPLIT]
    corpus = dataset_to_rows(split["corpus"])
    queries = dataset_to_rows(split["queries"])
    qrels = qrels_to_dict(split["relevant_docs"])
    print(f"Loaded corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)}")
    print(f"Corpus columns: {split['corpus'].column_names}")
    print(f"Query columns: {split['queries'].column_names}")

    common_metadata = {
        "task_name": TASK_NAME,
        "dataset": task.metadata.dataset["path"],
        "dataset_revision": task.metadata.dataset["revision"],
        "subset": SUBSET,
        "split": SPLIT,
        "top_k": TOP_K,
        "corpus_size": len(corpus),
        "query_count": len(queries),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "package_versions": package_versions(),
    }

    summaries: dict[str, dict[str, Any]] = {}
    dense_rankings: dict[str, dict[str, float]] | None = None
    dense_positive_ranks: dict[str, int | None] | None = None

    if not args.skip_bm25:
        start = time.perf_counter()
        bm25_rankings = retrieve_bm25(
            queries=queries,
            corpus=corpus,
            tokenizer_name=args.bm25_tokenizer,
            top_k=TOP_K,
        )
        runtime = time.perf_counter() - start
        metadata = {
            **common_metadata,
            "run_name": f"bm25_{args.bm25_tokenizer}",
            "method": "BM25Okapi",
            "preprocessing": args.bm25_tokenizer,
            "runtime_seconds": runtime,
        }
        summaries["BM25"] = save_run(
            run_name=f"bm25_{args.bm25_tokenizer}",
            rankings=bm25_rankings,
            qrels=qrels,
            metadata=metadata,
        )[0]
        print_metric_line("BM25", summaries["BM25"]["metrics"])
        print(f"BM25 runtime seconds: {runtime:.2f}")

    if not args.skip_dense:
        start = time.perf_counter()
        dense_rankings, dense_extra = retrieve_dense(
            queries=queries,
            corpus=corpus,
            model_name=args.dense_model,
            cache_dir=PROJECT_ROOT / "data" / "cache",
            top_k=TOP_K,
            corpus_batch_size=args.dense_corpus_batch_size,
            query_batch_size=args.dense_query_batch_size,
            max_seq_length=args.dense_max_seq_length,
        )
        runtime = time.perf_counter() - start
        dense_run_name = f"dense_{safe_run_component(args.dense_model)}"
        metadata = {
            **common_metadata,
            "run_name": dense_run_name,
            "method": "dense_cosine",
            "model": args.dense_model,
            "preprocessing": "title + text, SentenceTransformer tokenizer, normalized cosine",
            "max_seq_length": args.dense_max_seq_length,
            "runtime_seconds": runtime,
            **dense_extra,
        }
        summaries["Dense model"], dense_positive_ranks = save_run(
            run_name=dense_run_name,
            rankings=dense_rankings,
            qrels=qrels,
            metadata=metadata,
        )
        print_metric_line("Dense model", summaries["Dense model"]["metrics"])
        print(f"Dense runtime seconds: {runtime:.2f}")
        print(f"Dense corpus embeddings cached: {dense_extra['corpus_embedding_cache_hit']}")

    print("\nComparison:")
    print("| Method | NDCG@10 | MRR@10 | Recall@10 | HitRate@10 |")
    print("| --- | ---: | ---: | ---: | ---: |")
    for name, payload in summaries.items():
        metrics = payload["metrics"]
        print(
            f"| {name} | {metrics['ndcg_at_10']:.5f} | {metrics['mrr_at_10']:.5f} | "
            f"{metrics['recall_at_10']:.5f} | {metrics['hitrate_at_10']:.5f} |"
        )

    print("\nAdditional cutoffs:")
    for name, payload in summaries.items():
        metrics = payload["metrics"]
        print(
            f"{name}: "
            f"NDCG@1/3/5={metrics['ndcg_at_1']:.5f}/{metrics['ndcg_at_3']:.5f}/{metrics['ndcg_at_5']:.5f}; "
            f"MRR@1/3/5={metrics['mrr_at_1']:.5f}/{metrics['mrr_at_3']:.5f}/{metrics['mrr_at_5']:.5f}"
        )
        print(f"{name} sanity checks: {payload['sanity_checks']}")

    if dense_rankings is not None and dense_positive_ranks is not None:
        print_random_examples(
            queries=queries,
            rankings=dense_rankings,
            positive_ranks=dense_positive_ranks,
            seed=args.sample_seed,
        )

    print("\nDone. Rankings and metrics saved under results/<run_name>/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
