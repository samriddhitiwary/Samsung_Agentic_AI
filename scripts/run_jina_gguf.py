"""Run AppsRetrieval with Jina Code Embeddings 0.5B GGUF via llama.cpp."""

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
import requests
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings(
    "ignore",
    message="`torch.jit.script` is deprecated.*",
    category=FutureWarning,
)

from src.evaluation.metrics import DEFAULT_K_VALUES, evaluate_rankings, metrics_payload
from src.retrieval.jina_code import DOCUMENT_INSTRUCTION, MODEL_NAME, QUERY_INSTRUCTION, row_text, token_length_stats
from src.retrieval.jina_gguf import (
    GGUF_FILE,
    GGUF_REPO,
    embed_http,
    retrieve_jina_code_gguf,
    truncate_with_tokenizer,
)
from src.utils.env import env_path, llama_server_url, load_dotenv


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
        "requests": version("requests"),
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


def check_server(server_url: str) -> None:
    response = requests.get(server_url.rstrip("/") + "/health", timeout=10)
    response.raise_for_status()


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    datasets.disable_progress_bars()

    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default=llama_server_url())
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--corpus-batch-size", type=int, default=8)
    parser.add_argument("--query-batch-size", type=int, default=8)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--skip-full", action="store_true")
    args = parser.parse_args()
    cache_dir = env_path(
        "JCR_GGUF_CACHE_DIR",
        PROJECT_ROOT / "data" / "cache" / "jina_code_0.5b_gguf",
        project_root=PROJECT_ROOT,
    )

    print("Checking llama.cpp embedding server...")
    check_server(args.server_url)
    print("Server ok")
    print("Current package versions:")
    print(json.dumps(package_versions(), indent=2, sort_keys=True))
    print(f"Selected model: {MODEL_NAME} via {GGUF_REPO}/{GGUF_FILE}")
    print(f"Query instruction: {QUERY_INSTRUCTION!r}")
    print(f"Document instruction: {DOCUMENT_INSTRUCTION!r}")

    task = mteb.get_task(TASK_NAME)
    task.load_data()
    split = task.dataset[SUBSET][SPLIT]
    corpus = dataset_to_rows(split["corpus"])
    queries = dataset_to_rows(split["queries"])
    qrels = qrels_to_dict(split["relevant_docs"])
    print(f"Loaded corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    stats = token_length_stats(
        tokenizer=tokenizer,
        texts=[row_text(row) for row in corpus],
        instruction=DOCUMENT_INSTRUCTION,
    )
    print("Corpus code/document token length stats:")
    print(json.dumps(stats, indent=2, sort_keys=True))
    print(f"Chosen max sequence length: {args.max_length}")

    if args.skip_smoke:
        print("Skipping GGUF smoke test by request.")
    else:
        smoke_prompts = [
            truncate_with_tokenizer(
                tokenizer=tokenizer,
                text=row_text(row),
                instruction=DOCUMENT_INSTRUCTION,
                max_length=args.max_length,
            )
            for row in corpus[:100]
        ]
        smoke_query_prompts = [
            truncate_with_tokenizer(
                tokenizer=tokenizer,
                text=row_text(row),
                instruction=QUERY_INSTRUCTION,
                max_length=args.max_length,
            )
            for row in queries[:5]
        ]
        smoke_docs = embed_http(
            texts=smoke_prompts,
            server_url=args.server_url,
            batch_size=args.corpus_batch_size,
            desc="GGUF smoke encode corpus",
        )
        smoke_queries = embed_http(
            texts=smoke_query_prompts,
            server_url=args.server_url,
            batch_size=args.query_batch_size,
            desc="GGUF smoke encode queries",
        )
        smoke_scores = smoke_queries @ smoke_docs.T
        smoke = {
            "model_loaded": True,
            "embedding_dimension": int(smoke_docs.shape[1]),
            "query_instruction": QUERY_INSTRUCTION,
            "document_instruction": DOCUMENT_INSTRUCTION,
            "instructions_differ": QUERY_INSTRUCTION != DOCUMENT_INSTRUCTION,
            "pooling_strategy": "last_token",
            "normalization": True,
            "finite_embeddings": bool(np_isfinite(smoke_docs) and np_isfinite(smoke_queries)),
            "finite_scores": bool(np_isfinite(smoke_scores)),
            "smoke_corpus_size": 100,
            "smoke_query_count": 5,
        }
        write_json(PROJECT_ROOT / "results" / "jina_code_0.5b" / "gguf_smoke_test.json", smoke)
        print("Smoke test:")
        print(json.dumps(smoke, indent=2, sort_keys=True))

    if args.skip_full:
        print("Skipping full benchmark by request.")
        return 0

    start = time.perf_counter()
    rankings, retrieval_metadata = retrieve_jina_code_gguf(
        queries=queries,
        corpus=corpus,
        cache_dir=cache_dir,
        server_url=args.server_url,
        max_length=args.max_length,
        corpus_batch_size=args.corpus_batch_size,
        query_batch_size=args.query_batch_size,
        top_k=TOP_K,
    )
    runtime = time.perf_counter() - start
    report = evaluate_rankings(rankings, qrels, k_values=DEFAULT_K_VALUES)
    buckets = rank_buckets(positive_ranks_at_100(rankings, qrels))
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
        "method": "jina_code_gguf_dense_cosine",
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
    print("\nJina Code 0.5B GGUF metrics:")
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


def np_isfinite(values: Any) -> bool:
    import numpy as np

    return bool(np.isfinite(values).all())


if __name__ == "__main__":
    raise SystemExit(main())
