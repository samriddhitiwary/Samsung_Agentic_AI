"""Run full first-stage AppsRetrieval with Jina Code Embeddings 1.5B GGUF.

This is a controlled experiment:
  - jinaai/jina-code-embeddings-1.5b
  - jina-code-embeddings-1.5b-Q8_0.gguf
  - max sequence length 1024
  - all corpus documents as candidates
  - top-100 retrieval for every test query
"""

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
import requests
from tqdm import tqdm
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

warnings.filterwarnings(
    "ignore",
    message="`torch.jit.script` is deprecated.*",
    category=FutureWarning,
)

from src.evaluation.metrics import evaluate_rankings, metrics_payload, positive_ranks_at_k
from src.retrieval.jina_code import (
    DOCUMENT_INSTRUCTION,
    NORMALIZE_EMBEDDINGS,
    POOLING_STRATEGY,
    QUERY_INSTRUCTION,
    corpus_hash,
    row_text,
)
from src.retrieval.jina_gguf import embed_http_cached, model_slug, prompt_hash, truncate_with_tokenizer
from src.utils.env import env_path, llama_server_url, load_dotenv


TASK_NAME = "AppsRetrieval"
SUBSET = "default"
SPLIT = "test"
TOP_K = 100
K_VALUES = (1, 3, 5, 10, 20, 100)

RUN_NAME = "jina_code_1.5b_full_1024"
MODEL_NAME = "jinaai/jina-code-embeddings-1.5b"
GGUF_REPO = "jinaai/jina-code-embeddings-1.5b-GGUF"
GGUF_FILE = "jina-code-embeddings-1.5b-Q8_0.gguf"
QUANTIZATION = "Q8_0"
EXPECTED_EMBEDDING_DIMENSION = 1536
MAX_LENGTH = 1024

DEFAULT_CACHE_DIR = PROJECT_ROOT / "data" / "cache" / RUN_NAME
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / RUN_NAME
DEFAULT_MODEL_FILE = PROJECT_ROOT / "data" / "cache" / "model_files" / GGUF_FILE


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
        "requests": version("requests"),
        "torch": version("torch"),
        "transformers": version("transformers"),
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


def check_server(server_url: str) -> None:
    response = requests.get(server_url.rstrip("/") + "/health", timeout=10)
    response.raise_for_status()


def finite_array(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains non-finite values")


def validate_embeddings(name: str, values: np.ndarray) -> None:
    finite_array(name, values)
    if values.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {values.shape}")
    if values.shape[1] != EXPECTED_EMBEDDING_DIMENSION:
        raise ValueError(
            f"{name} embedding dimension mismatch: "
            f"expected {EXPECTED_EMBEDDING_DIMENSION}, got {values.shape[1]}"
        )


def enrich_embedding_cache_metadata(
    *,
    embedding_path: Path,
    role: str,
    ids: list[str],
    prompts: list[str],
    shape: tuple[int, int],
    corpus_fingerprint: str,
    query_count: int,
    corpus_count: int,
) -> None:
    meta_path = embedding_path.with_suffix(".meta.json")
    metadata: dict[str, Any] = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    metadata.update(
        {
            "ids": ids,
            "prompt_hash": prompt_hash(ids, prompts),
            "shape": [int(shape[0]), int(shape[1])],
            "done_count": int(shape[0]),
            "role": role,
            "model": MODEL_NAME,
            "model_name": MODEL_NAME,
            "gguf_repo": GGUF_REPO,
            "gguf_file": GGUF_FILE,
            "quantization": QUANTIZATION,
            "embedding_dimension": int(shape[1]),
            "expected_embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
            "max_sequence_length": MAX_LENGTH,
            "query_instruction": QUERY_INSTRUCTION,
            "passage_instruction": DOCUMENT_INSTRUCTION,
            "document_instruction": DOCUMENT_INSTRUCTION,
            "pooling": POOLING_STRATEGY,
            "pooling_strategy": POOLING_STRATEGY,
            "normalization": NORMALIZE_EMBEDDINGS,
            "corpus_fingerprint": corpus_fingerprint,
            "corpus_count": corpus_count,
            "query_count": query_count,
            "transport": "llama.cpp HTTP /embedding",
        }
    )
    meta_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def prepare_prompts(
    *,
    rows: list[dict[str, Any]],
    tokenizer: Any,
    instruction: str,
    max_length: int,
    desc: str,
) -> tuple[list[str], list[str]]:
    ids: list[str] = []
    prompts: list[str] = []
    for row in tqdm(rows, desc=desc):
        ids.append(str(row["id"]))
        prompts.append(
            truncate_with_tokenizer(
                tokenizer=tokenizer,
                text=row_text(row),
                instruction=instruction,
                max_length=max_length,
            )
        )
    return ids, prompts


def embed_with_cache(
    *,
    ids: list[str],
    prompts: list[str],
    role: str,
    cache_dir: Path,
    server_url: str,
    batch_size: int,
    corpus_fingerprint: str,
    query_count: int,
    corpus_count: int,
    desc: str,
) -> tuple[np.ndarray, bool, Path, float]:
    cache_stem = (
        f"{model_slug(GGUF_FILE)}_full1024_{role}_last_maxseq-{MAX_LENGTH}_"
        f"{prompt_hash(ids, prompts)}"
    )
    start = time.perf_counter()
    embeddings, cache_hit, cache_path = embed_http_cached(
        ids=ids,
        texts=prompts,
        server_url=server_url,
        batch_size=batch_size,
        desc=desc,
        cache_dir=cache_dir,
        cache_stem=cache_stem,
    )
    runtime = time.perf_counter() - start
    validate_embeddings(role, embeddings)
    enrich_embedding_cache_metadata(
        embedding_path=cache_path,
        role=role,
        ids=ids,
        prompts=prompts,
        shape=embeddings.shape,
        corpus_fingerprint=corpus_fingerprint,
        query_count=query_count,
        corpus_count=corpus_count,
    )
    return embeddings, cache_hit, cache_path, runtime


def retrieve_top100(
    *,
    query_ids: list[str],
    doc_ids: list[str],
    query_embeddings: np.ndarray,
    corpus_embeddings: np.ndarray,
    similarity_batch_size: int,
    top_k: int,
) -> dict[str, dict[str, float]]:
    top_k = min(top_k, len(doc_ids))
    results: dict[str, dict[str, float]] = {}
    corpus_matrix = corpus_embeddings.T
    for start in tqdm(range(0, len(query_ids), similarity_batch_size), desc="1.5B full top-k search"):
        end = min(start + similarity_batch_size, len(query_ids))
        scores = query_embeddings[start:end] @ corpus_matrix
        finite_array("similarity scores", scores)
        candidate_idx = np.argpartition(scores, -top_k, axis=1)[:, -top_k:]
        candidate_scores = np.take_along_axis(scores, candidate_idx, axis=1)
        order = np.argsort(candidate_scores, axis=1)[:, ::-1]
        sorted_idx = np.take_along_axis(candidate_idx, order, axis=1)
        sorted_scores = np.take_along_axis(candidate_scores, order, axis=1)
        for offset, query_id in enumerate(query_ids[start:end]):
            results[query_id] = {
                doc_ids[int(index)]: float(sorted_scores[offset, rank])
                for rank, index in enumerate(sorted_idx[offset])
            }
    return results


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


def run_embedding_and_retrieval(
    *,
    corpus: list[dict[str, Any]],
    queries: list[dict[str, Any]],
    tokenizer: Any,
    cache_dir: Path,
    server_url: str,
    corpus_batch_size: int,
    query_batch_size: int,
    similarity_batch_size: int,
    corpus_fingerprint: str,
    corpus_desc_prefix: str,
    query_desc_prefix: str,
    search_desc: str | None = None,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    del search_desc
    prompt_start = time.perf_counter()
    doc_ids, corpus_prompts = prepare_prompts(
        rows=corpus,
        tokenizer=tokenizer,
        instruction=DOCUMENT_INSTRUCTION,
        max_length=MAX_LENGTH,
        desc=f"{corpus_desc_prefix} prepare corpus prompts",
    )
    corpus_prompt_runtime = time.perf_counter() - prompt_start

    query_prompt_start = time.perf_counter()
    query_ids, query_prompts = prepare_prompts(
        rows=queries,
        tokenizer=tokenizer,
        instruction=QUERY_INSTRUCTION,
        max_length=MAX_LENGTH,
        desc=f"{query_desc_prefix} prepare query prompts",
    )
    query_prompt_runtime = time.perf_counter() - query_prompt_start

    corpus_embeddings, corpus_cache_hit, corpus_cache_path, corpus_embedding_runtime = embed_with_cache(
        ids=doc_ids,
        prompts=corpus_prompts,
        role="corpus_passage",
        cache_dir=cache_dir,
        server_url=server_url,
        batch_size=corpus_batch_size,
        corpus_fingerprint=corpus_fingerprint,
        query_count=len(queries),
        corpus_count=len(corpus),
        desc=f"{corpus_desc_prefix} encode corpus",
    )
    query_embeddings, query_cache_hit, query_cache_path, query_embedding_runtime = embed_with_cache(
        ids=query_ids,
        prompts=query_prompts,
        role="query_nl2code",
        cache_dir=cache_dir,
        server_url=server_url,
        batch_size=query_batch_size,
        corpus_fingerprint=corpus_fingerprint,
        query_count=len(queries),
        corpus_count=len(corpus),
        desc=f"{query_desc_prefix} encode queries",
    )

    retrieval_start = time.perf_counter()
    rankings = retrieve_top100(
        query_ids=query_ids,
        doc_ids=doc_ids,
        query_embeddings=query_embeddings,
        corpus_embeddings=corpus_embeddings,
        similarity_batch_size=similarity_batch_size,
        top_k=TOP_K,
    )
    retrieval_runtime = time.perf_counter() - retrieval_start

    metadata = {
        "corpus_prompt_runtime_seconds": corpus_prompt_runtime,
        "query_prompt_runtime_seconds": query_prompt_runtime,
        "corpus_encoding_runtime_seconds": corpus_embedding_runtime,
        "query_encoding_runtime_seconds": query_embedding_runtime,
        "retrieval_runtime_seconds": retrieval_runtime,
        "corpus_embedding_cache_hit": corpus_cache_hit,
        "query_embedding_cache_hit": query_cache_hit,
        "corpus_embedding_cache_path": str(corpus_cache_path),
        "query_embedding_cache_path": str(query_cache_path),
        "corpus_embedding_shape": list(corpus_embeddings.shape),
        "query_embedding_shape": list(query_embeddings.shape),
        "finite_embeddings": True,
    }
    return rankings, metadata


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    datasets.disable_progress_bars()

    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default=llama_server_url())
    parser.add_argument("--corpus-batch-size", type=int, default=4)
    parser.add_argument("--query-batch-size", type=int, default=4)
    parser.add_argument("--similarity-batch-size", type=int, default=64)
    parser.add_argument("--skip-smoke", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    args = parser.parse_args()

    cache_dir = env_path(
        "JCR_JINA_CODE_1_5B_FULL_1024_CACHE_DIR",
        DEFAULT_CACHE_DIR,
        project_root=PROJECT_ROOT,
    )
    results_dir = DEFAULT_RESULTS_DIR
    model_file = env_path(
        "JCR_JINA_CODE_1_5B_GGUF_PATH",
        DEFAULT_MODEL_FILE,
        project_root=PROJECT_ROOT,
    )

    print("Checking llama.cpp embedding server...")
    check_server(args.server_url)
    print("Server ok")
    if not model_file.exists():
        raise FileNotFoundError(f"Expected local GGUF file is missing: {model_file}")
    print(f"Using existing local GGUF: {model_file}")
    print(f"Model: {MODEL_NAME}")
    print(f"GGUF: {GGUF_REPO}/{GGUF_FILE}")
    print(f"Max sequence length: {MAX_LENGTH}")
    print(f"Expected embedding dimension: {EXPECTED_EMBEDDING_DIMENSION}")
    print("Package versions:")
    print(json.dumps(package_versions(), indent=2, sort_keys=True))

    task = mteb.get_task(TASK_NAME)
    task.load_data()
    split = task.dataset[SUBSET][SPLIT]
    corpus = dataset_to_rows(split["corpus"])
    queries = dataset_to_rows(split["queries"])
    qrels = qrels_to_dict(split["relevant_docs"])
    corpus_fingerprint = corpus_hash(corpus)
    print(f"Loaded corpus={len(corpus)} queries={len(queries)} qrels={len(qrels)}")
    print(f"Corpus fingerprint: {corpus_fingerprint}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if not args.skip_smoke:
        print("Running smoke test: 100 corpus documents x 20 queries")
        smoke_start = time.perf_counter()
        smoke_rankings, smoke_metadata = run_embedding_and_retrieval(
            corpus=corpus[:100],
            queries=queries[:20],
            tokenizer=tokenizer,
            cache_dir=cache_dir / "smoke",
            server_url=args.server_url,
            corpus_batch_size=args.corpus_batch_size,
            query_batch_size=args.query_batch_size,
            similarity_batch_size=args.similarity_batch_size,
            corpus_fingerprint=corpus_fingerprint,
            corpus_desc_prefix="smoke 1.5B full1024",
            query_desc_prefix="smoke 1.5B full1024",
        )
        smoke_runtime = time.perf_counter() - smoke_start
        smoke_scores = [score for ranking in smoke_rankings.values() for score in ranking.values()]
        smoke = {
            "model_loaded": True,
            "model": MODEL_NAME,
            "gguf_file": GGUF_FILE,
            "quantization": QUANTIZATION,
            "max_sequence_length": MAX_LENGTH,
            "embedding_dimension": smoke_metadata["corpus_embedding_shape"][1],
            "embedding_dimension_matches_expected": (
                smoke_metadata["corpus_embedding_shape"][1] == EXPECTED_EMBEDDING_DIMENSION
                and smoke_metadata["query_embedding_shape"][1] == EXPECTED_EMBEDDING_DIMENSION
            ),
            "finite_embeddings": bool(smoke_metadata["finite_embeddings"]),
            "finite_scores": bool(np.isfinite(np.asarray(smoke_scores, dtype=np.float32)).all()),
            "retrieval_works": len(smoke_rankings) == 20
            and all(len(ranking) == min(TOP_K, 100) for ranking in smoke_rankings.values()),
            "smoke_corpus_size": 100,
            "smoke_query_count": 20,
            "runtime_seconds": smoke_runtime,
            "corpus_encoding_runtime_seconds": smoke_metadata["corpus_encoding_runtime_seconds"],
            "query_encoding_runtime_seconds": smoke_metadata["query_encoding_runtime_seconds"],
            "retrieval_runtime_seconds": smoke_metadata["retrieval_runtime_seconds"],
            "projected_full_encoding_runtime_seconds": (
                smoke_metadata["corpus_encoding_runtime_seconds"] / 100 * len(corpus)
                + smoke_metadata["query_encoding_runtime_seconds"] / 20 * len(queries)
            ),
        }
        smoke["projected_full_encoding_runtime_hours"] = smoke[
            "projected_full_encoding_runtime_seconds"
        ] / 3600
        write_json(results_dir / "smoke_test.json", smoke)
        print("Smoke test:")
        print(json.dumps(smoke, indent=2, sort_keys=True))

    if args.smoke_only:
        print("Smoke-only run complete.")
        return 0

    full_start = time.perf_counter()
    rankings, retrieval_metadata = run_embedding_and_retrieval(
        corpus=corpus,
        queries=queries,
        tokenizer=tokenizer,
        cache_dir=cache_dir / "full",
        server_url=args.server_url,
        corpus_batch_size=args.corpus_batch_size,
        query_batch_size=args.query_batch_size,
        similarity_batch_size=args.similarity_batch_size,
        corpus_fingerprint=corpus_fingerprint,
        corpus_desc_prefix="full 1.5B full1024",
        query_desc_prefix="full 1.5B full1024",
    )
    embedding_retrieval_runtime = time.perf_counter() - full_start

    eval_start = time.perf_counter()
    report = evaluate_rankings(rankings, qrels, k_values=K_VALUES)
    positive_ranks_100 = positive_ranks_at_k(rankings, qrels, k=100)
    buckets = rank_buckets(positive_ranks_100)
    evaluation_runtime = time.perf_counter() - eval_start

    metadata = {
        "run_name": RUN_NAME,
        "task_name": TASK_NAME,
        "dataset": task.metadata.dataset["path"],
        "dataset_revision": task.metadata.dataset["revision"],
        "subset": SUBSET,
        "split": SPLIT,
        "top_k": TOP_K,
        "corpus_size": len(corpus),
        "query_count": len(queries),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_seconds": embedding_retrieval_runtime + evaluation_runtime,
        "retrieval_evaluation_runtime_seconds": (
            retrieval_metadata["retrieval_runtime_seconds"] + evaluation_runtime
        ),
        "evaluation_runtime_seconds": evaluation_runtime,
        "method": "jina_code_1.5b_gguf_full_dense_cosine_maxseq1024",
        "model": MODEL_NAME,
        "model_name": MODEL_NAME,
        "gguf_repo": GGUF_REPO,
        "gguf_file": GGUF_FILE,
        "quantization": QUANTIZATION,
        "embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
        "expected_embedding_dimension": EXPECTED_EMBEDDING_DIMENSION,
        "max_sequence_length": MAX_LENGTH,
        "query_instruction": QUERY_INSTRUCTION,
        "passage_instruction": DOCUMENT_INSTRUCTION,
        "document_instruction": DOCUMENT_INSTRUCTION,
        "pooling": POOLING_STRATEGY,
        "pooling_strategy": POOLING_STRATEGY,
        "normalization": NORMALIZE_EMBEDDINGS,
        "similarity": "cosine/dot-product on normalized embeddings",
        "corpus_fingerprint": corpus_fingerprint,
        "server_url": args.server_url,
        "model_file": str(model_file),
        "cache_dir": str(cache_dir),
        "results_dir": str(results_dir),
        "package_versions": package_versions(),
        **retrieval_metadata,
    }
    payload = metrics_payload(report, metadata=metadata)
    payload["rank_buckets"] = buckets
    write_json(results_dir / "metrics.json", payload)
    write_json(results_dir / "top100.json", {"metadata": metadata, "rankings": rankings})

    print("Final Jina 1.5B full 1024 metrics:")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Saved metrics: {results_dir / 'metrics.json'}")
    print(f"Saved top100: {results_dir / 'top100.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
