"""Selective high-context Jina GGUF reranking for AppsRetrieval.

This experiment keeps the verified Jina Q8 first-stage retrieval fixed and
reranks only the top 5 candidates for the lowest-margin 25% of queries using
the same model at max sequence length 1024.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
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

from src.evaluation.metrics import DEFAULT_K_VALUES, evaluate_rankings, metrics_payload, positive_ranks_at_k
from src.retrieval.jina_code import DOCUMENT_INSTRUCTION, MODEL_NAME, QUERY_INSTRUCTION, row_text
from src.retrieval.jina_gguf import GGUF_FILE, GGUF_REPO, embed_http_cached, model_slug, prompt_hash, truncate_with_tokenizer
from src.utils.env import env_path, llama_server_url, load_dotenv


TASK_NAME = "AppsRetrieval"
SUBSET = "default"
SPLIT = "test"
BASELINE_RUN = "jina_code_0.5b_q8"
OUTPUT_RUN = "jina_code_0.5b_q8_selective_1024"
BASELINE_TOP100 = PROJECT_ROOT / "results" / BASELINE_RUN / "top100.json"
OUTPUT_DIR = PROJECT_ROOT / "results" / OUTPUT_RUN
TOP_K = 100
RERANK_DEPTH = 5
AMBIGUOUS_FRACTION = 0.25
MAX_LENGTH = 1024


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
        "transformers": version("transformers"),
        "torch": version("torch"),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def dataset_to_rows(dataset: datasets.Dataset) -> list[dict[str, Any]]:
    return [dict(row) for row in dataset]


def qrels_to_dict(qrels: dict[str, dict[str, int]]) -> dict[str, dict[str, int]]:
    return {
        str(query_id): {str(doc_id): int(label) for doc_id, label in docs.items()}
        for query_id, docs in qrels.items()
    }


def ranked_items(scores: dict[str, float], *, limit: int | None = None) -> list[tuple[str, float]]:
    ordered = sorted(scores.items(), key=lambda item: (float(item[1]), item[0]), reverse=True)
    return ordered[:limit] if limit is not None else ordered


def load_baseline_rankings(path: Path) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rankings = {
        str(query_id): {str(doc_id): float(score) for doc_id, score in doc_scores.items()}
        for query_id, doc_scores in payload["rankings"].items()
    }
    return rankings, dict(payload.get("metadata", {}))


def select_ambiguous_queries(
    rankings: dict[str, dict[str, float]],
    *,
    fraction: float,
    limit: int | None = None,
) -> tuple[list[str], list[dict[str, float | str]], float]:
    margins: list[tuple[float, str]] = []
    for query_id, scores in rankings.items():
        top2 = ranked_items(scores, limit=2)
        if len(top2) < 2:
            raise ValueError(f"{query_id} has fewer than two retrieved documents")
        margin = float(top2[0][1]) - float(top2[1][1])
        margins.append((margin, query_id))

    margins.sort(key=lambda item: (item[0], item[1]))
    selected_count = math.ceil(len(margins) * fraction)
    if limit is not None:
        selected_count = min(selected_count, limit)
    selected = margins[:selected_count]
    details = [{"query_id": query_id, "rank1_rank2_margin": margin} for margin, query_id in selected]
    threshold = selected[-1][0] if selected else float("nan")
    return [query_id for margin, query_id in selected], details, threshold


def check_server(server_url: str) -> None:
    response = requests.get(server_url.rstrip("/") + "/health", timeout=10)
    response.raise_for_status()


def prepare_prompts(
    *,
    ids: list[str],
    text_by_id: dict[str, str],
    instruction: str,
    tokenizer: Any,
    max_length: int,
    desc: str,
) -> list[str]:
    return [
        truncate_with_tokenizer(
            tokenizer=tokenizer,
            text=text_by_id[item_id],
            instruction=instruction,
            max_length=max_length,
        )
        for item_id in tqdm(ids, desc=desc)
    ]


def unique_top_docs(
    *,
    rankings: dict[str, dict[str, float]],
    selected_query_ids: list[str],
    depth: int,
) -> tuple[list[str], dict[str, list[str]]]:
    seen: set[str] = set()
    doc_ids: list[str] = []
    top_docs_by_query: dict[str, list[str]] = {}
    for query_id in selected_query_ids:
        top_docs = [doc_id for doc_id, _ in ranked_items(rankings[query_id], limit=depth)]
        top_docs_by_query[query_id] = top_docs
        for doc_id in top_docs:
            if doc_id not in seen:
                seen.add(doc_id)
                doc_ids.append(doc_id)
    return doc_ids, top_docs_by_query


def synthetic_order_scores(count: int, *, start: float) -> list[float]:
    return [start - float(rank) for rank in range(count)]


def ranking_from_doc_order(doc_ids: list[str]) -> dict[str, float]:
    """Create deterministic evaluator-safe scores preserving an explicit order."""
    return {
        doc_id: score
        for doc_id, score in zip(doc_ids, synthetic_order_scores(len(doc_ids), start=1000.0), strict=True)
    }


def rerank_selected_queries(
    *,
    baseline_rankings: dict[str, dict[str, float]],
    selected_query_ids: list[str],
    top_docs_by_query: dict[str, list[str]],
    query_embeddings: np.ndarray,
    query_ids: list[str],
    doc_embeddings: np.ndarray,
    doc_ids: list[str],
) -> dict[str, dict[str, float]]:
    query_index = {query_id: index for index, query_id in enumerate(query_ids)}
    doc_index = {doc_id: index for index, doc_id in enumerate(doc_ids)}
    selected = set(selected_query_ids)
    reranked: dict[str, dict[str, float]] = {}

    for query_id, scores in baseline_rankings.items():
        original_order = ranked_items(scores, limit=TOP_K)
        if query_id not in selected:
            reranked[query_id] = ranking_from_doc_order([doc_id for doc_id, _ in original_order])
            continue

        top_docs = top_docs_by_query[query_id]
        query_vector = query_embeddings[query_index[query_id]]
        matrix = np.vstack([doc_embeddings[doc_index[doc_id]] for doc_id in top_docs])
        new_scores = matrix @ query_vector
        if not np.isfinite(new_scores).all():
            raise ValueError(f"Non-finite selective scores for {query_id}")

        top5_ordered = sorted(
            zip(top_docs, new_scores.tolist(), strict=True),
            key=lambda item: (float(item[1]), item[0]),
            reverse=True,
        )
        tail = [(doc_id, score) for doc_id, score in original_order if doc_id not in top_docs]
        final_doc_ids = [doc_id for doc_id, _ in top5_ordered] + [doc_id for doc_id, _ in tail]
        final_doc_ids = final_doc_ids[:TOP_K]

        # The evaluator consumes score-sorted rankings.  Use deterministic
        # ordinal scores for every query so the persisted file represents only
        # rank order; selected queries change only in the top 5 and ranks 6-100
        # remain in the original first-stage order.
        reranked[query_id] = ranking_from_doc_order(final_doc_ids)

    return reranked


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
    baseline_ranks: dict[str, int | None],
    reranked_ranks: dict[str, int | None],
) -> dict[str, int]:
    rank1_moved_down = sum(
        baseline_ranks[query_id] == 1 and reranked_ranks[query_id] != 1
        for query_id in baseline_ranks
    )
    rank2_5_to_rank1 = sum(
        baseline_ranks[query_id] is not None
        and 2 <= int(baseline_ranks[query_id]) <= 5
        and reranked_ranks[query_id] == 1
        for query_id in baseline_ranks
    )
    baseline_rank1 = sum(rank == 1 for rank in baseline_ranks.values())
    reranked_rank1 = sum(rank == 1 for rank in reranked_ranks.values())
    return {
        "rank_1_positives_moved_down": int(rank1_moved_down),
        "rank_2_5_positives_moved_to_rank_1": int(rank2_5_to_rank1),
        "baseline_rank_1_count": int(baseline_rank1),
        "reranked_rank_1_count": int(reranked_rank1),
        "net_change_in_rank_1_count": int(reranked_rank1 - baseline_rank1),
    }


def run_selective_pass(
    *,
    selected_query_ids: list[str],
    baseline_rankings: dict[str, dict[str, float]],
    query_text_by_id: dict[str, str],
    doc_text_by_id: dict[str, str],
    tokenizer: Any,
    server_url: str,
    cache_dir: Path,
    batch_size: int,
    cache_prefix: str,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    unique_doc_ids, top_docs_by_query = unique_top_docs(
        rankings=baseline_rankings,
        selected_query_ids=selected_query_ids,
        depth=RERANK_DEPTH,
    )
    query_prompts = prepare_prompts(
        ids=selected_query_ids,
        text_by_id=query_text_by_id,
        instruction=QUERY_INSTRUCTION,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        desc=f"{cache_prefix} prepare query prompts",
    )
    doc_prompts = prepare_prompts(
        ids=unique_doc_ids,
        text_by_id=doc_text_by_id,
        instruction=DOCUMENT_INSTRUCTION,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        desc=f"{cache_prefix} prepare doc prompts",
    )

    query_cache_stem = (
        f"{model_slug(GGUF_FILE)}_{cache_prefix}_query_nl2code_last_maxseq-{MAX_LENGTH}_"
        f"{prompt_hash(selected_query_ids, query_prompts)}"
    )
    doc_cache_stem = (
        f"{model_slug(GGUF_FILE)}_{cache_prefix}_doc_passage_last_maxseq-{MAX_LENGTH}_"
        f"{prompt_hash(unique_doc_ids, doc_prompts)}"
    )
    query_embeddings, query_cache_hit, query_cache_path = embed_http_cached(
        ids=selected_query_ids,
        texts=query_prompts,
        server_url=server_url,
        batch_size=batch_size,
        desc=f"{cache_prefix} encode queries",
        cache_dir=cache_dir,
        cache_stem=query_cache_stem,
    )
    doc_embeddings, doc_cache_hit, doc_cache_path = embed_http_cached(
        ids=unique_doc_ids,
        texts=doc_prompts,
        server_url=server_url,
        batch_size=batch_size,
        desc=f"{cache_prefix} encode docs",
        cache_dir=cache_dir,
        cache_stem=doc_cache_stem,
    )
    if not np.isfinite(query_embeddings).all() or not np.isfinite(doc_embeddings).all():
        raise ValueError("Non-finite embeddings returned by llama.cpp server")

    reranked = rerank_selected_queries(
        baseline_rankings=baseline_rankings,
        selected_query_ids=selected_query_ids,
        top_docs_by_query=top_docs_by_query,
        query_embeddings=query_embeddings,
        query_ids=selected_query_ids,
        doc_embeddings=doc_embeddings,
        doc_ids=unique_doc_ids,
    )
    metadata = {
        "selected_query_count": len(selected_query_ids),
        "top5_candidate_slots": len(selected_query_ids) * RERANK_DEPTH,
        "unique_candidate_documents_reencoded": len(unique_doc_ids),
        "query_embedding_cache_hit": query_cache_hit,
        "query_embedding_cache_path": str(query_cache_path),
        "doc_embedding_cache_hit": doc_cache_hit,
        "doc_embedding_cache_path": str(doc_cache_path),
        "query_embedding_shape": list(query_embeddings.shape),
        "doc_embedding_shape": list(doc_embeddings.shape),
    }
    return reranked, metadata


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    load_dotenv(PROJECT_ROOT / ".env")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    datasets.disable_progress_bars()

    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default=llama_server_url())
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()

    cache_dir = env_path(
        "JCR_GGUF_SELECTIVE_1024_CACHE_DIR",
        PROJECT_ROOT / "data" / "cache" / OUTPUT_RUN,
        project_root=PROJECT_ROOT,
    )

    print("Checking llama.cpp embedding server...")
    check_server(args.server_url)
    print("Server ok")
    print(f"Baseline top100: {BASELINE_TOP100}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Model: {MODEL_NAME} via {GGUF_REPO}/{GGUF_FILE}")
    print(f"Max sequence length for selective pass: {MAX_LENGTH}")
    print(f"Rerank rule: lowest {AMBIGUOUS_FRACTION:.0%} rank1-rank2 first-stage score margins")
    print(f"Rerank depth: top {RERANK_DEPTH}; ranks 6-100 preserved")
    print("Package versions:")
    print(json.dumps(package_versions(), indent=2, sort_keys=True))

    baseline_rankings, baseline_metadata = load_baseline_rankings(BASELINE_TOP100)
    selected_query_ids, ambiguity_details, ambiguity_threshold = select_ambiguous_queries(
        baseline_rankings,
        fraction=AMBIGUOUS_FRACTION,
    )
    print(f"Loaded baseline rankings for {len(baseline_rankings)} queries")
    print(f"Selected ambiguous queries: {len(selected_query_ids)}")
    print(f"Ambiguity threshold margin: {ambiguity_threshold:.8f}")

    print("Loading official AppsRetrieval data...")
    task = mteb.get_task(TASK_NAME)
    task.load_data()
    split = task.dataset[SUBSET][SPLIT]
    corpus = dataset_to_rows(split["corpus"])
    queries = dataset_to_rows(split["queries"])
    query_text_by_id = {row["id"]: row_text(row) for row in queries}
    doc_text_by_id = {row["id"]: row_text(row) for row in corpus}
    print(f"Loaded corpus={len(corpus)} queries={len(queries)}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if not args.skip_smoke:
        smoke_query_ids = selected_query_ids[:20]
        smoke_start = time.perf_counter()
        smoke_rankings, smoke_metadata = run_selective_pass(
            selected_query_ids=smoke_query_ids,
            baseline_rankings=baseline_rankings,
            query_text_by_id=query_text_by_id,
            doc_text_by_id=doc_text_by_id,
            tokenizer=tokenizer,
            server_url=args.server_url,
            cache_dir=cache_dir / "smoke",
            batch_size=args.batch_size,
            cache_prefix="smoke20",
        )
        smoke_runtime = time.perf_counter() - smoke_start
        smoke_payload = {
            "model_loaded": True,
            "finite_scores": all(
                np.isfinite(list(smoke_rankings[query_id].values())).all()
                for query_id in smoke_query_ids
            ),
            "smoke_query_count": len(smoke_query_ids),
            "rerank_depth": RERANK_DEPTH,
            "runtime_seconds": smoke_runtime,
            "seconds_per_query": smoke_runtime / len(smoke_query_ids),
            "projected_full_runtime_seconds": smoke_runtime / len(smoke_query_ids) * len(selected_query_ids),
            "metadata": smoke_metadata,
        }
        write_json(OUTPUT_DIR / "smoke_test.json", smoke_payload)
        print("Smoke test:")
        print(json.dumps(smoke_payload, indent=2, sort_keys=True))
        if args.smoke_only:
            return 0

    full_start = time.perf_counter()
    reranked, rerank_metadata = run_selective_pass(
        selected_query_ids=selected_query_ids,
        baseline_rankings=baseline_rankings,
        query_text_by_id=query_text_by_id,
        doc_text_by_id=doc_text_by_id,
        tokenizer=tokenizer,
        server_url=args.server_url,
        cache_dir=cache_dir / "full",
        batch_size=args.batch_size,
        cache_prefix="selective1024",
    )
    rerank_runtime = time.perf_counter() - full_start

    # Qrels are intentionally loaded only after final rankings exist.
    qrels = qrels_to_dict(split["relevant_docs"])
    report = evaluate_rankings(reranked, qrels, k_values=DEFAULT_K_VALUES)
    baseline_ranks_100 = positive_ranks_at_k(baseline_rankings, qrels, k=100)
    reranked_ranks_100 = positive_ranks_at_k(reranked, qrels, k=100)
    baseline_buckets = rank_buckets(baseline_ranks_100)
    reranked_buckets = rank_buckets(reranked_ranks_100)
    movements = movement_counts(baseline_ranks_100, reranked_ranks_100)

    metadata = {
        "run_name": OUTPUT_RUN,
        "task_name": TASK_NAME,
        "dataset": task.metadata.dataset["path"],
        "dataset_revision": task.metadata.dataset["revision"],
        "subset": SUBSET,
        "split": SPLIT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "selective_jina_code_gguf_dense_cosine_rerank_top5_maxseq1024",
        "baseline_run": BASELINE_RUN,
        "baseline_top100_path": str(BASELINE_TOP100),
        "baseline_immutable": True,
        "model": MODEL_NAME,
        "gguf_repo": GGUF_REPO,
        "gguf_file": GGUF_FILE,
        "max_sequence_length": MAX_LENGTH,
        "first_stage_max_sequence_length": baseline_metadata.get("max_sequence_length"),
        "query_instruction": QUERY_INSTRUCTION,
        "document_instruction": DOCUMENT_INSTRUCTION,
        "pooling_strategy": "last_token",
        "normalization": True,
        "similarity": "cosine/dot-product on normalized embeddings",
        "ambiguity_rule": "lowest rank1-rank2 first-stage score margins",
        "ambiguity_fraction": AMBIGUOUS_FRACTION,
        "ambiguity_threshold_margin": ambiguity_threshold,
        "rerank_depth": RERANK_DEPTH,
        "top_k": TOP_K,
        "corpus_size": len(corpus),
        "query_count": len(queries),
        "selected_query_count": len(selected_query_ids),
        "runtime_seconds": rerank_runtime,
        "server_url": args.server_url,
        "batch_size": args.batch_size,
        "package_versions": package_versions(),
        **rerank_metadata,
        **movements,
    }
    metrics = metrics_payload(report, metadata=metadata)
    metrics["rank_buckets_before"] = baseline_buckets
    metrics["rank_buckets_after"] = reranked_buckets
    metrics["movement_counts"] = movements
    metrics["ambiguity_details_first_20"] = ambiguity_details[:20]

    top100_payload = {
        "metadata": metadata,
        "rankings": reranked,
    }
    write_json(OUTPUT_DIR / "metrics.json", metrics)
    write_json(OUTPUT_DIR / "top100.json", top100_payload)

    print("Final selective Jina 1024 metrics:")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved metrics: {OUTPUT_DIR / 'metrics.json'}")
    print(f"Saved top100: {OUTPUT_DIR / 'top100.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
