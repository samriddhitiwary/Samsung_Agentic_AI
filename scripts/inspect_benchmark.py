"""Inspect the official MTEB AppsRetrieval task without running retrieval."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import sys
import warnings
from collections import Counter
from typing import Any

import datasets

warnings.filterwarnings(
    "ignore",
    message="`torch.jit.script` is deprecated.*",
    category=FutureWarning,
)
import mteb


TASK_NAME = "AppsRetrieval"
SUBSET = "default"
MAX_SAMPLE_CHARS = 500


def version(distribution: str) -> str:
    """Return an installed distribution version without importing it again."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def compact(value: Any, limit: int = MAX_SAMPLE_CHARS) -> str:
    """Render a bounded, single-line representation suitable for a console."""
    text = str(value).replace("\r", "\\r").replace("\n", "\\n")
    return text if len(text) <= limit else f"{text[:limit]}... <truncated>"


def print_record(kind: str, row: dict[str, Any]) -> None:
    """Print one dataset row without dumping large fields."""
    print(f"  {kind} id: {row.get('id')}")
    for field in ("title", "text", "language", "partition", "meta_information"):
        value = row.get(field)
        if value not in (None, "", [], {}):
            print(f"    {field}: {compact(value)}")


def main() -> int:
    # Windows PowerShell commonly starts Python with a legacy console encoding.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")

    datasets.disable_progress_bars()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    print("=== Environment ===")
    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"OS: {platform.platform()}")
    print(f"MTEB: {version('mteb')}")
    print(f"datasets: {version('datasets')}")
    print(f"NumPy: {version('numpy')}")
    print(f"tqdm: {version('tqdm')}")

    try:
        import torch
    except ImportError:
        print("PyTorch: not installed")
        print("CUDA available: unavailable (PyTorch is not installed)")
    else:
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")

    print("\n=== Official task discovery ===")
    task = mteb.get_task(TASK_NAME)
    metadata = task.metadata
    print(f"Exact task name: {metadata.name}")
    print(f"Task class: {type(task).__module__}.{type(task).__name__}")
    print(f"Task type: {metadata.type}")
    print(f"Description: {metadata.description}")
    print(f"Dataset: {metadata.dataset['path']}")
    print(f"Pinned dataset revision: {metadata.dataset['revision']}")
    print(f"Available MTEB evaluation splits: {list(metadata.eval_splits)}")
    print(f"Languages/modalities: {list(metadata.eval_langs)}")
    print(f"Subsets: {list(task.hf_subsets)}")
    print(f"Main score: {metadata.main_score}")
    print(f"Reference: {metadata.reference}")

    print("\nLoading official dataset for inspection only (no retrieval is run)...")
    task.load_data()
    split_name = metadata.eval_splits[0]
    split = task.dataset[SUBSET][split_name]
    corpus = split["corpus"]
    queries = split["queries"]
    qrels = split["relevant_docs"]
    qrel_pairs = sum(len(documents) for documents in qrels.values())
    relevance_labels = sorted(
        {label for documents in qrels.values() for label in documents.values()}
    )

    print("\n=== Loaded benchmark data ===")
    print(f"Loaded subsets: {list(task.dataset)}")
    print(f"Loaded splits ({SUBSET}): {list(task.dataset[SUBSET])}")
    print(f"Evaluation split inspected: {split_name}")
    print(f"Corpus size: {len(corpus)}")
    print(f"Query count: {len(queries)}")
    print(f"Qrels query count: {len(qrels)}")
    print(f"Qrels query-document pair count: {qrel_pairs}")
    print(f"Relevance labels: {relevance_labels}")
    print(f"Corpus partition counts: {dict(Counter(corpus['partition']))}")
    print(f"Query partition counts: {dict(Counter(queries['partition']))}")
    print(f"Corpus columns: {corpus.column_names}")
    print(f"Query columns: {queries.column_names}")

    k_values = tuple(task.k_values)
    metric_families = (
        "NDCG",
        "MAP",
        "Recall",
        "Precision",
        "MRR",
        "NAUC variants",
        "Hit rate (success)",
    )
    print("\n=== Official evaluation configuration ===")
    print(f"Metric families exposed: {', '.join(metric_families)}")
    print(f"Cutoffs: {k_values}")
    print(f"NDCG@10 directly supported: {10 in k_values}")
    print(f"MRR@10 directly supported: {10 in k_values}")
    print(f"Default retrieval top_k requested by task: {task._top_k}")
    print("NDCG@10 implementation: pytrec_eval ndcg_cut_10, averaged over queries.")
    print(
        "MRR@10 implementation: first positively relevant document within ranks "
        "1..10 contributes 1/rank, otherwise 0; averaged over queries."
    )
    print(
        "AppsRetrieval has one binary-positive qrel per test query, so NDCG@10 "
        "is discounted gain at the positive document's rank (or 0 if absent)."
    )

    print("\n=== Evaluation result interface (MTEB SearchProtocol) ===")
    print("In-memory ranking type: dict[str, dict[str, float]]")
    print("Shape: {query_id: {document_id: score, ...}, ...}")
    print("Query IDs: exact strings from queries['id'] (for example, q5001).")
    print("Document IDs: exact strings from corpus['id'] (for example, d5001).")
    print("Scores: finite numeric relevance scores; higher scores rank first.")
    print("Expected coverage: every evaluation query; return up to top_k=1000 docs/query.")
    print("Only the first k ranks affect a metric at cutoff k.")
    print("Score ties are broken by document ID descending in MRR to match pytrec_eval.")
    print(
        "Optional saved prediction JSON envelope: "
        "{'mteb_model_meta': {...}, 'default': {'test': <ranking dict>}}"
    )

    sample_queries = [queries[index] for index in range(min(2, len(queries)))]
    print("\n=== Two sample queries and their official qrels ===")
    for row in sample_queries:
        print_record("query", row)
        print(f"    qrels: {json.dumps(qrels.get(row['id'], {}), sort_keys=True)}")

    corpus_index = {document_id: index for index, document_id in enumerate(corpus["id"])}
    sample_document_ids = [
        document_id
        for row in sample_queries
        for document_id in qrels.get(row["id"], {})
    ][:2]
    print("\n=== Their two relevant corpus/code documents ===")
    for document_id in sample_document_ids:
        print_record("document", corpus[corpus_index[document_id]])

    print("\nAppsRetrieval loaded successfully: yes")
    print("Retrieval/evaluation executed: no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
