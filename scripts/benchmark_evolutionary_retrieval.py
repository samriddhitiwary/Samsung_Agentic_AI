from __future__ import annotations

import argparse
import csv
import json
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
from src.utils.env import llama_server_url, load_dotenv
from src.versioning.embedding_cache import EmbeddingConfig
from src.versioning.evolution_search import load_evolution_metadata, search_across_versions
from src.versioning.incremental_embedder import _embed_http, check_server


DEFAULT_ARTIFACT_ROOT = PROJECT_ROOT / "data/versioning/real_benchmark_artifacts/itsdangerous"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data/versioning/benchmarks/evolutionary"
DEFAULT_TOP_K = 5


BENCHMARK_QUERIES: list[dict[str, Any]] = [
    {
        "id": "hmac_lazy_sha1_fips",
        "query": "HMAC algorithm using lazy SHA1 for FIPS builds",
        "expected_content_ids": [
            "b523c46cab791ce6b7fc9e4a5674502811a12644bc9e51ea5f86572e1789c268",
        ],
        "expected_symbol": "HMACAlgorithm",
        "expected_occurrence_key": "src/itsdangerous/signer.py:class:HMACAlgorithm",
        "manual_expected_basis": "Verified from the real itsdangerous evolution chain: this state changes HMACAlgorithm.default_digest_method to staticmethod(_lazy_sha1).",
    },
    {
        "id": "hmac_direct_hashlib_sha1",
        "query": "HMAC algorithm default digest method uses hashlib sha1 directly",
        "expected_content_ids": [
            "3720635ce985f50bc6b252afe7df7dacdd554b6615330439e2ffd94a4fa417ac",
            "bcdcada98c235a39485bbc2eff37cf4dfb499c97664403a83af513f62816187b",
        ],
        "expected_symbol": "HMACAlgorithm",
        "expected_occurrence_key": "src/itsdangerous/signer.py:class:HMACAlgorithm",
        "manual_expected_basis": "Verified from the real itsdangerous evolution chain: both earlier HMACAlgorithm states use staticmethod(hashlib.sha1).",
    },
    {
        "id": "serializer_init_before_generics",
        "query": "serializer constructor before generic typing support",
        "expected_content_ids": [
            "443ba8100225c955d0ecf12a98bd863d4edd7bec6171270426eb2b371c6a1868",
            "2b670c87f5866c203c7983814536edf724048f2a90800b65ce116935d151cefa",
        ],
        "expected_symbol": "Serializer.__init__",
        "expected_occurrence_key": "src/itsdangerous/serializer.py:method:Serializer.__init__",
        "manual_expected_basis": "Verified from the real serializer constructor chain: these states precede the Serializer[_TAnyStr] and _TSerialized generic states.",
    },
    {
        "id": "serializer_init_tserialized_overloads",
        "query": "serializer constructor using generic TSerialized overloads",
        "expected_content_ids": [
            "149e9c11a1b3d928b852cb3e72286095ecaafe53f10f4185bc29a9306af96145",
            "e1f7a38ceb3625c8a1a57600fc4f87450994e602760f712e1f3d7254a066ff57",
        ],
        "expected_symbol": "Serializer.__init__",
        "expected_occurrence_key": "src/itsdangerous/serializer.py:method:Serializer.__init__",
        "manual_expected_basis": "Verified from the real serializer constructor chain: later states introduce _TSerialized and overload-style byte serializer typing.",
    },
    {
        "id": "serializer_load_payload_generic_protocol",
        "query": "serializer load payload with generic data serializer protocol",
        "expected_content_ids": [
            "22fabd5907eb78d681f5db94fb083377adb8a52a81d9711666e0b4ddae286d0f",
            "350e106641cf3045f3d5e7582965965255ccfa3b7475a6af29cab33cf3da3163",
        ],
        "expected_symbol": "Serializer.load_payload",
        "expected_occurrence_key": "src/itsdangerous/serializer.py:method:Serializer.load_payload",
        "manual_expected_basis": "Verified from the real load_payload chain: these states use _PDataSerializer generic protocol typing.",
    },
    {
        "id": "timed_serializer_pyright_ignore",
        "query": "timed serializer pyright ignore type checking",
        "expected_content_ids": [
            "31ab31d30d12016bbcb875f69bc6994f2808a648320f83714f4bb431566a90d0",
        ],
        "expected_symbol": "TimedSerializer",
        "expected_occurrence_key": "src/itsdangerous/timed.py:class:TimedSerializer",
        "manual_expected_basis": "Verified from the real TimedSerializer chain: final state contains the pyright ignore typing update.",
    },
]


def parse_args() -> argparse.Namespace:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Internal benchmark for version-aware evolutionary retrieval.")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--server-url", default=llama_server_url())
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--batch-size", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    index_dir = args.artifact_root / "index_incremental"
    evolution_dir = args.artifact_root / "evolution"
    output_json = args.output_dir / "evolutionary_retrieval_benchmark.json"
    output_csv = args.output_dir / "evolutionary_retrieval_benchmark.csv"

    started = time.perf_counter()
    metadata = load_evolution_metadata(evolution_dir)
    validate_expected_states(metadata)
    config = EmbeddingConfig(prompt_instruction=QUERY_INSTRUCTION, prompt_version="jina_code_1.5b_query_v1")

    check_server(args.server_url)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    prompts = [
        truncate_with_tokenizer(
            tokenizer=tokenizer,
            text=item["query"],
            instruction=QUERY_INSTRUCTION,
            max_length=config.max_sequence_length,
        )
        for item in BENCHMARK_QUERIES
    ]
    embed_started = time.perf_counter()
    embeddings = _embed_http(
        texts=prompts,
        server_url=args.server_url,
        batch_size=args.batch_size,
        normalize=True,
        desc="embed evolutionary benchmark queries",
    )
    embedding_seconds = time.perf_counter() - embed_started
    if embeddings.shape != (len(BENCHMARK_QUERIES), config.embedding_dimension):
        raise ValueError(f"Unexpected query embedding shape: {embeddings.shape}")
    if not np.isfinite(embeddings).all():
        raise ValueError("Non-finite query embeddings")

    per_query: list[dict[str, Any]] = []
    raw_eval: list[dict[str, Any]] = []
    grouped_eval: list[dict[str, Any]] = []
    retrieval_started = time.perf_counter()
    for index, query_case in enumerate(BENCHMARK_QUERIES):
        vector = embeddings[index].astype(np.float32)
        raw_payload = search_across_versions(
            index_dir=index_dir,
            evolution_dir=evolution_dir,
            query_embedding=vector,
            top_k=args.top_k,
            raw=True,
        )
        grouped_payload = search_across_versions(
            index_dir=index_dir,
            evolution_dir=evolution_dir,
            query_embedding=vector,
            top_k=args.top_k,
            raw=False,
            include_evolution_context=True,
        )
        raw_summary = summarize_results(raw_payload["results"], query_case["expected_content_ids"])
        grouped_summary = summarize_results(grouped_payload["results"], query_case["expected_content_ids"])
        raw_eval.append(raw_summary)
        grouped_eval.append(grouped_summary)
        per_query.append(
            {
                **query_case,
                "raw_occurrence_search": {
                    **raw_summary,
                    "results": compact_results(raw_payload["results"]),
                    "runtime": raw_payload["runtime"],
                },
                "grouped_evolution_state_search": {
                    **grouped_summary,
                    "results": compact_results(grouped_payload["results"], include_context=True),
                    "runtime": grouped_payload["runtime"],
                },
            }
        )
    retrieval_seconds = time.perf_counter() - retrieval_started

    report = {
        "benchmark": "internal_itsdangerous_evolutionary_retrieval_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "purpose": "Validate whether grouped version-aware state retrieval reduces duplicate version occurrences on difficult similar-version queries.",
        "official_benchmark": False,
        "repo": metadata["repo"],
        "commit_order": metadata["commit_graph"]["order"],
        "artifact_root": args.artifact_root.as_posix(),
        "index_dir": index_dir.as_posix(),
        "evolution_dir": evolution_dir.as_posix(),
        "model_config": {
            "model_name": config.model_name,
            "gguf_file": config.gguf_file,
            "quantization": config.quantization,
            "embedding_dimension": config.embedding_dimension,
            "max_sequence_length": config.max_sequence_length,
            "query_instruction": QUERY_INSTRUCTION,
            "pooling": config.pooling,
            "normalization": config.normalization,
        },
        "top_k": args.top_k,
        "query_count": len(BENCHMARK_QUERIES),
        "manually_verified_expected_states": True,
        "metrics": {
            "raw_occurrence_search": aggregate(raw_eval, args.top_k),
            "grouped_evolution_state_search": aggregate(grouped_eval, args.top_k),
        },
        "duplicate_summary": {
            "raw_duplicate_results_returned": sum(item["duplicate_results"] for item in raw_eval),
            "grouped_duplicate_results_returned": sum(item["duplicate_results"] for item in grouped_eval),
            "raw_unique_semantic_states_returned": sum(item["unique_content_ids"] for item in raw_eval),
            "grouped_unique_semantic_states_returned": sum(item["unique_content_ids"] for item in grouped_eval),
        },
        "per_query": per_query,
        "runtime": {
            "embedding_seconds": embedding_seconds,
            "retrieval_seconds": retrieval_seconds,
            "total_seconds": time.perf_counter() - started,
        },
        "outputs": {
            "json": output_json.as_posix(),
            "csv": output_csv.as_posix(),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(output_csv, per_query)
    print_summary(report)
    return 0


def validate_expected_states(metadata: dict[str, Any]) -> None:
    states = metadata["content_states"]
    missing = [
        content_id
        for query_case in BENCHMARK_QUERIES
        for content_id in query_case["expected_content_ids"]
        if content_id not in states
    ]
    if missing:
        raise KeyError(f"Benchmark expected unknown content IDs: {missing}")


def summarize_results(results: list[dict[str, Any]], expected_content_ids: list[str]) -> dict[str, Any]:
    expected = set(expected_content_ids)
    ids = [item["content_id"] for item in results]
    first_rank = next((rank for rank, content_id in enumerate(ids, start=1) if content_id in expected), None)
    return {
        "first_relevant_rank": first_rank,
        "hit_at_1": bool(first_rank == 1),
        "hit_at_3": bool(first_rank is not None and first_rank <= 3),
        "hit_at_5": bool(first_rank is not None and first_rank <= 5),
        "reciprocal_rank": 0.0 if first_rank is None else 1.0 / first_rank,
        "result_count": len(results),
        "unique_content_ids": len(set(ids)),
        "duplicate_results": len(ids) - len(set(ids)),
    }


def aggregate(items: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    query_count = len(items)
    return {
        "query_count": query_count,
        "top_k": top_k,
        "HitRate@1": sum(item["hit_at_1"] for item in items) / query_count,
        "HitRate@3": sum(item["hit_at_3"] for item in items) / query_count,
        "HitRate@5": sum(item["hit_at_5"] for item in items) / query_count,
        "MRR": sum(item["reciprocal_rank"] for item in items) / query_count,
        "duplicate_results_returned": sum(item["duplicate_results"] for item in items),
        "unique_semantic_states_returned": sum(item["unique_content_ids"] for item in items),
    }


def compact_results(results: list[dict[str, Any]], include_context: bool = False) -> list[dict[str, Any]]:
    compact = []
    for rank, item in enumerate(results, start=1):
        value = {
            "rank": rank,
            "score": float(item["score"]),
            "content_id": item["content_id"],
            "path": item.get("path") or item.get("best_match", {}).get("path"),
            "symbol": item.get("symbol") or item.get("best_match", {}).get("symbol"),
            "chunk_type": item.get("chunk_type") or item.get("best_match", {}).get("chunk_type"),
            "commit": item.get("commit") or item.get("best_match", {}).get("commit"),
            "occurrence_count": len(item.get("occurrences", [])),
            "preview": item.get("preview"),
        }
        if include_context:
            value["evolution_context"] = item.get("evolution_context")
        compact.append(value)
    return compact


def write_csv(path: Path, per_query: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "query_id",
                "query",
                "raw_first_relevant_rank",
                "grouped_first_relevant_rank",
                "raw_duplicate_results",
                "grouped_duplicate_results",
                "raw_top_content_id",
                "grouped_top_content_id",
            ],
        )
        writer.writeheader()
        for item in per_query:
            raw = item["raw_occurrence_search"]
            grouped = item["grouped_evolution_state_search"]
            writer.writerow(
                {
                    "query_id": item["id"],
                    "query": item["query"],
                    "raw_first_relevant_rank": raw["first_relevant_rank"],
                    "grouped_first_relevant_rank": grouped["first_relevant_rank"],
                    "raw_duplicate_results": raw["duplicate_results"],
                    "grouped_duplicate_results": grouped["duplicate_results"],
                    "raw_top_content_id": raw["results"][0]["content_id"] if raw["results"] else None,
                    "grouped_top_content_id": grouped["results"][0]["content_id"] if grouped["results"] else None,
                }
            )


def print_summary(report: dict[str, Any]) -> None:
    raw = report["metrics"]["raw_occurrence_search"]
    grouped = report["metrics"]["grouped_evolution_state_search"]
    print("Evolutionary retrieval internal benchmark")
    print(f"Repo: {report['repo']}")
    print(f"Queries: {report['query_count']}, top_k={report['top_k']}")
    print(f"Raw occurrence search: HitRate@1={raw['HitRate@1']:.4f}, HitRate@3={raw['HitRate@3']:.4f}, MRR={raw['MRR']:.4f}, duplicates={raw['duplicate_results_returned']}")
    print(f"Grouped state search:  HitRate@1={grouped['HitRate@1']:.4f}, HitRate@3={grouped['HitRate@3']:.4f}, MRR={grouped['MRR']:.4f}, duplicates={grouped['duplicate_results_returned']}")
    print(f"Saved JSON: {report['outputs']['json']}")
    print(f"Saved CSV:  {report['outputs']['csv']}")
    print(f"Runtime seconds: {report['runtime']['total_seconds']:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
