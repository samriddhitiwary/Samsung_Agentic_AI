from __future__ import annotations

import argparse
import json
import math
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import median
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api.service import PROJECT_ROOT, ApiService  # noqa: E402


DEFAULT_OUTPUT = PROJECT_ROOT / "data/api/hands_on_latency_benchmark.json"
DEFAULT_QUERIES = [
    {"query": "How is a payload signed?", "type": "semantic", "expected": [{"path": "src/signer.js", "symbol": "createSignature"}]},
    {"query": "How is the signature digest created?", "type": "semantic", "expected": [{"path": "src/signer.js", "symbol": "sha1"}]},
    {"query": "How is a device persisted?", "type": "semantic", "expected": [{"path": "src/workflow.js", "symbol": "save"}]},
    {"query": "How does Bluetooth settings get opened?", "type": "semantic", "expected": [{"path": "src/settings.js", "symbol": "openBluetoothSettings"}]},
    {"query": "How is the payload serialized?", "type": "semantic", "expected": [{"path": "src/signer.js", "symbol": "serialize"}]},
    {"query": "Where is openBluetoothSettings used?", "type": "usage", "expected": [{"path": "src/settings.js", "symbol": "openBluetoothSettings"}, {"path": "src/workflow.js", "symbol": "setupDevice"}]},
    {"query": "Where is validate referenced?", "type": "usage", "expected": [{"path": "src/workflow.js", "symbol": "setupDevice"}]},
    {"query": "Where is save referenced?", "type": "usage", "expected": [{"path": "src/workflow.js", "symbol": "setupDevice"}]},
    {"query": "Where is createSignature used?", "type": "usage", "expected": [{"path": "src/signer.js", "symbol": "legacySign"}]},
    {"query": "Where is BluetoothSettings referenced?", "type": "usage", "expected": [{"path": "src/settings.js", "symbol": "BluetoothSettings"}]},
    {"query": "Which functions call validate before save?", "type": "structural", "expected": [{"path": "src/workflow.js", "symbol": "setupDevice"}]},
    {"query": "Which functions call serialize before sha1?", "type": "structural", "expected": [{"path": "src/signer.js", "symbol": "createSignature"}]},
    {"query": "Which functions call sha1 before serialize?", "type": "structural", "expected": [{"path": "src/signer.js", "symbol": "legacySign"}]},
    {"query": "Which functions call save before openBluetoothSettings?", "type": "structural", "expected": [{"path": "src/workflow.js", "symbol": "setupDevice"}]},
    {"query": "Which methods call openBluetoothSettings?", "type": "structural", "expected": [{"path": "src/settings.js", "symbol": "open"}]},
    {"query": "Where is SHA1 signer used before serialization?", "type": "mixed", "expected": [{"path": "src/signer.js", "symbol": "legacySign"}]},
    {"query": "Where does device validation happen before opening settings?", "type": "mixed", "expected": [{"path": "src/workflow.js", "symbol": "setupDevice"}]},
    {"query": "Where is Bluetooth settings deeplink used by workflow?", "type": "mixed", "expected": [{"path": "src/workflow.js", "symbol": "setupDevice"}]},
    {"query": "Where does serialization feed signature creation?", "type": "mixed", "expected": [{"path": "src/signer.js", "symbol": "createSignature"}]},
    {"query": "Where is the settings opener used after save?", "type": "mixed", "expected": [{"path": "src/workflow.js", "symbol": "setupDevice"}]},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark hands-on repository query latency and quality.")
    parser.add_argument("--repo-path", default=None, help="Local Git repo to register before benchmarking.")
    parser.add_argument("--repo-id", default="js_hands_on_smoke", help="Repository ID to use/register.")
    parser.add_argument("--commit", default="HEAD", help="Commit/ref to register/query.")
    parser.add_argument("--queries-file", default=None, help="JSON query set with manually verified expected answers.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output JSON report path.")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    service = ApiService()
    health = service.health()
    if not health["llama_cpp"]["server_healthy"]:
        raise SystemExit("Embedding server is offline; start llama.cpp first.")

    queries = load_queries(Path(args.queries_file)) if args.queries_file else DEFAULT_QUERIES
    limitation = None if args.queries_file else "Default query set is for the temporary JS smoke repo; use --queries-file for Samsung sample validation."

    indexing = None
    if args.repo_path:
        started = time.perf_counter()
        indexing = service.register_repo(repo_path=args.repo_path, repo_id=args.repo_id, commit=args.commit)
        indexing["benchmark_wrapper_seconds"] = time.perf_counter() - started

    record = service._repo_record(args.repo_id)
    commit = service._resolve_commit(Path(record["repo_path"]), args.commit if args.repo_path else record.get("active_commit"))

    cold_rows = run_queries(service, args.repo_id, commit, queries, args.top_k, cold=True)
    warm_rows = run_queries(service, args.repo_id, commit, queries, args.top_k, cold=False)

    report = {
        "label": "Hands-On Repository Latency Benchmark",
        "generated_at": datetime.now(UTC).isoformat(),
        "repo_id": args.repo_id,
        "commit": commit,
        "query_set_limitation": limitation,
        "cpu_only": {
            "parser": "Tree-sitter CPU",
            "embedding": "llama.cpp HTTP server / CPU configuration",
            "vector_search": "NumPy CPU exact dot product",
            "graph_search": "Python in-memory structural graph",
        },
        "indexing": indexing,
        "quality": {
            "cold": quality_summary(cold_rows),
            "warm": quality_summary(warm_rows),
        },
        "latency": {
            "cold": latency_summary(cold_rows),
            "warm": latency_summary(warm_rows),
            "by_type_cold": by_type_latency(cold_rows),
            "by_type_warm": by_type_latency(warm_rows),
        },
        "queries": {
            "cold": cold_rows,
            "warm": warm_rows,
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report["latency"], indent=2, sort_keys=True))
    print(json.dumps(report["quality"], indent=2, sort_keys=True))
    print(f"saved={output}")


def run_queries(service: ApiService, repo_id: str, commit: str, queries: list[dict[str, Any]], top_k: int, *, cold: bool) -> list[dict[str, Any]]:
    if cold:
        service._query_embedding_cache.clear()
    rows = []
    for item in queries:
        if cold:
            service._query_embedding_cache.clear()
        started = time.perf_counter()
        payload = service.query(repo_id=repo_id, commit=commit, query=item["query"], top_k=top_k)
        roundtrip_ms = (time.perf_counter() - started) * 1000.0
        results = payload["results"]
        rows.append(
            {
                "query": item["query"],
                "expected_type": item.get("type"),
                "classified_as": payload["query_type"],
                "cold": cold,
                "roundtrip_ms": roundtrip_ms,
                "timing": payload["timing"],
                "trace": payload["trace"],
                "top_result": result_ref(results[0]) if results else None,
                "expected_retrieved": any(is_relevant(r, item.get("expected", [])) for r in results[:top_k]),
                "precision_at_1": precision_at_k(results, item, 1),
                "precision_at_5": precision_at_k(results, item, 5),
                "recall_at_5": recall_at_k(results, item, 5),
                "recall_at_10": recall_at_k(results, item, 10),
                "mrr": mrr(results, item),
            }
        )
    return rows


def load_queries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("queries-file must contain a JSON list")
    return payload


def result_ref(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "rank": result.get("rank"),
        "path": result.get("path"),
        "symbol": result.get("symbol"),
        "location": f"{result.get('path')}:{result.get('start_line')}-{result.get('end_line')}",
        "score": result.get("score"),
        "evidence_type": result.get("evidence_type"),
    }


def is_relevant(result: dict[str, Any], expected: list[dict[str, Any]]) -> bool:
    for item in expected:
        if item.get("path") and result.get("path") != item.get("path"):
            continue
        if item.get("symbol") and result.get("symbol") != item.get("symbol"):
            continue
        if item.get("start_line") and int(result.get("start_line") or 0) < int(item["start_line"]):
            continue
        if item.get("end_line") and int(result.get("end_line") or 0) > int(item["end_line"]):
            continue
        return True
    return False


def precision_at_k(results: list[dict[str, Any]], item: dict[str, Any], k: int) -> float:
    subset = results[:k]
    if not subset:
        return 0.0
    return sum(1 for result in subset if is_relevant(result, item.get("expected", []))) / len(subset)


def recall_at_k(results: list[dict[str, Any]], item: dict[str, Any], k: int) -> float:
    expected = item.get("expected", [])
    if not expected:
        return 0.0
    found = 0
    for exp in expected:
        if any(is_relevant(result, [exp]) for result in results[:k]):
            found += 1
    return found / len(expected)


def mrr(results: list[dict[str, Any]], item: dict[str, Any]) -> float:
    for idx, result in enumerate(results, start=1):
        if is_relevant(result, item.get("expected", [])):
            return 1.0 / idx
    return 0.0


def quality_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "precision_at_1": avg(rows, "precision_at_1"),
        "precision_at_5": avg(rows, "precision_at_5"),
        "recall_at_5": avg(rows, "recall_at_5"),
        "recall_at_10": avg(rows, "recall_at_10"),
        "mrr": avg(rows, "mrr"),
    }


def latency_summary(rows: list[dict[str, Any]]) -> dict[str, float]:
    totals = [float(row["timing"]["total_ms"]) for row in rows]
    return {
        "median_ms": median(totals) if totals else 0.0,
        "p95_ms": percentile_nearest_rank(totals, 0.95),
    }


def by_type_latency(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["expected_type"]), []).append(row)
    return {key: latency_summary(value) for key, value in sorted(grouped.items())}


def avg(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(1, len(rows))


def percentile_nearest_rank(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(percentile * len(ordered)) - 1))
    return ordered[index]


if __name__ == "__main__":
    main()

