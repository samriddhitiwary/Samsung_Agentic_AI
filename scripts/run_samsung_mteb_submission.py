from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import mteb

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retrieval.samsung_mteb_encoder import SamsungAppsRetrievalEncoder, frozen_encoder_config
from src.utils.env import env_path, llama_server_url, load_dotenv


TASK_NAME = "AppsRetrieval"
SUBMISSION_DIR = PROJECT_ROOT / "submission"
RESULT_JSON = SUBMISSION_DIR / "appsretrieval_results.json"
VALIDATION_JSON = SUBMISSION_DIR / "appsretrieval_validation.json"
PREDICTION_DIR = SUBMISSION_DIR / "mteb_predictions"
FROZEN_CACHE_DIR = PROJECT_ROOT / "data/cache/jina_code_1.5b_full_1024"
FROZEN_METRICS = {
    "NDCG@10": 0.86950,
    "MRR@10": 0.84141,
    "HitRate@10": 0.95564,
}


def parse_args() -> argparse.Namespace:
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Generate Samsung Theme 1 MTEB AppsRetrieval submission JSON.")
    parser.add_argument("--batch-size", type=int, default=4, help="MTEB DataLoader/embedding batch size.")
    parser.add_argument("--server-url", default=llama_server_url())
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=env_path(
            "JCR_JINA_CODE_1_5B_FULL_1024_CACHE_DIR",
            FROZEN_CACHE_DIR,
            project_root=PROJECT_ROOT,
        ),
    )
    parser.add_argument(
        "--require-cache",
        action="store_true",
        help="Fail instead of falling back to llama.cpp encoding if frozen caches are unavailable.",
    )
    parser.add_argument(
        "--keep-predictions",
        action="store_true",
        help="Keep MTEB prediction JSON used for local validation. Not required for Samsung release upload.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)

    model = SamsungAppsRetrievalEncoder(
        cache_dir=args.cache_dir,
        server_url=args.server_url,
        batch_size=args.batch_size,
        require_cache=args.require_cache,
    )
    task = mteb.get_task(TASK_NAME)
    started = time.perf_counter()
    result = mteb.evaluate(
        model,
        [task],
        encode_kwargs={"batch_size": args.batch_size},
        cache=None,
        prediction_folder=PREDICTION_DIR,
        show_progress_bar=True,
    )
    runtime_seconds = time.perf_counter() - started

    task_result = list(result.task_results)[0]
    task_dict = task_result.to_dict()
    with RESULT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(task_dict, handle, indent=2, default=str)

    validation = validate_submission(
        task_dict=task_dict,
        task=task,
        encoder=model,
        runtime_seconds=runtime_seconds,
        prediction_dir=PREDICTION_DIR,
    )
    if not args.keep_predictions and PREDICTION_DIR.exists():
        shutil.rmtree(PREDICTION_DIR)
        validation["paths"]["prediction_json"] = "temporary validation artifact removed; rerun with --keep-predictions to retain it"
    with VALIDATION_JSON.open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, sort_keys=True)

    print("Samsung MTEB AppsRetrieval submission generated")
    print(f"Task: {validation['task_name']}")
    print(f"Split: {validation['split']}")
    print(f"Submission JSON: {RESULT_JSON}")
    print(f"Validation JSON: {VALIDATION_JSON}")
    print("Official metric fields:")
    for key, value in validation["official_metrics"].items():
        print(f"  {key}: {value}")
    print("Frozen-result comparison:")
    for key, value in validation["comparison_to_frozen"].items():
        print(f"  {key}: observed={value['observed']} expected={value['expected']} diff={value['absolute_difference']}")
    print(f"Validation passed: {validation['validation_passed']}")
    if not validation["validation_passed"]:
        raise SystemExit("Submission validation failed; inspect submission/appsretrieval_validation.json")
    return 0


def validate_submission(
    *,
    task_dict: dict[str, Any],
    task: Any,
    encoder: SamsungAppsRetrievalEncoder,
    runtime_seconds: float,
    prediction_dir: Path,
) -> dict[str, Any]:
    prediction_path = prediction_dir / f"{TASK_NAME}_predictions.json"
    prediction_payload = json.loads(prediction_path.read_text(encoding="utf-8"))
    predictions = prediction_payload["default"]["test"]

    task.load_data()
    split = task.dataset["default"]["test"]
    corpus_ids = {str(item) for item in split["corpus"]["id"]}
    query_ids = [str(item) for item in split["queries"]["id"]]

    prediction_query_ids = set(predictions)
    duplicate_doc_queries = []
    nonfinite_score_queries = []
    unknown_doc_queries = []
    for query_id, ranking in predictions.items():
        doc_ids = list(ranking)
        if len(doc_ids) != len(set(doc_ids)):
            duplicate_doc_queries.append(query_id)
        if any(doc_id not in corpus_ids for doc_id in doc_ids):
            unknown_doc_queries.append(query_id)
        if any(not math.isfinite(float(score)) for score in ranking.values()):
            nonfinite_score_queries.append(query_id)

    flat_metrics = flatten_numeric_metrics(task_dict)
    official_metrics = select_official_metrics(flat_metrics)
    comparison = compare_to_frozen(official_metrics)
    cache_counts = {
        event["role"]: {
            "count": event["count"],
            "cache_hit": event["cache_hit"],
            "cache_path": event["cache_path"],
        }
        for event in encoder.cache_events
    }
    checks = {
        "task_name_exact": getattr(task.metadata, "name", None) == TASK_NAME,
        "split_is_test": "test" in prediction_payload.get("default", {}),
        "all_queries_processed": len(predictions) == 3765 and prediction_query_ids == set(query_ids),
        "corpus_candidate_count": len(corpus_ids) == 8765,
        "query_count": len(query_ids) == 3765,
        "no_duplicate_document_ids_per_query": not duplicate_doc_queries,
        "all_document_ids_from_official_corpus": not unknown_doc_queries,
        "scores_are_finite": not nonfinite_score_queries,
        "json_parses_successfully": True,
        "metrics_match_frozen_within_tolerance": all(item["within_tolerance"] for item in comparison.values()),
        "corpus_cache_reused": cache_counts.get("document", {}).get("cache_hit") is True,
        "query_cache_reused": cache_counts.get("query", {}).get("cache_hit") is True,
    }
    return {
        "task_name": getattr(task.metadata, "name", None),
        "split": "test",
        "subset": "default",
        "runtime_seconds": runtime_seconds,
        "mteb_version": package_version("mteb"),
        "encoder_interface": "mteb.models.abs_encoder.AbsEncoder.encode(inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs)",
        "frozen_encoder_config": frozen_encoder_config(),
        "cache_events": encoder.cache_events,
        "official_metrics": official_metrics,
        "all_numeric_metric_fields": flat_metrics,
        "comparison_to_frozen": comparison,
        "coverage": {
            "query_count": len(query_ids),
            "processed_query_count": len(predictions),
            "corpus_size": len(corpus_ids),
            "duplicate_doc_query_count": len(duplicate_doc_queries),
            "unknown_doc_query_count": len(unknown_doc_queries),
            "nonfinite_score_query_count": len(nonfinite_score_queries),
        },
        "checks": checks,
        "validation_passed": all(checks.values()),
        "paths": {
            "submission_json": RESULT_JSON.as_posix(),
            "validation_json": VALIDATION_JSON.as_posix(),
            "prediction_json": prediction_path.as_posix(),
        },
        "csv_schema_note": "Official guideline provides explicit MTEB JSON generation/upload instructions but does not provide an explicit CSV schema.",
    }


def flatten_numeric_metrics(value: Any, prefix: str = "") -> dict[str, float]:
    flat: dict[str, float] = {}
    if isinstance(value, dict):
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            flat.update(flatten_numeric_metrics(child, child_prefix))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            child_prefix = f"{prefix}[{index}]"
            flat.update(flatten_numeric_metrics(child, child_prefix))
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        flat[prefix] = float(value)
    return flat


def select_official_metrics(flat_metrics: dict[str, float]) -> dict[str, float]:
    selected: dict[str, float] = {}
    wanted = {
        "NDCG@10": ("ndcg_at_10", "ndcg@10", "ndcg_cut_10"),
        "MRR@10": ("mrr_at_10", "mrr@10"),
        "HitRate@10": ("hit_rate_at_10", "hit@10", "hitrate@10"),
        "Recall@100": ("recall_at_100", "recall@100"),
    }
    for label, needles in wanted.items():
        for key, value in flat_metrics.items():
            normalized = key.lower().replace("-", "_")
            if any(needle in normalized for needle in needles):
                selected[key] = value
                break
    return selected


def compare_to_frozen(official_metrics: dict[str, float]) -> dict[str, dict[str, Any]]:
    comparison: dict[str, dict[str, Any]] = {}
    field_needles = {
        "NDCG@10": "ndcg_at_10",
        "MRR@10": "mrr_at_10",
        "HitRate@10": "hit_rate_at_10",
    }
    for label, expected in FROZEN_METRICS.items():
        observed_key = next(
            (key for key in official_metrics if field_needles[label] in key.lower()),
            None,
        )
        observed = official_metrics.get(observed_key) if observed_key else None
        diff = None if observed is None else abs(observed - expected)
        comparison[label] = {
            "metric_field": observed_key,
            "observed": observed,
            "expected": expected,
            "absolute_difference": diff,
            "within_tolerance": bool(diff is not None and diff <= 1e-4),
        }
    return comparison


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


if __name__ == "__main__":
    raise SystemExit(main())
