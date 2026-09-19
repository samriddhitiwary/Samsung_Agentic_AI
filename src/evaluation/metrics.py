"""MTEB-compatible retrieval metrics for AppsRetrieval-style rankings."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Any

from mteb._evaluators.retrieval_metrics import calculate_retrieval_scores


DEFAULT_K_VALUES = (1, 3, 5, 10)


@dataclass(frozen=True)
class EvaluationReport:
    """Compact evaluation result plus AppsRetrieval sanity checks."""

    metrics: dict[str, float]
    sanity_checks: dict[str, bool]
    positive_ranks: dict[str, int | None]
    evaluated_queries: int


def _ranked_doc_ids(doc_scores: Mapping[str, float], limit: int | None = None) -> list[str]:
    ranked = sorted(doc_scores.items(), key=lambda item: (item[1], item[0]), reverse=True)
    if limit is not None:
        ranked = ranked[:limit]
    return [doc_id for doc_id, _ in ranked]


def _validate_rankings(
    rankings: Mapping[str, Mapping[str, float]],
    qrels: Mapping[str, Mapping[str, int]],
) -> None:
    missing = set(qrels) - set(rankings)
    if missing:
        sample = ", ".join(sorted(missing)[:5])
        raise ValueError(f"Rankings are missing {len(missing)} qrel queries, e.g. {sample}")

    for query_id in qrels:
        scores = rankings[query_id]
        if not scores:
            raise ValueError(f"Ranking for {query_id} is empty")
        bad_scores = [
            doc_id
            for doc_id, score in scores.items()
            if not isinstance(score, int | float) or not isfinite(float(score))
        ]
        if bad_scores:
            sample = ", ".join(bad_scores[:5])
            raise ValueError(f"Ranking for {query_id} has non-finite scores, e.g. {sample}")


def positive_ranks_at_k(
    rankings: Mapping[str, Mapping[str, float]],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    k: int,
) -> dict[str, int | None]:
    """Return the 1-based rank of each query's single positive doc within top k."""

    ranks: dict[str, int | None] = {}
    for query_id, rels in qrels.items():
        positive_docs = [doc_id for doc_id, label in rels.items() if label > 0]
        if len(positive_docs) != 1:
            raise ValueError(
                f"Expected exactly one positive qrel for {query_id}, found {len(positive_docs)}"
            )
        positive_doc_id = positive_docs[0]
        ranked_doc_ids = _ranked_doc_ids(rankings[query_id], limit=k)
        try:
            ranks[query_id] = ranked_doc_ids.index(positive_doc_id) + 1
        except ValueError:
            ranks[query_id] = None
    return ranks


def evaluate_rankings(
    rankings: Mapping[str, Mapping[str, float]],
    qrels: Mapping[str, Mapping[str, int]],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> EvaluationReport:
    """Evaluate an MTEB ranking dict and run AppsRetrieval-specific checks.

    Expected input shape:
        {"q5001": {"d5001": 12.34, "d1234": 10.20}}
    """

    k_values = tuple(sorted(set(k_values)))
    _validate_rankings(rankings, qrels)

    official = calculate_retrieval_scores(rankings, qrels, k_values)
    metrics: dict[str, float] = {}
    for source in (official.ndcg, official.mrr, official.recall, official.hit_rate):
        for key, value in source.items():
            metric_name = key.lower().replace("@", "_at_")
            metrics[metric_name] = float(value)

    ranks_at_10 = positive_ranks_at_k(rankings, qrels, k=10)
    expected_mrr_10 = sum(0.0 if rank is None else 1.0 / rank for rank in ranks_at_10.values()) / len(qrels)
    expected_hit_rate_10 = sum(rank is not None for rank in ranks_at_10.values()) / len(qrels)

    # MTEB rounds pytrec_eval-derived aggregate metrics to 5 decimals.
    aggregate_tolerance = 5e-6
    sanity_checks = {
        "one_positive_qrel_per_query": all(
            sum(1 for label in rels.values() if label > 0) == 1 for rels in qrels.values()
        ),
        "mrr_at_10_matches_positive_reciprocal_rank": abs(
            metrics["mrr_at_10"] - expected_mrr_10
        )
        < 1e-12,
        "hit_rate_at_10_matches_positive_top10_fraction": abs(
            metrics["hitrate_at_10"] - expected_hit_rate_10
        )
        <= aggregate_tolerance,
    }

    return EvaluationReport(
        metrics=metrics,
        sanity_checks=sanity_checks,
        positive_ranks=ranks_at_10,
        evaluated_queries=len(qrels),
    )


def metrics_payload(
    report: EvaluationReport,
    *,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Create the persisted metrics JSON payload."""

    return {
        "metadata": dict(metadata),
        "metrics": report.metrics,
        "sanity_checks": report.sanity_checks,
        "evaluated_queries": report.evaluated_queries,
    }
