"""Ranking metrics used for candidate discovery evaluation."""

from __future__ import annotations

from typing import Iterable

import numpy as np


def dcg_at_k(relevances: Iterable[float], k: int) -> float:
    """Discounted cumulative gain at K."""
    rels = np.asarray(list(relevances)[:k], dtype=float)
    if k <= 0 or rels.size == 0:
        return 0.0
    positions = np.arange(1, rels.size + 1)
    discounts = np.log2(positions + 1)
    gains = (2**rels - 1) / discounts
    return float(np.sum(gains))


def ndcg_at_k(relevances: Iterable[float], k: int) -> float:
    """Normalised discounted cumulative gain at K."""
    rel_list = list(relevances)
    ideal = sorted(rel_list, reverse=True)
    ideal_dcg = dcg_at_k(ideal, k)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(rel_list, k) / ideal_dcg


def reciprocal_rank(relevances: Iterable[float]) -> float:
    """Reciprocal rank for a single query."""
    for index, relevance in enumerate(relevances, start=1):
        if relevance > 0:
            return 1.0 / index
    return 0.0


def mrr(relevance_lists: Iterable[Iterable[float]]) -> float:
    """Mean reciprocal rank over many queries."""
    relevance_lists = [list(values) for values in relevance_lists]
    if not relevance_lists:
        return 0.0
    return float(np.mean([reciprocal_rank(values) for values in relevance_lists]))


def average_precision(relevances: Iterable[float]) -> float:
    """Average precision for one query using binary relevance."""
    hits = 0
    precision_sum = 0.0
    rel_list = list(relevances)
    for index, relevance in enumerate(rel_list, start=1):
        if relevance > 0:
            hits += 1
            precision_sum += hits / index
    return precision_sum / hits if hits else 0.0


def map_at_k(relevance_lists: Iterable[Iterable[float]], k: int) -> float:
    """Mean average precision at K."""
    relevance_lists = [list(values)[:k] for values in relevance_lists]
    if not relevance_lists:
        return 0.0
    return float(np.mean([average_precision(values) for values in relevance_lists]))


def precision_at_k(relevances: Iterable[float], k: int) -> float:
    """Precision at K using relevance > 0 as a hit."""
    rel_list = list(relevances)[:k]
    if k <= 0 or not rel_list:
        return 0.0
    return float(np.mean([value > 0 for value in rel_list]))


def recall_at_k(relevances: Iterable[float], k: int, n_relevant: int) -> float:
    """Recall at K using the known number of relevant items for the query."""
    if k <= 0 or n_relevant <= 0:
        return 0.0
    rel_list = list(relevances)[:k]
    hits = sum(1 for value in rel_list if value > 0)
    return hits / n_relevant


def evaluate_rankings(
    predictions: dict[str, list[str]],
    ground_truth: dict[str, dict[str, float]],
    k_values: Iterable[int] = (1, 5, 10, 20),
) -> dict[str, float]:
    """Evaluate a full set of predictions against job-level ground truth."""
    k_values = sorted(set(int(k) for k in k_values))
    if not ground_truth:
        raise ValueError("Ground truth is empty.")

    relevance_lists: list[list[float]] = []
    relevant_counts: list[int] = []

    for job_id, relevance_map in ground_truth.items():
        ranked_candidates = predictions.get(str(job_id), [])
        rel_list = [float(relevance_map.get(str(candidate_id), 0.0)) for candidate_id in ranked_candidates]
        relevance_lists.append(rel_list)
        relevant_counts.append(sum(1 for value in relevance_map.values() if value > 0))

    results: dict[str, float] = {"n_queries": float(len(relevance_lists))}
    for k in k_values:
        results[f"ndcg@{k}"] = float(np.mean([ndcg_at_k(values, k) for values in relevance_lists]))
        results[f"precision@{k}"] = float(
            np.mean([precision_at_k(values, k) for values in relevance_lists])
        )
        results[f"recall@{k}"] = float(
            np.mean(
                [
                    recall_at_k(values, k, n_relevant)
                    for values, n_relevant in zip(relevance_lists, relevant_counts, strict=False)
                ]
            )
        )
        results[f"map@{k}"] = map_at_k(relevance_lists, k)
    results["mrr"] = mrr(relevance_lists)
    return results


def print_metrics(metrics: dict[str, float]) -> None:
    """Pretty-print evaluation results to stdout."""
    query_count = int(metrics.get("n_queries", 0))
    print("\n" + "=" * 48)
    print(f"Evaluation Results (n={query_count} queries)")
    print("=" * 48)
    for key in sorted(metric for metric in metrics if metric != "n_queries"):
        print(f"{key:<16} {metrics[key]:.4f}")
    print("=" * 48 + "\n")
