"""
Closed-Set Evaluation Module for Wildlife Re-Identification

Evaluates retrieval performance exclusively on known query images (queries whose true identities 
are present in the gallery) without applying decision thresholds.

Reports:
- BAKS (Balanced Accuracy on Known Sets) — The primary benchmark metric.
- Standard Top-K Accuracy — Fraction of queries with correct top-K predictions.
- mAP & Dataset-Balanced mAP — Mean Average Precision across candidates and datasets.
- Shortlist Recall — Fraction of known queries whose true identity appears in the top-K candidate list.
"""

from typing import Any, Dict, Protocol, Tuple, Union
import numpy as np

from src.retrieval.metrics import (
    closed_set_baks,
    dataset_balanced_mean,
    ranked_average_precision,
    ranked_topk_accuracy,
)


class RankedResultsProtocol(Protocol):
    """Protocol interface representing shortlisted candidate ranking outputs."""
    cand: np.ndarray  # Candidate gallery indices matrix of shape (N_queries, K)
    score: np.ndarray  # Candidate similarity scores matrix of shape (N_queries, K)


class QueryInfoProtocol(Protocol):
    """Protocol interface representing dataset query metadata container."""
    true: np.ndarray  # True ground truth identity labels array of shape (N_queries,)
    datasets: np.ndarray  # Origin dataset name strings array of shape (N_queries,)
    known: np.ndarray  # Boolean mask indicating whether query identity exists in gallery


def evaluate_closed_set(
    ranked_results: RankedResultsProtocol,
    gallery_identities: np.ndarray,
    query_info: QueryInfoProtocol,
    top_k_ranks: Tuple[int, ...] = (1, 5, 10),
    include_per_dataset_breakdown: bool = False,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Evaluates closed-set retrieval performance for known queries.

    Args:
        ranked_results: Object containing candidate gallery indices (.cand) and similarity scores (.score).
        gallery_identities: Array mapping gallery indices to animal identity labels.
        query_info: Object holding true ground truth identities (.true), dataset origin tags (.datasets),
                   and boolean known query indicators (.known).
        top_k_ranks: Tuple of rank positions K to evaluate for top-K accuracy.
        include_per_dataset_breakdown: Whether to include individual dataset accuracy breakdowns.

    Returns:
        Dictionary mapping metric names (e.g., 'BAKS', 'mAP', 'top1', 'shortlist_recall') to score values.
    """
    gallery_identities_array = np.asarray(gallery_identities)

    # Map candidate indices matrix to identity strings/labels: shape (N_queries, K)
    candidate_identities = gallery_identities_array[ranked_results.cand]
    top1_predicted_identities = candidate_identities[:, 0]

    known_queries_mask = np.asarray(query_info.known, dtype=bool)
    has_known_queries = bool(known_queries_mask.any())

    # Compute baseline closed-set BAKS and raw Top-1 Accuracy
    closed_set_baks_score = closed_set_baks(
        best_similarities=ranked_results.score[:, 0],
        predicted_labels=top1_predicted_identities,
        true_labels=query_info.true,
        dataset_names=query_info.datasets,
        is_known_mask=known_queries_mask,
    )

    if has_known_queries:
        top1_accuracy = float(
            np.mean(
                top1_predicted_identities[known_queries_mask]
                == query_info.true[known_queries_mask]
            )
        )
    else:
        top1_accuracy = float("nan")

    evaluation_metrics: Dict[str, Union[float, Dict[str, float]]] = {
        "BAKS": closed_set_baks_score,
        "accuracy": top1_accuracy,
    }

    # Evaluate Top-K accuracies across distinct identities
    topk_accuracies = ranked_topk_accuracy(
        ranked_candidate_ids=candidate_identities,
        true_labels=query_info.true,
        is_known_mask=known_queries_mask,
        top_k_values=top_k_ranks,
    )
    for rank, accuracy_score in topk_accuracies.items():
        evaluation_metrics[f"top{rank}"] = accuracy_score

    # Compute ground truth gallery counts for each query's true identity
    unique_identities, counts = np.unique(gallery_identities_array, return_counts=True)
    identity_to_gallery_count_map = dict(zip(unique_identities, counts))

    total_positive_gallery_counts = np.array(
        [identity_to_gallery_count_map.get(true_id, 0) for true_id in query_info.true]
    )

    # Compute Average Precision (AP@K) per query
    average_precisions = ranked_average_precision(
        ranked_candidate_ids=candidate_identities,
        true_labels=query_info.true,
        total_positive_gallery_counts=total_positive_gallery_counts,
    )

    if has_known_queries:
        mean_average_precision = float(
            np.nanmean(average_precisions[known_queries_mask])
        )
    else:
        mean_average_precision = float("nan")

    dataset_balanced_map = compute_dataset_balanced_mean(
        metric_values=average_precisions,
        dataset_names=query_info.datasets,
        selection_mask=known_queries_mask,
    )

    evaluation_metrics["mAP"] = mean_average_precision
    evaluation_metrics["mAP_ds"] = dataset_balanced_map

    # Compute shortlist recall: fraction of known queries where true identity appears anywhere in top-K
    if has_known_queries:
        is_match_matrix = (
            candidate_identities[known_queries_mask]
            == query_info.true[known_queries_mask][:, None]
        )
        shortlist_recall = float(np.mean(is_match_matrix.any(axis=1)))
    else:
        shortlist_recall = float("nan")

    evaluation_metrics["shortlist_recall"] = shortlist_recall

    # Compute per-dataset accuracy breakdown if requested
    if include_per_dataset_breakdown and has_known_queries:
        is_correct_prediction = top1_predicted_identities == query_info.true
        unique_known_datasets = np.unique(query_info.datasets[known_queries_mask])

        per_dataset_accuracies: Dict[str, float] = {}
        for dataset_name in unique_known_datasets:
            dataset_query_mask = (query_info.datasets == dataset_name) & known_queries_mask
            per_dataset_accuracies[dataset_name] = float(
                np.mean(is_correct_prediction[dataset_query_mask])
            )

        evaluation_metrics["per_dataset"] = per_dataset_accuracies

    return evaluation_metrics