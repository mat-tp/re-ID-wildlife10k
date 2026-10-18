"""
Open-Set Evaluation Module for Wildlife Re-Identification

Evaluates retrieval performance under open-set conditions:
- Known queries (identities present in the gallery) must be correctly matched.
- Unknown queries (unseen identities) must be correctly classified as new individuals.

Decisions rely on a global or per-species similarity threshold (tau) selected on 
validation data and frozen for testing.
"""

from typing import Any, Dict, List, Optional, Protocol, Tuple, Union
import numpy as np

from src.retrieval.metrics import (
    best_threshold,
    best_threshold_per_group,
    evaluate_open_set_per_group,
    open_set_at,
    open_set_curve,
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
    species: np.ndarray  # Species label strings array of shape (N_queries,)


def _extract_evaluation_inputs(
    ranked_results: RankedResultsProtocol,
    gallery_identities: np.ndarray,
    query_info: QueryInfoProtocol,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts top similarity scores, predicted identities, ground truth targets,
    dataset labels, and known query masks.
    """
    top_similarity_scores = ranked_results.score[:, 0]
    top1_candidate_indices = ranked_results.cand[:, 0]
    top1_predicted_identities = np.asarray(gallery_identities)[top1_candidate_indices]

    ground_truth_targets = np.asarray(query_info.true)
    dataset_names = np.asarray(query_info.datasets)
    is_known_query_mask = np.asarray(query_info.known, dtype=bool)

    return (
        top_similarity_scores,
        top1_predicted_identities,
        ground_truth_targets,
        dataset_names,
        is_known_query_mask,
    )


def select_decision_threshold(
    ranked_results: RankedResultsProtocol,
    gallery_identities: np.ndarray,
    query_info: QueryInfoProtocol,
    calibrate_per_species: bool = False,
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    VALIDATION ONLY: Calibrates the optimal decision threshold (tau) maximizing open-set performance.

    Args:
        ranked_results: Object containing candidate gallery indices (.cand) and similarity scores (.score).
        gallery_identities: Array mapping gallery indices to animal identity labels.
        query_info: Container holding ground truth identities, dataset origin tags, and species labels.
        calibrate_per_species: Whether to calibrate individual thresholds per species where supported.

    Returns:
        Dictionary containing optimal threshold tau, BAKS, BAUS, NBA, and optional per-species thresholds.
    """
    (
        similarity_scores,
        predicted_identities,
        ground_truth_targets,
        dataset_names,
        known_query_mask,
    ) = _extract_evaluation_inputs(ranked_results, gallery_identities, query_info)

    optimal_metrics = best_threshold(
        similarity_scores,
        predicted_identities,
        ground_truth_targets,
        dataset_names,
        known_query_mask,
    )

    if calibrate_per_species:
        species_thresholds, _ = best_threshold_per_group(
            similarity_scores,
            predicted_identities,
            ground_truth_targets,
            dataset_names,
            known_query_mask,
            query_info.species,
        )
        optimal_metrics["species_tau"] = species_thresholds

    return optimal_metrics


def evaluate_open_set(
    ranked_results: RankedResultsProtocol,
    gallery_identities: np.ndarray,
    query_info: QueryInfoProtocol,
    threshold_tau: float,
    species_threshold_map: Optional[Dict[str, float]] = None,
    include_evaluation_curve: bool = False,
) -> Dict[str, Union[float, List[Dict[str, float]], Dict[str, float]]]:
    """
    Evaluates open-set retrieval performance using a frozen decision threshold (tau).

    Args:
        ranked_results: Object containing candidate gallery indices (.cand) and similarity scores (.score).
        gallery_identities: Array mapping gallery indices to animal identity labels.
        query_info: Container holding true identities, dataset origin tags, and species labels.
        threshold_tau: Frozen global decision threshold.
        species_threshold_map: Optional dictionary mapping species names to calibrated tau thresholds.
        include_evaluation_curve: Whether to compute the full threshold evaluation curve and AUC.

    Returns:
        Dictionary containing BAKS, BAUS, NBA, threshold tau, and optional curve/AUC metrics.
    """
    (
        similarity_scores,
        predicted_identities,
        ground_truth_targets,
        dataset_names,
        known_query_mask,
    ) = _extract_evaluation_inputs(ranked_results, gallery_identities, query_info)

    if species_threshold_map is not None:
        open_set_metrics = evaluate_open_set_per_group(
            similarity_scores,
            predicted_identities,
            ground_truth_targets,
            dataset_names,
            known_query_mask,
            query_info.species,
            species_threshold_map,
            threshold_tau,
        )
        open_set_metrics["tau"] = float(threshold_tau)
    else:
        open_set_metrics = open_set_at(
            threshold_tau,
            similarity_scores,
            predicted_identities,
            ground_truth_targets,
            dataset_names,
            known_query_mask,
        )

    if include_evaluation_curve:
        curve_records, auc_score = open_set_curve(
            similarity_scores,
            predicted_identities,
            ground_truth_targets,
            dataset_names,
            known_query_mask,
        )
        open_set_metrics["curve"] = curve_records
        open_set_metrics["AUC"] = auc_score

    return open_set_metrics


# Backward compatibility alias
select_tau = select_decision_threshold