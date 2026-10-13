"""
Open-set evaluation module for wildlife re-identification.
Known queries must be identified correctly, while unknown queries are classified as new individuals.
Decisions rely on a decision threshold (tau) selected on validation data and frozen for testing.
"""
from typing import Dict, Optional, Union, Tuple
import numpy as np
from src.retrieval.metrics import (
    open_set_at,
    open_set_curve,
    best_threshold,
    best_threshold_per_group,
    evaluate_open_set_per_group
)


def _extract_evaluation_inputs(ranked_results, gallery_identities: np.ndarray, query_info) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extracts top similarity scores, predicted identities, true targets, datasets, and known masks."""
    top_scores = ranked_results.score[:, 0]
    top_predictions = np.asarray(gallery_identities)[ranked_results.cand[:, 0]]
    return top_scores, top_predictions, query_info.true, query_info.datasets, query_info.known


def select_decision_threshold(
    ranked_results,
    gallery_identities: np.ndarray,
    query_info,
    per_species: bool = False
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    VALIDATION ONLY: Determines the optimal decision threshold (tau) maximizing open-set performance.
    
    Returns a dictionary containing tau, BAKS, BAUS, NBA, and optional per-species thresholds.
    """
    scores, predictions, true_targets, datasets, known_mask = _extract_evaluation_inputs(
        ranked_results, gallery_identities, query_info
    )
    best_metrics = best_threshold(scores, predictions, true_targets, datasets, known_mask)
    
    if per_species:
        species_taus, _ = best_threshold_per_group(
            scores, predictions, true_targets, datasets, known_mask, query_info.species
        )
        best_metrics["species_tau"] = species_taus
        
    return best_metrics


def evaluate_open_set(
    ranked_results,
    gallery_identities: np.ndarray,
    query_info,
    tau: float,
    species_tau: Optional[dict] = None,
    with_curve: bool = False
) -> Dict[str, Union[float, list, dict]]:
    """Evaluates open-set retrieval performance using a frozen decision threshold (tau)."""
    scores, predictions, true_targets, datasets, known_mask = _extract_evaluation_inputs(
        ranked_results, gallery_identities, query_info
    )
    
    if species_tau is not None:
        results = evaluate_open_set_per_group(
            scores, predictions, true_targets, datasets, known_mask, query_info.species, species_tau, tau
        )
        results["tau"] = float(tau)
    else:
        results = open_set_at(tau, scores, predictions, true_targets, datasets, known_mask)
        
    if with_curve:
        curve_rows, auc_score = open_set_curve(scores, predictions, true_targets, datasets, known_mask)
        results["curve"] = curve_rows
        results["AUC"] = auc_score
        
    return results