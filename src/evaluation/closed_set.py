"""
Closed-set evaluation module for wildlife re-identification.
Only known queries are evaluated (identities present in the gallery) without thresholding.
Reports BAKS (benchmark headline metric), standard top-K accuracy, and mAP.
"""
from typing import Dict, Tuple, Union
import numpy as np
from src.retrieval.metrics import (
    closed_set_baks,
    ranked_topk_accuracy,
    ranked_average_precision,
    dataset_balanced_mean
)


def evaluate_closed_set(
    ranked_results,
    gallery_identities: np.ndarray,
    query_info,
    ks: Tuple[int, ...] = (1, 5, 10),
    per_dataset: bool = False
) -> Dict[str, Union[float, Dict[str, float]]]:
    """
    Evaluates closed-set retrieval performance for known queries.
    
    Args:
        ranked_results: Object containing candidate indices (.cand) and similarity scores (.score).
        gallery_identities: Array mapping gallery indices to animal identities.
        query_info: Container holding true identities (.true), dataset tags (.datasets), and known masks (.known).
        ks: Tuple of top-k ranks for accuracy evaluation.
        per_dataset: Whether to compute per-dataset breakdown for top-1 accuracy.
        
    Returns:
        Dictionary containing evaluation metrics including BAKS, top-k accuracies, mAP, and shortlist recall.
    """
    gallery_identities = np.asarray(gallery_identities)
    candidate_identities = gallery_identities[ranked_results.cand]  # Shape: (N_queries, K)
    top_predictions = candidate_identities[:, 0]
    
    metrics_output = {
        "BAKS": closed_set_baks(
            ranked_results.score[:, 0],
            top_predictions,
            query_info.true,
            query_info.datasets,
            query_info.known
        ),
        "accuracy": float(np.mean(top_predictions[query_info.known] == query_info.true[query_info.known]))
        if query_info.known.any() else float("nan")
    }
    
    metrics_output.update({
        f"top{k}": val for k, val in ranked_topk_accuracy(
            candidate_identities, query_info.true, query_info.known, ks
        ).items()
    })
    
    identity_counts = {identity: count for identity, count in zip(*np.unique(gallery_identities, return_counts=True))}
    positive_counts = np.array([identity_counts.get(true_identity, 0) for true_identity in query_info.true])
    average_precisions = ranked_average_precision(candidate_identities, query_info.true, positive_counts)
    
    metrics_output["mAP"] = float(np.nanmean(average_precisions[query_info.known])) if query_info.known.any() else float("nan")
    metrics_output["mAP_ds"] = dataset_balanced_mean(average_precisions, query_info.datasets, query_info.known)
    
    metrics_output["shortlist_recall"] = float(
        np.mean((candidate_identities[query_info.known] == query_info.true[query_info.known][:, None]).any(axis=1))
    ) if query_info.known.any() else float("nan")
    
    if per_dataset:
        correct_predictions = top_predictions == query_info.true
        metrics_output["per_dataset"] = {
            dataset_name: float(np.mean(correct_predictions[(query_info.datasets == dataset_name) & query_info.known]))
            for dataset_name in np.unique(query_info.datasets[query_info.known])
        }
        
    return metrics_output
