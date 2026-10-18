"""
Evaluation Metrics for Closed-Set and Open-Set Re-Identification (Re-ID)

Evaluates global vector search models and threshold performance against benchmark standards.

Key Metrics:
- BAKS: Balanced Accuracy on Known Sets (mean accuracy across queries of known identities).
- BAUS: Balanced Accuracy on Unknown Sets (mean accuracy of correctly identifying unseen identities).
- Closed-Set Score: BAKS on known queries (ignoring unknown queries).
- Open-Set Score: Normalized Balanced Accuracy (NBA = (BAKS + BAUS) / 2).

Prediction Rule:
Uses 1-Nearest-Neighbor based on cosine similarity against gallery features.
For open-set evaluation, a query is classified as a "new individual" when best similarity < tau threshold.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np

# Identifier label reserved for unseen or unknown identities
UNKNOWN_IDENTITY_LABEL = "__new__"


def find_nearest_neighbors(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    gallery_labels: np.ndarray,
    chunk_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the single nearest gallery neighbor for each query image based on cosine similarity.

    Args:
        query_features: Matrix of shape (N_queries, D) with L2-normalized feature vectors.
        gallery_features: Matrix of shape (N_gallery, D) with L2-normalized feature vectors.
        gallery_labels: Array of shape (N_gallery,) with label identities.
        chunk_size: Query batch size for memory-efficient computation.

    Returns:
        Tuple containing:
            - best_similarities: Array of shape (N_queries,) with peak cosine similarities.
            - predicted_labels: Array of shape (N_queries,) with predicted identity labels.
    """
    labels_array = np.asarray(gallery_labels)
    num_queries = len(query_features)

    best_similarities = np.empty(num_queries, dtype=np.float32)
    best_neighbor_indices = np.empty(num_queries, dtype=np.int64)

    # Transpose gallery array once in contiguous layout for optimized matrix multiplication
    gallery_transposed = np.ascontiguousarray(gallery_features.T)

    # Process query batch in chunks to maintain low memory overhead
    for start_idx in range(0, num_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, num_queries)
        query_chunk = query_features[start_idx:end_idx]

        # Compute cosine similarity matrix chunk of shape (chunk_size, N_gallery)
        similarity_matrix = query_chunk @ gallery_transposed

        # Find the column index of the maximum similarity score per query row
        max_indices = similarity_matrix.argmax(axis=1)
        row_indices = np.arange(len(similarity_matrix))

        best_neighbor_indices[start_idx:end_idx] = max_indices
        best_similarities[start_idx:end_idx] = similarity_matrix[row_indices, max_indices]

    predicted_labels = labels_array[best_neighbor_indices]
    return best_similarities, predicted_labels


def compute_topk_identity_accuracy(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    gallery_labels: np.ndarray,
    query_labels: np.ndarray,
    top_k_values: Tuple[int, ...] = (1, 5, 10),
    chunk_size: int = 256,
) -> Dict[int, float]:
    """
    Computes top-K accuracy considering distinct identities (Hit@K across distinct identities).

    Args:
        query_features: Matrix of shape (N_queries, D) with normalized query features.
        gallery_features: Matrix of shape (N_gallery, D) with normalized gallery features.
        gallery_labels: Identity labels corresponding to gallery vectors.
        query_labels: Ground truth identity labels for query images.
        top_k_values: Tuple of K thresholds to evaluate.
        chunk_size: Number of queries processed per batch chunk.

    Returns:
        Dictionary mapping each K value to its corresponding accuracy score.
    """
    labels_gallery = np.asarray(gallery_labels)
    labels_query = np.asarray(query_labels)
    num_queries = len(query_features)

    hit_counts = {k: 0 for k in top_k_values}
    gallery_transposed = np.ascontiguousarray(gallery_features.T)
    max_k = max(top_k_values)

    for start_idx in range(0, num_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, num_queries)
        query_chunk = query_features[start_idx:end_idx]

        similarity_matrix = query_chunk @ gallery_transposed
        
        # Sort candidate neighbors per query row up to a candidate buffer limit (400 max)
        candidate_buffer_limit = min(400, similarity_matrix.shape[1])
        top_candidate_indices = np.argsort(-similarity_matrix, axis=1)[:, :candidate_buffer_limit]

        # Iterate over individual queries in current chunk
        for row_offset, sorted_neighbor_rows in enumerate(top_candidate_indices):
            query_idx = start_idx + row_offset
            ground_truth_label = labels_query[query_idx]

            # Extract distinct gallery identities preserving ranked similarity order
            unique_identities: List[object] = []
            for neighbor_idx in sorted_neighbor_rows:
                candidate_identity = labels_gallery[neighbor_idx]
                if candidate_identity not in unique_identities:
                    unique_identities.append(candidate_identity)
                if len(unique_identities) == max_k:
                    break

            # Evaluate top-K hits across distinct identities
            for k in top_k_values:
                if ground_truth_label in unique_identities[:k]:
                    hit_counts[k] += 1

    return {k: hit_counts[k] / num_queries for k in top_k_values}


class BenchmarkGroupIndexer:
    """
    Pre-computes dataset and identity hierarchy groupings to allow fast evaluation 
    of BAKS and BAUS across various decision thresholds (tau).
    """

    def __init__(
        self,
        true_labels: np.ndarray,
        dataset_names: np.ndarray,
        is_known_mask: np.ndarray,
    ) -> None:
        """
        Args:
            true_labels: Ground truth identity labels array.
            dataset_names: Array of dataset origin labels per query.
            is_known_mask: Boolean mask indicating if query identity is present in gallery.
        """
        labels_arr = np.asarray(true_labels)
        datasets_arr = np.asarray(dataset_names)
        self.is_known_mask = np.asarray(is_known_mask, dtype=bool)

        # Process known query samples
        self.known_indices = np.flatnonzero(self.is_known_mask)
        known_datasets = datasets_arr[self.known_indices].astype(str)
        known_identities = labels_arr[self.known_indices].astype(str)

        # Create composite dataset-identity keys to group queries by identity cells
        composite_cell_keys = np.char.add(known_datasets, "||" + known_identities)
        self.unique_cell_names, self.query_to_cell_mapping = np.unique(
            composite_cell_keys, return_inverse=True
        )
        self.queries_per_cell_count = np.bincount(self.query_to_cell_mapping)

        # Map each individual cell to its parent dataset
        dataset_of_each_cell = np.array([cell.split("||")[0] for cell in self.unique_cell_names])
        self.dataset_names, self.cell_to_dataset_mapping = np.unique(
            dataset_of_each_cell, return_inverse=True
        )
        self.cells_per_dataset_count = np.bincount(self.cell_to_dataset_mapping)

        # Process unknown query samples (unseen identities)
        self.unknown_indices = np.flatnonzero(~self.is_known_mask)
        unknown_datasets = datasets_arr[self.unknown_indices].astype(str)
        self.unknown_dataset_names, self.unknown_query_to_dataset_mapping = np.unique(
            unknown_datasets, return_inverse=True
        )
        self.queries_per_unknown_dataset_count = (
            np.bincount(self.unknown_query_to_dataset_mapping)
            if len(self.unknown_indices) > 0
            else np.array([])
        )

    def calculate_baks(self, is_correct_known_predictions: np.ndarray) -> float:
        """Calculates Balanced Accuracy on Known Sets (BAKS)."""
        if len(self.known_indices) == 0:
            return float("nan")

        # Compute accuracy per identity cell: average correct predictions within each cell
        known_correct_mask = is_correct_known_predictions[self.known_indices].astype(float)
        cell_accuracies = (
            np.bincount(self.query_to_cell_mapping, weights=known_correct_mask)
            / self.queries_per_cell_count
        )

        # Compute accuracy per dataset: average cell accuracies within each dataset
        dataset_accuracies = (
            np.bincount(self.cell_to_dataset_mapping, weights=cell_accuracies)
            / self.cells_per_dataset_count
        )

        # Return mean accuracy across all participating datasets
        return float(np.mean(dataset_accuracies))

    def calculate_baus(self, is_correct_unknown_predictions: np.ndarray) -> float:
        """Calculates Balanced Accuracy on Unknown Sets (BAUS)."""
        if len(self.unknown_indices) == 0:
            return float("nan")

        # Compute accuracy per unknown dataset: average correct 'new' identifications per dataset
        unknown_correct_mask = is_correct_unknown_predictions[self.unknown_indices].astype(float)
        dataset_accuracies = (
            np.bincount(self.unknown_query_to_dataset_mapping, weights=unknown_correct_mask)
            / self.queries_per_unknown_dataset_count
        )

        # Return mean accuracy across datasets containing unknown queries
        return float(np.mean(dataset_accuracies))


def compute_closed_set_baks(
    best_similarities: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    dataset_names: np.ndarray,
    is_known_mask: np.ndarray,
) -> float:
    """Computes closed-set BAKS score (unknown queries ignored, no threshold applied)."""
    indexer = BenchmarkGroupIndexer(true_labels, dataset_names, is_known_mask)
    correct_predictions = np.asarray(predicted_labels) == np.asarray(true_labels)
    return indexer.calculate_baks(correct_predictions)


def evaluate_open_set_at_threshold(
    threshold_tau: float,
    best_similarities: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    dataset_names: np.ndarray,
    is_known_mask: np.ndarray,
    group_indexer: Optional[BenchmarkGroupIndexer] = None,
) -> Dict[str, float]:
    """
    Evaluates open-set performance (BAKS, BAUS, NBA) at a specific similarity threshold tau.

    Args:
        threshold_tau: Similarity score threshold above which a query is assigned a gallery identity.
        best_similarities: Array of top-1 similarity scores per query.
        predicted_labels: Array of predicted top-1 identity labels.
        true_labels: Array of ground truth identity labels.
        dataset_names: Dataset source identifier for each query.
        is_known_mask: Boolean mask indicating if query identity is present in gallery.
        group_indexer: Cached BenchmarkGroupIndexer instance for accelerated batch processing.

    Returns:
        Dictionary containing threshold tau, BAKS, BAUS, and Normalized Balanced Accuracy (NBA).
    """
    indexer = group_indexer or BenchmarkGroupIndexer(true_labels, dataset_names, is_known_mask)
    similarities = np.asarray(best_similarities)

    # Acceptance rule: similarity must meet or exceed threshold tau
    is_accepted = similarities >= threshold_tau

    # Known query is correct if identity matches AND similarity is above threshold tau
    correct_known = (np.asarray(predicted_labels) == np.asarray(true_labels)) & is_accepted
    
    # Unknown query is correct if similarity is below threshold tau (classified as new identity)
    correct_unknown = ~is_accepted

    baks_score = indexer.calculate_baks(correct_known)
    baus_score = indexer.calculate_baus(correct_unknown)
    nba_score = float(np.nanmean([baks_score, baus_score]))

    return {
        "tau": float(threshold_tau),
        "BAKS": baks_score,
        "BAUS": baus_score,
        "NBA": nba_score,
    }


def compute_open_set_curve(
    best_similarities: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    dataset_names: np.ndarray,
    is_known_mask: np.ndarray,
    num_thresholds: int = 200,
) -> Tuple[List[Dict[str, float]], float]:
    """
    Computes open-set BAKS and BAUS across multiple thresholds and calculates Area Under Curve (AUC).

    Returns:
        Tuple containing list of metric evaluation dictionaries per threshold, and total AUC.
    """
    indexer = BenchmarkGroupIndexer(true_labels, dataset_names, is_known_mask)
    similarities = np.asarray(best_similarities)

    # Sample quantiles across similarity scores to evaluate thresholds evenly
    quantile_points = np.linspace(0, 1, num_thresholds)
    threshold_values = np.unique(np.quantile(similarities, quantile_points))
    
    # Prepend and append extreme boundary values
    threshold_values = np.concatenate(
        [[similarities.min() - 1e-6], threshold_values, [similarities.max() + 1e-6]]
    )

    evaluation_rows = [
        evaluate_open_set_at_threshold(
            tau, similarities, predicted_labels, true_labels, dataset_names, is_known_mask, indexer
        )
        for tau in threshold_values
    ]

    baks_values = np.array([row["BAKS"] for row in evaluation_rows])
    baus_values = np.array([row["BAUS"] for row in evaluation_rows])

    # Sort points by ascending BAKS values for trapezoidal integration
    sort_order = np.argsort(baks_values)
    sorted_baks = baks_values[sort_order]
    sorted_baus = baus_values[sort_order]

    # Calculate AUC using trapezoidal rule manual summation (NumPy 2.0 compatible)
    auc_score = float(
        np.sum((sorted_baks[1:] - sorted_baks[:-1]) * (sorted_baus[1:] + sorted_baus[:-1]) / 2.0)
    )

    return evaluation_rows, auc_score


def find_optimal_threshold(
    best_similarities: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    dataset_names: np.ndarray,
    is_known_mask: np.ndarray,
    num_thresholds: int = 200,
) -> Dict[str, float]:
    """Finds decision threshold tau maximizing Normalized Balanced Accuracy (NBA) on validation data."""
    evaluation_rows, _ = compute_open_set_curve(
        best_similarities, predicted_labels, true_labels, dataset_names, is_known_mask, num_thresholds
    )
    return max(evaluation_rows, key=lambda row: row["NBA"])


def find_optimal_threshold_per_group(
    best_similarities: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    dataset_names: np.ndarray,
    is_known_mask: np.ndarray,
    group_identifiers: np.ndarray,
    min_unknown_samples: int = 20,
    min_known_samples: int = 20,
) -> Tuple[Dict[object, float], float]:
    """
    Calibrates individual tau thresholds per species or dataset group.

    Falls back to global optimal threshold if sample count in a group is insufficient.
    """
    similarities = np.asarray(best_similarities)
    groups = np.asarray(group_identifiers)
    known_mask = np.asarray(is_known_mask, dtype=bool)

    # Compute global fallback threshold
    global_optimal_tau = find_optimal_threshold(
        similarities, predicted_labels, true_labels, dataset_names, known_mask
    )["tau"]

    group_threshold_map: Dict[object, float] = {}

    for group_name in np.unique(groups):
        group_mask = groups == group_name
        
        known_count = known_mask[group_mask].sum()
        unknown_count = (~known_mask[group_mask]).sum()

        # Fall back if sample count constraints are not met
        if unknown_count < min_unknown_samples or known_count < min_known_samples:
            group_threshold_map[group_name] = global_optimal_tau
            continue

        group_tau = find_optimal_threshold(
            similarities[group_mask],
            np.asarray(predicted_labels)[group_mask],
            np.asarray(true_labels)[group_mask],
            np.asarray(dataset_names)[group_mask],
            known_mask[group_mask],
        )["tau"]
        group_threshold_map[group_name] = group_tau

    return group_threshold_map, global_optimal_tau


def evaluate_open_set_with_group_thresholds(
    best_similarities: np.ndarray,
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
    dataset_names: np.ndarray,
    is_known_mask: np.ndarray,
    group_identifiers: np.ndarray,
    threshold_map: Dict[object, float],
    fallback_threshold: float,
) -> Dict[str, float]:
    """Evaluates open-set metrics using calibrated per-group thresholds."""
    similarities = np.asarray(best_similarities)
    groups = np.asarray(group_identifiers)

    # Resolve threshold dynamically for each individual query based on group membership
    query_thresholds = np.array(
        [threshold_map.get(group, fallback_threshold) for group in groups]
    )

    indexer = BenchmarkGroupIndexer(true_labels, dataset_names, is_known_mask)
    is_accepted = similarities >= query_thresholds

    correct_known = (np.asarray(predicted_labels) == np.asarray(true_labels)) & is_accepted
    correct_unknown = ~is_accepted

    baks_score = indexer.calculate_baks(correct_known)
    baus_score = indexer.calculate_baus(correct_unknown)
    nba_score = float(np.nanmean([baks_score, baus_score]))

    return {"BAKS": baks_score, "BAUS": baus_score, "NBA": nba_score}


def compute_ranked_topk_accuracy(
    ranked_candidate_ids: np.ndarray,
    true_labels: np.ndarray,
    is_known_mask: np.ndarray,
    top_k_values: Tuple[int, ...] = (1, 5, 10),
) -> Dict[int, float]:
    """
    Computes top-K accuracy from a candidate shortlist for known queries.

    Hit@K is recorded if the ground truth identity appears within the first K distinct gallery identities.
    """
    candidate_ids = np.asarray(ranked_candidate_ids)
    labels = np.asarray(true_labels)
    known_indices = np.flatnonzero(is_known_mask)

    hit_counts = {k: 0 for k in top_k_values}
    max_k = max(top_k_values)

    for query_idx in known_indices:
        ground_truth = labels[query_idx]
        
        # Collect distinct candidate identities up to max_k limit
        unique_candidates: List[object] = []
        for candidate in candidate_ids[query_idx]:
            if candidate not in unique_candidates:
                unique_candidates.append(candidate)
            if len(unique_candidates) >= max_k:
                break

        for k in top_k_values:
            if ground_truth in unique_candidates[:k]:
                hit_counts[k] += 1

    total_known_queries = max(len(known_indices), 1)
    return {k: hit_counts[k] / total_known_queries for k in top_k_values}


def compute_ranked_average_precision(
    ranked_candidate_ids: np.ndarray,
    true_labels: np.ndarray,
    total_positive_gallery_counts: np.ndarray,
) -> np.ndarray:
    """
    Computes Average Precision (AP@K) per query from a ranked candidate shortlist.

    Args:
        ranked_candidate_ids: Array of shape (N_queries, K_candidates) with candidate identities.
        true_labels: Array of shape (N_queries,) with ground truth identity labels.
        total_positive_gallery_counts: Total ground truth gallery images matching the true identity per query.

    Returns:
        Array of AP@K scores per query. Unknown queries evaluate to NaN.
    """
    candidate_ids = np.asarray(ranked_candidate_ids)
    true_labels_matrix = np.asarray(true_labels)[:, None]
    
    # Binary relevance mask: 1 where candidate matches ground truth, 0 elsewhere
    relevance_matrix = (candidate_ids == true_labels_matrix).astype(np.float64)
    k_candidates = relevance_matrix.shape[1]

    # Cumulative sum of true positives divided by position index gives precision at each rank
    cumulative_hits = np.cumsum(relevance_matrix, axis=1)
    rank_positions = np.arange(1, k_candidates + 1)
    precision_at_ranks = cumulative_hits / rank_positions

    # Denominator truncated at shortlist length K
    normalizing_denominators = np.minimum(
        np.asarray(total_positive_gallery_counts, dtype=float), k_candidates
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        average_precisions = (
            np.sum(precision_at_ranks * relevance_matrix, axis=1) / normalizing_denominators
        )

    # Set unknown queries (where positive gallery count is 0) to NaN
    average_precisions[normalizing_denominators == 0] = np.nan
    return average_precisions


def compute_dataset_balanced_mean(
    metric_values: np.ndarray,
    dataset_names: np.ndarray,
    selection_mask: np.ndarray,
) -> float:
    """Computes dataset-balanced mean score, weighting each dataset equally regardless of query count."""
    selected_values = np.asarray(metric_values, dtype=float)[selection_mask]
    selected_datasets = np.asarray(dataset_names)[selection_mask]

    if len(selected_values) == 0:
        return float("nan")

    unique_datasets = np.unique(selected_datasets)
    per_dataset_means: List[float] = []

    for dataset in unique_datasets:
        dataset_mask = selected_datasets == dataset
        dataset_values = selected_values[dataset_mask]

        if not np.all(np.isnan(dataset_values)):
            per_dataset_means.append(float(np.nanmean(dataset_values)))

    if len(per_dataset_means) == 0:
        return float("nan")

    return float(np.mean(per_dataset_means))