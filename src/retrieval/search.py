"""
Gallery and Search Primitives

Implements two-stage retrieval for image re-identification:
  Stage 1: A fast global vector similarity search over the full gallery (using cosine similarity
           or Locality-Sensitive Hashing) to generate a shortlist of K top candidates.
  Stage 2: Heavy multi-modal re-scoring models (such as local ALIKED + LightGlue matching) score
           only the shortlisted candidates before fusion and final ranking.

All downstream evaluation metrics (Accuracy, mAP, BAKS, BAUS) consume the structured `RankedCandidates` result.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

# Updated module references for normalized features and chunked similarity
from src.retrieval.similarity import find_top_k_cosine_neighbors, normalize_l2


@dataclass
class RankedCandidates:
    """
    Encapsulates candidate search results across all queries.

    Attributes:
        candidate_indices: Array of shape (N_queries, K) containing candidate row indices into
                           the gallery dataset, sorted by score in descending order.
        candidate_scores: Array of shape (N_queries, K) containing similarity scores corresponding
                          to candidate_indices, sorted in descending order.
    """

    candidate_indices: np.ndarray
    candidate_scores: np.ndarray

    def get_top_1_predictions(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Retrieves the top-1 candidate gallery position and similarity score for each query.

        Returns:
            Tuple containing:
                - top1_indices: Array of shape (N_queries,) with peak gallery positions.
                - top1_scores: Array of shape (N_queries,) with peak similarity scores.
        """
        top1_indices = self.candidate_indices[:, 0]
        top1_scores = self.candidate_scores[:, 0]
        return top1_indices, top1_scores

    def resort_candidates(self, new_similarity_scores: np.ndarray) -> "RankedCandidates":
        """
        Re-ranks the existing shortlisted candidates using an updated scoring matrix.

        Used during multi-modal score fusion to update candidate ranks without re-evaluating
        the entire gallery.

        Args:
            new_similarity_scores: Updated score matrix of shape (N_queries, K).

        Returns:
            A new RankedCandidates instance sorted according to the new similarity scores.
        """
        # Sort scores in descending order along axis 1 using a stable sort algorithm
        sort_indices = np.argsort(-new_similarity_scores, axis=1, kind="stable")

        reordered_candidate_indices = np.take_along_axis(
            self.candidate_indices, sort_indices, axis=1
        )
        reordered_candidate_scores = np.take_along_axis(
            new_similarity_scores, sort_indices, axis=1
        )

        return RankedCandidates(
            candidate_indices=reordered_candidate_indices,
            candidate_scores=reordered_candidate_scores,
        )


def rank_by_embedding_similarity(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    top_k_candidates: int = 100,
    chunk_size: int = 512,
) -> RankedCandidates:
    """
    Performs Stage 1 candidate generation using exact chunked cosine similarity.

    Args:
        query_features: Query feature matrix of shape (N_queries, D).
        gallery_features: Gallery feature matrix of shape (N_gallery, D).
        top_k_candidates: Number of top candidate gallery items to retrieve per query.
        chunk_size: Query chunk size to prevent memory overflows.

    Returns:
        RankedCandidates object containing top-K gallery indices and similarity scores.
    """
    candidate_indices, candidate_scores = find_top_k_cosine_neighbors(
        query_features=query_features,
        gallery_features=gallery_features,
        top_k=top_k_candidates,
        chunk_size=chunk_size,
    )
    return RankedCandidates(
        candidate_indices=candidate_indices,
        candidate_scores=candidate_scores,
    )


def extract_distinct_identities(
    ranked_results: RankedCandidates,
    gallery_identity_labels: np.ndarray,
    max_distinct_k: int = 5,
) -> List[List[Tuple[object, float, int]]]:
    """
    Collapses multiple candidate image hits into a list of top distinct identities per query.

    Args:
        ranked_results: Ranked candidate predictions for all queries.
        gallery_identity_labels: Identity labels for each item in the gallery.
        max_distinct_k: Maximum number of distinct identities to retain per query.

    Returns:
        List containing, per query, a list of tuples formatted as:
        (identity_label, best_similarity_score, gallery_position_index)
    """
    distinct_results_per_query: List[List[Tuple[object, float, int]]] = []

    for query_candidates, query_scores in zip(
        ranked_results.candidate_indices, ranked_results.candidate_scores
    ):
        seen_identities = set()
        query_distinct_list: List[Tuple[object, float, int]] = []

        for gallery_position, similarity_score in zip(query_candidates, query_scores):
            identity_label = gallery_identity_labels[gallery_position]

            # Skip duplicate occurrences of an identity already encountered higher in the list
            if identity_label in seen_identities:
                continue

            seen_identities.add(identity_label)
            query_distinct_list.append(
                (identity_label, float(similarity_score), int(gallery_position))
            )

            if len(query_distinct_list) == max_distinct_k:
                break

        distinct_results_per_query.append(query_distinct_list)

    return distinct_results_per_query


def classify_open_set_identity(
    best_similarity_score: float,
    predicted_identity: object,
    similarity_threshold_tau: float,
    unknown_identity_label: str = "__new__",
) -> object:
    """
    Applies the open-set classification rule.

    If peak similarity score is greater than or equal to threshold tau, returns the predicted identity;
    otherwise, classifies the query as an unseen/new individual.

    Args:
        best_similarity_score: Maximum candidate similarity score for the query.
        predicted_identity: Top-1 candidate predicted identity label.
        similarity_threshold_tau: Decision threshold for known identity acceptance.
        unknown_identity_label: Fallback label assigned to un-matched open-set queries.

    Returns:
        The confirmed predicted identity or the unknown_identity_label.
    """
    if best_similarity_score >= similarity_threshold_tau:
        return predicted_identity
    return unknown_identity_label


class LocalitySensitiveHashingIndex:
    """
    Approximate Nearest Neighbor search using Random Hyperplane Locality-Sensitive Hashing (LSH)
    for Cosine Similarity.

    Constructs multiple hash tables generated by random hyperplanes. Candidate sets are built from
    the union of gallery items falling into identical hash buckets as the query vector. Exact cosine
    similarity is then computed exclusively on this candidate union.
    """

    def __init__(
        self,
        gallery_features: np.ndarray,
        num_hash_tables: int = 8,
        num_bits_per_table: int = 12,
        random_seed: int = 0,
    ) -> None:
        """
        Args:
            gallery_features: Gallery feature matrix of shape (N_gallery, D).
            num_hash_tables: Number of independent hash tables to construct.
            num_bits_per_table: Number of random hyperplanes (bits) per table.
            random_seed: Seed for reproducible hyper-plane initialization.
        """
        # Normalize gallery vectors to unit length
        self.normalized_gallery = normalize_l2(gallery_features)
        feature_dim = self.normalized_gallery.shape[1]

        random_generator = np.random.default_rng(random_seed)

        # Generate random hyperplane normal vectors: shape (num_tables, num_bits, feature_dim)
        self.hyperplanes = random_generator.normal(
            size=(num_hash_tables, num_bits_per_table, feature_dim)
        ).astype(np.float32)

        # Bitwise powers of 2 array for converting boolean projections into integer hash keys
        self.bit_powers = (1 << np.arange(num_bits_per_table)).astype(np.int64)

        # Construct hash tables mapping integer bucket codes to arrays of gallery entry indices
        self.hash_tables: List[Dict[int, np.ndarray]] = []

        for table_idx in range(num_hash_tables):
            # Compute bit codes for all gallery vectors in the current table
            bucket_codes = self._compute_bucket_codes(self.normalized_gallery, table_idx)

            # Sort indices by hash code values to group identical buckets together
            sorted_gallery_indices = np.argsort(bucket_codes, kind="stable")
            sorted_bucket_codes = bucket_codes[sorted_gallery_indices]

            # Locate bucket code boundary transitions
            code_boundary_starts = np.r_[0, np.flatnonzero(np.diff(sorted_bucket_codes)) + 1]
            code_boundary_ends = np.r_[code_boundary_starts[1:], len(sorted_bucket_codes)]

            # Map integer hash keys to arrays of gallery item indices
            table_dict = {
                int(sorted_bucket_codes[start_idx]): sorted_gallery_indices[start_idx:end_idx]
                for start_idx, end_idx in zip(code_boundary_starts, code_boundary_ends)
            }
            self.hash_tables.append(table_dict)

    def _compute_bucket_codes(self, feature_vectors: np.ndarray, table_index: int) -> np.ndarray:
        """Projects feature vectors against hyperplanes to derive integer bucket codes."""
        # Matrix multiplication against normal vectors yields projection signs
        hyperplane_projections = feature_vectors @ self.hyperplanes[table_index].T
        boolean_bit_vector = (hyperplane_projections > 0).astype(np.int64)

        # Convert boolean bit representation into integer hash keys using bitwise dot product
        return boolean_bit_vector @ self.bit_powers

    def search(
        self, 
        query_features: np.ndarray, 
        top_k_candidates: int = 100
    ) -> RankedCandidates:
        """
        Executes approximate nearest neighbor search for query features using LSH index buckets.

        Args:
            query_features: Query feature matrix of shape (N_queries, D).
            top_k_candidates: Number of top candidate gallery items to retrieve.

        Returns:
            RankedCandidates containing approximate top-K indices and cosine similarity scores.
        """
        normalized_queries = normalize_l2(query_features)
        num_queries = len(normalized_queries)

        # Precompute bucket hash codes across all tables for all queries
        query_table_codes = [
            self._compute_bucket_codes(normalized_queries, table_idx)
            for table_idx in range(len(self.hash_tables))
        ]

        output_indices = np.full((num_queries, top_k_candidates), -1, dtype=np.int64)
        output_scores = np.full((num_queries, top_k_candidates), -1.0, dtype=np.float32)

        for query_idx in range(num_queries):
            # Collect union of candidate gallery indices from matching buckets across all hash tables
            candidate_buckets = [
                self.hash_tables[table_idx].get(
                    int(query_table_codes[table_idx][query_idx]), np.empty(0, dtype=np.int64)
                )
                for table_idx in range(len(self.hash_tables))
            ]
            unique_candidates = np.unique(np.concatenate(candidate_buckets))

            if len(unique_candidates) == 0:
                continue

            # Compute exact cosine similarity only for shortlisted candidates in the bucket union
            candidate_vectors = self.normalized_gallery[unique_candidates]
            query_vector = normalized_queries[query_idx]
            similarity_scores = candidate_vectors @ query_vector

            # Sort shortlisted candidates by similarity score in descending order
            top_sorted_offsets = np.argsort(-similarity_scores)[:top_k_candidates]
            retained_count = len(top_sorted_offsets)

            output_indices[query_idx, :retained_count] = unique_candidates[top_sorted_offsets]
            output_scores[query_idx, :retained_count] = similarity_scores[top_sorted_offsets]

        return RankedCandidates(
            candidate_indices=output_indices,
            candidate_scores=output_scores,
        )

    @staticmethod
    def calculate_recall_vs_exact(
        approximate_results: RankedCandidates,
        exact_results: RankedCandidates,
        top_at: int = 1,
    ) -> float:
        """
        Evaluates retrieval recall of approximate LSH search compared to exact cosine search.

        Args:
            approximate_results: Ranked candidate predictions from LSH index.
            exact_results: Ranked candidate predictions from exact search.
            top_at: Number of top positions (Hit@At) to consider for overlap calculation.

        Returns:
            Mean recall fraction across all query evaluation runs.
        """
        recall_scores: List[float] = []

        for approx_candidates, exact_candidates in zip(
            approximate_results.candidate_indices, exact_results.candidate_indices
        ):
            approx_set = set(approx_candidates[:top_at])
            exact_set = set(exact_candidates[:top_at])

            # Filter out padding indices (-1) if LSH found fewer candidates than top_at
            approx_set.discard(-1)

            overlapping_hits = len(approx_set & exact_set)
            recall_scores.append(overlapping_hits / top_at)

        return float(np.mean(recall_scores))