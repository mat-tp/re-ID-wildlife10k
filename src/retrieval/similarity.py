"""
Similarity Primitives

Provides memory-efficient chunked search and scoring primitives for image retrieval tasks.
Prevents out-of-memory errors on large gallery matrix operations (e.g., 25,000 queries x 85,000 gallery images).

Evidence Types:
1. Global / Complementary / Baseline: L2-normalized feature vectors evaluated via Cosine Similarity.
2. Local (ALIKED + LightGlue): Geometrically verified keypoint match counts for a shortlist of candidate pairs.
"""

from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple
import numpy as np


def normalize_l2(
    feature_vectors: np.ndarray, 
    epsilon: float = 1e-12
) -> np.ndarray:
    """
    Normalizes feature vectors to unit length using L2 norm.
    
    Args:
        feature_vectors: Input feature array of shape (N, D).
        epsilon: Small constant to avoid division by zero.
        
    Returns:
        L2-normalized feature array of shape (N, D) in float32 format.
    """
    # Force float32 precision for consistent memory consumption and PyTorch/NumPy compatibility
    vectors = np.asarray(feature_vectors, dtype=np.float32)
    
    # Compute L2 norm along the feature dimension (axis=1) while maintaining shape (N, 1)
    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    
    # Clamp minimum value to epsilon to prevent zero division
    return vectors / np.maximum(vector_norms, epsilon)


def find_top_k_cosine_neighbors(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    top_k: int,
    chunk_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes exact top-K gallery neighbors for each query using chunked cosine similarity.

    Args:
        query_features: Query feature matrix of shape (N_queries, D).
        gallery_features: Gallery feature matrix of shape (N_gallery, D).
        top_k: Number of nearest gallery neighbors to retrieve per query.
        chunk_size: Number of queries processed simultaneously to conserve memory.

    Returns:
        Tuple containing:
            - neighbor_indices: Array of shape (N_queries, top_k) with gallery indices.
            - similarity_scores: Array of shape (N_queries, top_k) with cosine similarity scores,
              sorted in descending order.
    """
    num_queries = len(query_features)
    num_gallery = len(gallery_features)
    
    # Ensure top_k does not exceed the total available gallery items
    effective_top_k = min(top_k, num_gallery)

    # Fast Path: Attempt GPU acceleration using PyTorch if CUDA is available
    try:
        import torch

        if torch.cuda.is_available():
            # L2-normalize vectors so matrix multiplication equals cosine similarity
            normalized_queries = normalize_l2(query_features)
            normalized_gallery = normalize_l2(gallery_features)

            with torch.no_grad():
                # Transpose gallery vectors and transfer to GPU once: shape (D, N_gallery)
                gallery_tensor_transposed = (
                    torch.from_numpy(normalized_gallery).cuda().t().contiguous()
                )

                top_indices = np.empty((num_queries, effective_top_k), dtype=np.int64)
                top_scores = np.empty((num_queries, effective_top_k), dtype=np.float32)

                # Process query matrix in chunks to avoid GPU OOM on large query sets
                for start_idx in range(0, num_queries, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_queries)
                    
                    # Transfer current query chunk to GPU
                    query_chunk = torch.from_numpy(
                        normalized_queries[start_idx:end_idx]
                    ).cuda()

                    # Matrix multiplication yields cosine similarity matrix of shape (chunk_size, N_gallery)
                    similarity_matrix = torch.matmul(
                        query_chunk, gallery_tensor_transposed
                    )
                    
                    # Retrieve top-K values and indices directly on GPU
                    chunk_scores, chunk_indices = torch.topk(
                        similarity_matrix, k=effective_top_k, dim=1
                    )

                    # Store results back into CPU memory
                    top_indices[start_idx:end_idx] = chunk_indices.cpu().numpy()
                    top_scores[start_idx:end_idx] = chunk_scores.cpu().numpy()

                return top_indices, top_scores
    except Exception:
        # Fall back to NumPy CPU processing if PyTorch/CUDA is unavailable or fails
        pass

    # CPU Fallback Path (NumPy)
    normalized_queries = normalize_l2(query_features)
    normalized_gallery = normalize_l2(gallery_features)
    
    # Pre-transpose gallery matrix in contiguous memory for optimized dot products
    gallery_transposed = np.ascontiguousarray(normalized_gallery.T)

    top_indices = np.empty((num_queries, effective_top_k), dtype=np.int64)
    top_scores = np.empty((num_queries, effective_top_k), dtype=np.float32)

    # Process query matrix in chunks on CPU
    for start_idx in range(0, num_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, num_queries)
        query_chunk = normalized_queries[start_idx:end_idx]

        # Compute cosine similarity matrix chunk: shape (chunk_size, N_gallery)
        similarity_chunk = query_chunk @ gallery_transposed

        # Use argpartition (O(N)) to find unsorted top-k indices rather than sorting the whole row (O(N log N))
        unsorted_top_k_indices = np.argpartition(
            -similarity_chunk, effective_top_k - 1, axis=1
        )[:, :effective_top_k]

        # Extract score values corresponding to the partitioned top-k indices
        unsorted_top_k_scores = np.take_along_axis(
            similarity_chunk, unsorted_top_k_indices, axis=1
        )
        
        # Sort only the top-k subset in descending order
        sort_orders = np.argsort(-unsorted_top_k_scores, axis=1)

        # Map back indices and scores into sorted order
        top_indices[start_idx:end_idx] = np.take_along_axis(
            unsorted_top_k_indices, sort_orders, axis=1
        )
        top_scores[start_idx:end_idx] = np.take_along_axis(
            unsorted_top_k_scores, sort_orders, axis=1
        )

    return top_indices, top_scores


def compute_candidate_cosine_scores(
    query_features: np.ndarray,
    gallery_features: np.ndarray,
    candidate_indices: np.ndarray,
    chunk_size: int = 256,
) -> np.ndarray:
    """
    Computes cosine similarity between queries and a pre-selected shortlist of candidate gallery items.

    Used to score candidate pairs produced by a primary candidate generation branch.

    Args:
        query_features: Query feature matrix of shape (N_queries, D).
        gallery_features: Gallery feature matrix of shape (N_gallery, D).
        candidate_indices: Candidate gallery indices of shape (N_queries, K_candidates).
        chunk_size: Number of queries processed per iteration chunk.

    Returns:
        Array of shape (N_queries, K_candidates) containing cosine similarity scores.
    """
    normalized_queries = normalize_l2(query_features)
    normalized_gallery = normalize_l2(gallery_features)

    num_queries = len(query_features)
    candidate_scores = np.empty(candidate_indices.shape, dtype=np.float32)

    # Chunk processing to optimize CPU memory footprint
    for start_idx in range(0, num_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, num_queries)

        query_chunk = normalized_queries[start_idx:end_idx]
        candidate_chunk_indices = candidate_indices[start_idx:end_idx]
        
        # Slice candidate feature vectors: shape (chunk_size, K_candidates, D)
        gallery_candidates_chunk = normalized_gallery[candidate_chunk_indices]

        # Einstein summation for pair-wise dot products along feature dimension 'd':
        # query_chunk: (N, d), gallery_candidates_chunk: (N, K, d) -> result: (N, K)
        candidate_scores[start_idx:end_idx] = np.einsum(
            "nd,nkd->nk", query_chunk, gallery_candidates_chunk
        )

    return candidate_scores


class LocalFeatureScorer:
    """
    Computes local feature matching scores (e.g., RANSAC inlier count) for candidate pairs using
    ALIKED keypoint extraction and LightGlue feature matching.

    Keypoint features are cached per image using an LRU strategy, and pairwise scores are memoized to
    avoid redundant computations during iterative experiment runs.

    Note:
        Evaluating every pair requires a dedicated LightGlue forward pass.
        Keep the candidate count (K) small (e.g., 10 to 20 candidates per query).
    """

    def __init__(
        self,
        image_loader_fn: Callable[[int], np.ndarray],
        local_model: Optional[object] = None,
        max_cached_images: int = 3000,
    ) -> None:
        """
        Args:
            image_loader_fn: Function that takes an image row/ID and returns an RGB uint8 NumPy array.
            local_model: Local feature extractor/matcher instance (e.g., LocalModel).
                         Instantiated automatically if not provided.
            max_cached_images: Maximum number of extracted keypoint representations to cache in memory.
        """
        # Lazy initialization of local model if none is provided
        if local_model is None:
            from src.models.local_models import LocalModel
            local_model = LocalModel()

        self.local_model = local_model
        self.load_image = image_loader_fn

        # LRU cache to store extracted keypoints per image ID
        self._keypoint_cache: OrderedDict[int, object] = OrderedDict()
        self._max_cached_images = max_cached_images
        
        # Dictionary cache for memoizing pair matching scores: (query_id, gallery_id) -> score
        self._pairwise_score_cache: Dict[Tuple[int, int], float] = {}

    def _extract_or_get_features(self, image_id: int) -> object:
        """Extracts local features for an image, utilizing the LRU cache if already extracted."""
        # Return cached feature representation if available and update LRU position
        if image_id in self._keypoint_cache:
            self._keypoint_cache.move_to_end(image_id)
            return self._keypoint_cache[image_id]

        # Load image array and extract features using the underlying local model
        image = self.load_image(image_id)
        extracted_features = self.local_model.extract(image)

        # Store in LRU cache
        self._keypoint_cache[image_id] = extracted_features

        # Evict oldest item if cache size exceeds capacity limit
        if len(self._keypoint_cache) > self._max_cached_images:
            self._keypoint_cache.popitem(last=False)

        return extracted_features

    def score_pairs(
        self, 
        query_image_ids: np.ndarray, 
        candidate_gallery_ids: np.ndarray
    ) -> np.ndarray:
        """
        Computes pairwise matching scores between query images and candidate gallery images.

        Args:
            query_image_ids: Array of query image row IDs of shape (N_queries,).
            candidate_gallery_ids: Matrix of candidate image row IDs of shape (N_queries, K_candidates).

        Returns:
            Array of shape (N_queries, K_candidates) with feature matching scores in float32.
        """
        pair_scores = np.zeros(candidate_gallery_ids.shape, dtype=np.float32)

        for query_idx, query_id in enumerate(query_image_ids):
            query_id_int = int(query_id)
            query_features = self._extract_or_get_features(query_id_int)

            for candidate_idx, gallery_id in enumerate(candidate_gallery_ids[query_idx]):
                gallery_id_int = int(gallery_id)
                pair_key = (query_id_int, gallery_id_int)

                # Compute and cache pair score if not previously computed
                if pair_key not in self._pairwise_score_cache:
                    gallery_features = self._extract_or_get_features(gallery_id_int)
                    score = self.local_model.score(query_features, gallery_features)
                    self._pairwise_score_cache[pair_key] = score

                pair_scores[query_idx, candidate_idx] = self._pairwise_score_cache[pair_key]

        return pair_scores
    