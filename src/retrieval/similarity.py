"""
Similarity Primitives

Provides memory-efficient chunked search and scoring primitives for image retrieval tasks.
Prevents out-of-memory errors on large gallery matrix operations (e.g., 25,000 queries x 85,000 gallery images).

Evidence Types:
1. Global / Complementary / Baseline: L2-normalized feature vectors evaluated via Cosine Similarity.
2. Local (ALIKED + LightGlue): Geometrically verified keypoint match counts for a shortlist of candidate pairs.
"""
import os
import pickle
import time
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
    vectors = np.asarray(feature_vectors, dtype=np.float32)
    vector_norms = np.linalg.norm(vectors, axis=1, keepdims=True)
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
    effective_top_k = min(top_k, num_gallery)

    # Fast Path: Attempt GPU acceleration using PyTorch if CUDA is available
    try:
        import torch

        if torch.cuda.is_available():
            normalized_queries = normalize_l2(query_features)
            normalized_gallery = normalize_l2(gallery_features)

            with torch.no_grad():
                gallery_tensor_transposed = (
                    torch.from_numpy(normalized_gallery).cuda().t().contiguous()
                )

                top_indices = np.empty((num_queries, effective_top_k), dtype=np.int64)
                top_scores = np.empty((num_queries, effective_top_k), dtype=np.float32)

                for start_idx in range(0, num_queries, chunk_size):
                    end_idx = min(start_idx + chunk_size, num_queries)
                    query_chunk = torch.from_numpy(
                        normalized_queries[start_idx:end_idx]
                    ).cuda()

                    similarity_matrix = torch.matmul(
                        query_chunk, gallery_tensor_transposed
                    )
                    
                    chunk_scores, chunk_indices = torch.topk(
                        similarity_matrix, k=effective_top_k, dim=1
                    )

                    top_indices[start_idx:end_idx] = chunk_indices.cpu().numpy()
                    top_scores[start_idx:end_idx] = chunk_scores.cpu().numpy()

                return top_indices, top_scores
    except Exception:
        pass

    # CPU Fallback Path (NumPy)
    normalized_queries = normalize_l2(query_features)
    normalized_gallery = normalize_l2(gallery_features)
    gallery_transposed = np.ascontiguousarray(normalized_gallery.T)

    top_indices = np.empty((num_queries, effective_top_k), dtype=np.int64)
    top_scores = np.empty((num_queries, effective_top_k), dtype=np.float32)

    for start_idx in range(0, num_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, num_queries)
        query_chunk = normalized_queries[start_idx:end_idx]

        similarity_chunk = query_chunk @ gallery_transposed

        unsorted_top_k_indices = np.argpartition(
            -similarity_chunk, effective_top_k - 1, axis=1
        )[:, :effective_top_k]

        unsorted_top_k_scores = np.take_along_axis(
            similarity_chunk, unsorted_top_k_indices, axis=1
        )
        sort_orders = np.argsort(-unsorted_top_k_scores, axis=1)

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

    for start_idx in range(0, num_queries, chunk_size):
        end_idx = min(start_idx + chunk_size, num_queries)
        query_chunk = normalized_queries[start_idx:end_idx]
        candidate_chunk_indices = candidate_indices[start_idx:end_idx]
        gallery_candidates_chunk = normalized_gallery[candidate_chunk_indices]

        candidate_scores[start_idx:end_idx] = np.einsum(
            "nd,nkd->nk", query_chunk, gallery_candidates_chunk
        )

    return candidate_scores


class LocalFeatureScorer:
    """
    Computes local feature matching scores for candidate pairs using ALIKED keypoint extraction
    and LightGlue feature matching.

    Keypoint features are cached per image using an LRU strategy, and pairwise scores are memoized
    and persisted to disk to avoid redundant computations across experiment runs.
    """

    def __init__(
        self,
        image_loader_fn: Callable[[int], np.ndarray],
        local_model: Optional[object] = None,
        max_cached_images: int = 3000,
        use_ransac: bool = False,
        cache_file: Optional[str] = None
    ) -> None:
        if local_model is None:
            from src.models.local_models import LocalModel
            local_model = LocalModel()

        self.local_model = local_model
        self.load_image = image_loader_fn
        self.use_ransac = use_ransac
        self.cache_file = cache_file or "results/local_pairs_cache.pkl"

        self._keypoint_cache: OrderedDict[int, object] = OrderedDict()
        self._max_cached_images = max_cached_images
        
        self._pairwise_score_cache: Dict[Tuple[int, int], float] = {}
        if os.path.isfile(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    self._pairwise_score_cache = pickle.load(f)
                print(f"[LocalFeatureScorer] Loaded {len(self._pairwise_score_cache)} pre-computed pair scores from {self.cache_file}")
            except Exception:
                self._pairwise_score_cache = {}

        # Attribute aliases
        self.m = self.local_model
        self.load = self.load_image
        self._kp = self._keypoint_cache
        self._cap = self._max_cached_images
        self._pairs = self._pairwise_score_cache

    def save_cache(self) -> None:
        if self.cache_file:
            try:
                os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
                with open(self.cache_file, "wb") as f:
                    pickle.dump(self._pairwise_score_cache, f)
            except Exception:
                pass

    def _extract_or_get_features(self, image_id: int) -> object:
        """Extracts local features for an image, utilizing the LRU cache if already extracted."""
        if image_id in self._keypoint_cache:
            self._keypoint_cache.move_to_end(image_id)
            return self._keypoint_cache[image_id]

        image = self.load_image(image_id)
        extracted_features = self.local_model.extract(image)
        self._keypoint_cache[image_id] = extracted_features

        if len(self._keypoint_cache) > self._max_cached_images:
            self._keypoint_cache.popitem(last=False)

        return extracted_features

    # Method alias
    _feats = _extract_or_get_features

    def extract_image(self, image) -> object:
        """Extracts ALIKED keypoints for a raw PIL or NumPy image."""
        return self.local_model.extract(image)

    def score_features_pair(self, feats0, feats1) -> float:
        """Scores a pair of pre-extracted local feature dictionaries."""
        return float(self.local_model.score(feats0, feats1, use_ransac=self.use_ransac))

    def score_query_against_candidate_rows(self, query_feats, cand_rows: np.ndarray) -> np.ndarray:
        """Scores query features against an array of gallery candidate rows."""
        scores = np.empty(len(cand_rows), dtype=np.float32)
        for idx, gr in enumerate(cand_rows):
            cand_feats = self._extract_or_get_features(int(gr))
            scores[idx] = self.score_features_pair(query_feats, cand_feats)
        return scores

    def score_pairs(
        self, 
        query_image_ids: np.ndarray, 
        candidate_gallery_ids: np.ndarray,
        log_interval: int = 25
    ) -> np.ndarray:
        """
        Computes pairwise matching scores between query images and candidate gallery images.
        """
        pair_scores = np.zeros(candidate_gallery_ids.shape, dtype=np.float32)
        total_q = len(query_image_ids)
        t_start = time.perf_counter()

        for query_idx, query_id in enumerate(query_image_ids):
            query_id_int = int(query_id)
            query_features = self._extract_or_get_features(query_id_int)

            for candidate_idx, gallery_id in enumerate(candidate_gallery_ids[query_idx]):
                gallery_id_int = int(gallery_id)
                pair_key = (query_id_int, gallery_id_int)

                if pair_key not in self._pairwise_score_cache:
                    gallery_features = self._extract_or_get_features(gallery_id_int)
                    score = self.score_features_pair(query_features, gallery_features)
                    self._pairwise_score_cache[pair_key] = score

                pair_scores[query_idx, candidate_idx] = self._pairwise_score_cache[pair_key]

            if log_interval and ((query_idx + 1) % log_interval == 0 or (query_idx + 1) == total_q):
                elapsed = time.perf_counter() - t_start
                rate = (query_idx + 1) / max(elapsed, 0.001)
                eta = (total_q - (query_idx + 1)) / max(rate, 0.001)
                print(f"[local matching] {query_idx + 1}/{total_q} queries processed ({((query_idx + 1)/total_q)*100:.1f}%) | ETA: {eta:.0f}s", flush=True)
                self.save_cache()

        self.save_cache()
        return pair_scores


# Backwards compatibility aliases
l2n = normalize_l2
topk_cosine = find_top_k_cosine_neighbors
candidate_cosine = compute_candidate_cosine_scores
LocalScorer = LocalFeatureScorer
