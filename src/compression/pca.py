import numpy as np
from sklearn.decomposition import PCA

"""
    Memory reduction and compression for embedding features using PCA and L2 normalization.
    
"""

def l2_normalize(features: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """L2-normalizes feature vectors along the last axis."""
    features = np.asarray(features, dtype=np.float32)
    norms = np.maximum(np.linalg.norm(features, axis=1, keepdims=True), eps)
    return features / norms


class PCACompressor:
    """Applies PCA compression and optional L2 whitening/normalization to embedding features."""

    def __init__(self, n_components: int = 256, whiten: bool = True, seed: int = 0):
        self.n_components = n_components
        self.whiten = whiten
        self.seed = seed
        self.pca = None

    def fit(self, train_features: np.ndarray):
        """Fits the PCA model exclusively on L2-normalized training features to prevent leakage."""
        normalized_train = l2_normalize(train_features)
        
        # Ensure n_components does not exceed available dimensions or sample size - 1
        max_components = min(normalized_train.shape[1], normalized_train.shape[0] - 1)
        k = min(self.n_components, max_components)
        
        # Select efficient SVD solver based on matrix dimensions
        svd_solver = "randomized" if k < 0.8 * min(normalized_train.shape) else "full"
        
        self.pca = PCA(
            n_components=k,
            whiten=self.whiten,
            svd_solver=svd_solver,
            random_state=self.seed
        )
        self.pca.fit(normalized_train)
        return self

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transforms features using the fitted PCA model and applies L2 re-normalization."""
        if self.pca is None:
            raise ValueError("PCACompressor must be fitted before calling transform().")
        
        normalized_features = l2_normalize(features)
        projected = self.pca.transform(normalized_features)
        return l2_normalize(projected)

    def explained_variance(self) -> float:
        """Returns the cumulative explained variance ratio of the retained components."""
        if self.pca is None:
            return 0.0
        return float(self.pca.explained_variance_ratio_.sum())

    def bytes_per_vector(self, dtype=np.dtype(np.float32)) -> int:
        """Calculates storage size in bytes per compressed vector."""
        if self.pca is None:
            return 0
        return int(self.pca.n_components_ * np.dtype(dtype).itemsize)


def evaluate_dimensionality_sweep(
    train_features: np.ndarray,
    gallery_features: np.ndarray,
    query_features: np.ndarray,
    dimensions: list,
    evaluate_fn,
    whiten: bool = True
) -> list:
    """
    Evaluates retrieval performance across several target dimensionalities.
    
    Args:
        train_features: Training features used to fit PCA.
        gallery_features: Gallery dataset features.
        query_features: Query dataset features.
        dimensions: List of target dimensions to test (supports None or 'full' for unreduced).
        evaluate_fn: Callable mapping `(gallery_embeddings, query_embeddings)` -> dict of metrics.
        whiten: Whether to apply PCA whitening.
        
    Returns:
        List of dictionaries containing evaluation results and metadata per dimension.
    """
    results = []
    for dim in dimensions:
        if dim in (None, "full"):
            gallery_processed = l2_normalize(gallery_features)
            query_processed = l2_normalize(query_features)
            dim_value = gallery_features.shape[1]
            explained_var = 1.0
        else:
            compressor = PCACompressor(n_components=dim, whiten=whiten).fit(train_features)
            gallery_processed = compressor.transform(gallery_features)
            query_processed = compressor.transform(query_features)
            dim_value = dim
            explained_var = compressor.explained_variance()
            
        metrics = evaluate_fn(gallery_processed, query_processed)
        results.append({
            "dim": dim_value,
            "explained_var": explained_var,
            "bytes_per_img": dim_value * np.dtype(np.float32).itemsize,
            **metrics
        })
    return results


def concat_weighted_embeddings(embeddings: list, weights: list) -> np.ndarray:
    """
    Concatenates multiple embedding branches with weights such that the cosine similarity 
    of the concatenated vector equals the weighted sum of individual branch cosine similarities.
    
    Formula: [sqrt(w_i) * l2n(e_i)] -> cosine equals sum(w_i * cos_i).
    """
    weights_arr = np.asarray(weights, dtype=np.float64)
    weights_arr = weights_arr / weights_arr.sum()
    
    weighted_branches = [
        np.sqrt(weight) * l2_normalize(emb) 
        for weight, emb in zip(weights_arr, embeddings)
    ]
    return np.concatenate(weighted_branches, axis=1)