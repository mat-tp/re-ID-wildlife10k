"""
Synthetic Demo Data Generators

Generates mock feature embeddings and local matching scores aligned with real metadata roles.
Allows the full retrieval pipeline (calibration, fusion, ablation analysis, and UI visualization)
to be tested and validated locally without requiring heavy models, image datasets, or GPU hardware.

Note:
    NUMBERS PRODUCED BY THIS MODULE ARE MEANINGLESS AND FOR DEMONSTRATION ONLY.
    Its purpose is to confirm pipeline execution, numerical stability, and branch integration.
"""

from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.retrieval.similarity import normalize_l2


def generate_demo_features(
    metadata_dataframe: pd.DataFrame,
    dimension_config: Optional[Dict[str, int]] = None,
    signal_strength_config: Optional[Dict[str, float]] = None,
    random_seed: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Generates synthetic feature matrices for specified feature branches.

    Simulates identity cluster centroids with additive Gaussian noise scaled by dataset difficulty.

    Args:
        metadata_dataframe: Pandas DataFrame containing 'identity' and 'dataset' columns.
        dimension_config: Mapping of feature branch names to feature dimensionality.
                          Defaults to {'global': 64, 'complementary': 32, 'baseline': 32}.
        signal_strength_config: Mapping of feature branch names to identity signal strengths.
                                Defaults to {'global': 1.0, 'complementary': 0.6, 'baseline': 0.4}.
        random_seed: Seed for reproducible synthetic feature generation.

    Returns:
        Dictionary mapping branch names to L2-normalized synthetic feature arrays of shape (N_samples, D).
    """
    # Set default dimension configurations if not provided
    default_dimensions = {"global": 64, "complementary": 32, "baseline": 32}
    feature_dimensions = dimension_config or default_dimensions

    # Set default signal strength configurations if not provided
    default_signal_strengths = {"global": 1.0, "complementary": 0.6, "baseline": 0.4}
    signal_strengths = signal_strength_config or default_signal_strengths

    random_generator = np.random.default_rng(random_seed)

    # Convert identity string labels into integer indices (0 to N_identities - 1)
    identity_labels = metadata_dataframe["identity"].to_numpy()
    unique_identities, identity_indices = np.unique(identity_labels, return_inverse=True)
    num_total_identities = len(unique_identities)

    # Convert dataset string labels into integer indices (0 to N_datasets - 1)
    dataset_labels = metadata_dataframe["dataset"].to_numpy()
    unique_datasets, dataset_indices = np.unique(dataset_labels, return_inverse=True)
    num_total_datasets = len(unique_datasets)

    # Assign a random difficulty scale factor to each dataset to simulate domain shifts
    dataset_difficulty_factors = random_generator.uniform(
        low=0.8, high=1.6, size=num_total_datasets
    )
    # Map dataset difficulty factors back to individual sample rows: shape (N_samples, 1)
    sample_difficulty_multipliers = dataset_difficulty_factors[dataset_indices][:, None]

    synthetic_feature_branches: Dict[str, np.ndarray] = {}
    num_samples = len(metadata_dataframe)

    for branch_name, feature_dim in feature_dimensions.items():
        branch_strength = signal_strengths.get(branch_name, 1.0)

        # Generate random ground truth identity cluster centroids: shape (N_identities, D)
        identity_centroids = random_generator.normal(
            size=(num_total_identities, feature_dim)
        ).astype(np.float32)

        # Generate zero-mean Gaussian observation noise per sample: shape (N_samples, D)
        observation_noise = random_generator.normal(
            size=(num_samples, feature_dim)
        ).astype(np.float32)

        # Synthesize feature vector: blend scaled identity centroid signal with difficulty-weighted noise
        unnormalized_features = (
            branch_strength * 2.0 * identity_centroids[identity_indices]
            + sample_difficulty_multipliers * observation_noise
        )

        # Normalize generated feature vectors to unit L2 norm
        normalized_features = normalize_l2(unnormalized_features)
        synthetic_feature_branches[branch_name] = normalized_features

    return synthetic_feature_branches


class DemoLocalScorer:
    """
    Mock replacement for LocalFeatureScorer (e.g., ALIKED + LightGlue RANSAC match count).

    Simulates local feature matching counts without requiring actual images or keypoint matchers.
    Generates Poisson-distributed mock counts (higher Poisson mean for matching identities).
    """

    def __init__(
        self, 
        metadata_dataframe: pd.DataFrame, 
        random_seed: int = 0
    ) -> None:
        """
        Args:
            metadata_dataframe: Pandas DataFrame containing ground truth identity labels.
            random_seed: Seed for reproducible Poisson score sampling.
        """
        self.identity_labels = metadata_dataframe["identity"].to_numpy()
        self.random_generator = np.random.default_rng(random_seed)

    def score_pairs(
        self,
        query_row_indices: np.ndarray,
        candidate_gallery_row_indices: np.ndarray,
    ) -> np.ndarray:
        """
        Generates synthetic keypoint match scores for candidate pair matrices.

        Args:
            query_row_indices: Array of shape (N_queries,) containing query row indices.
            candidate_gallery_row_indices: Array of shape (N_queries, K_candidates) containing candidate row indices.

        Returns:
            Array of shape (N_queries, K_candidates) with synthetic Poisson-distributed match counts in float32.
        """
        query_indices_array = np.asarray(query_row_indices)
        candidate_indices_array = np.asarray(candidate_gallery_row_indices)

        # Retrieve true identity labels for query rows: shape (N_queries, 1)
        query_identities = self.identity_labels[query_indices_array][:, None]

        # Retrieve true identity labels for candidate gallery rows: shape (N_queries, K_candidates)
        candidate_gallery_identities = self.identity_labels[candidate_indices_array]

        # Determine boolean match mask where query identity matches gallery candidate identity
        is_same_identity_mask = query_identities == candidate_gallery_identities

        # Sample Poisson counts: Mean lambda=14 for true positive pairs, lambda=3 for negative pairs
        true_positive_counts = self.random_generator.poisson(
            lam=14, size=is_same_identity_mask.shape
        )
        negative_counts = self.random_generator.poisson(
            lam=3, size=is_same_identity_mask.shape
        )

        synthetic_scores = np.where(
            is_same_identity_mask, true_positive_counts, negative_counts
        ).astype(np.float32)

        return synthetic_scores