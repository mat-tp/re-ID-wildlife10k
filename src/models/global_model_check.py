"""
Dataset Feature Extraction Benchmark

Performs feature extraction and baseline 1-Nearest-Neighbor evaluation on a target dataset split
using global feature embeddings.

Workflow:
1. Loads the WildlifeReID-10k dataset and initializes a pretrained GlobalModel instance.
2. Extracts L2-normalized feature vectors for gallery and query splits of a target dataset.
3. Evaluates top-1 cosine similarity accuracy and query-to-gallery identity coverage.
"""

from typing import Tuple
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data_processing.dataloader import WildlifeReID10kDataset
from src.models.global_models import GlobalModel, extract_features


def extract_subset_features(
    dataset: WildlifeReID10kDataset,
    subset_metadata: pd.DataFrame,
    global_model: GlobalModel,
    max_sample_limit: int = 300,
    batch_size: int = 8,
    use_automatic_mixed_precision: bool = True,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """
    Extracts feature representations for a subset of dataset rows up to a specified limit.

    Args:
        dataset: The full WildlifeReID10kDataset instance.
        subset_metadata: Subset DataFrame corresponding to selected query/gallery rows.
        global_model: Evaluated GlobalModel instance for feature extraction.
        max_sample_limit: Maximum number of rows to sample from the subset.
        batch_size: Data loader batch size.
        use_automatic_mixed_precision: Whether to use mixed-precision CUDA execution.

    Returns:
        Tuple containing:
            - feature_matrix: Array of shape (N_samples, feature_dim) with L2-normalized features.
            - labels_tuple: Tuple containing (identity_labels, dataset_names) arrays.
    """
    # Slice target dataset index values up to the maximum sample threshold
    target_row_indices = subset_metadata.index.to_numpy()[:max_sample_limit]
    dataset_subset = Subset(dataset, target_row_indices)

    subset_data_loader = DataLoader(
        dataset_subset,
        batch_size=batch_size,
        num_workers=0,
        shuffle=False,
    )

    feature_matrix, (identity_labels, dataset_names) = extract_features(
        model=global_model,
        loader=subset_data_loader,
        device=None,
        amp=use_automatic_mixed_precision,
    )
    return feature_matrix, (identity_labels, dataset_names)


def main() -> None:
    """Main execution function for feature extraction and benchmarking."""
    target_dataset_name = "AerialCattle2017"

    # Initialize model in evaluation mode
    global_model = GlobalModel().eval()

    # Load dataset using model's dedicated evaluation preprocessing pipeline
    reid_dataset = WildlifeReID10kDataset(
        metadata_path="data/metadata.csv",
        root_dir="data",
        transform=global_model.eval_transform(),
    )

    # Print summary of overall data split distribution
    print("Overall Data Split Counts:")
    print(reid_dataset.data["split"].value_counts())
    print("\nEight Smallest Datasets by Sample Count:")
    print(reid_dataset.data["dataset"].value_counts().tail(8))

    # Filter metadata for the specific benchmark dataset
    target_subset_metadata = reid_dataset.data[
        reid_dataset.data["dataset"] == target_dataset_name
    ]
    print(f"\nSplit distribution for dataset '{target_dataset_name}':")
    print(target_subset_metadata["split"].value_counts())

    # Filter metadata into gallery (train) and query (test) splits
    gallery_metadata = target_subset_metadata[
        target_subset_metadata["split"] == "train"
    ]
    query_metadata = target_subset_metadata[
        target_subset_metadata["split"] == "test"
    ]

    start_time = time.time()

    # Extract features for gallery and query splits
    gallery_features, (gallery_identities, _) = extract_subset_features(
        dataset=reid_dataset,
        subset_metadata=gallery_metadata,
        global_model=global_model,
        max_sample_limit=300,
    )

    query_features, (query_identities, _) = extract_subset_features(
        dataset=reid_dataset,
        subset_metadata=query_metadata,
        global_model=global_model,
        max_sample_limit=300,
    )

    elapsed_seconds = time.time() - start_time
    print(
        f"\nExtracted features for {len(gallery_features)} gallery samples and "
        f"{len(query_features)} query samples in {elapsed_seconds:.0f}s"
    )

    # Compute cosine similarity matrix between query features and gallery features: shape (N_query, N_gallery)
    similarity_matrix = query_features @ gallery_features.T

    # Retrieve nearest gallery neighbor identity for each query vector
    nearest_neighbor_indices = similarity_matrix.argmax(axis=1).tolist()
    predicted_query_identities = [
        gallery_identities[neighbor_idx] for neighbor_idx in nearest_neighbor_indices
    ]

    # Calculate metrics: top-1 1NN accuracy and gallery identity coverage
    top1_accuracy = np.mean(
        [
            predicted_label == ground_truth_label
            for predicted_label, ground_truth_label in zip(
                predicted_query_identities, query_identities
            )
        ]
    )

    unique_gallery_identities = set(gallery_identities)
    query_identity_coverage = np.mean(
        [query_label in unique_gallery_identities for query_label in query_identities]
    )

    print(f"Top-1 Accuracy: {top1_accuracy:.3f}")
    print(f"Fraction of query identities present in gallery: {query_identity_coverage:.3f}")


if __name__ == "__main__":
    main()