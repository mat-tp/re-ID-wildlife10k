"""
Feature extraction and disk-caching pipeline for all model types.

Works with any model that exposes eval_transform() and embed():
- GlobalModel      (global_models.py)        - MegaDescriptor
- ComplementaryModel (complementary_models.py) - EVA02
- DINOv2Baseline   (complementary_models.py)   - DINOv2 baseline

Extracted embeddings are saved to .pt files so inference only runs once
across experiments on the 140k image gallery.
"""

import os
import torch
from torch.utils.data import DataLoader, Subset

# Reuse the existing batch extraction loop from global_models rather than duplicating it
from src.models.global_models import extract_features


# Extract embeddings for a dataset split and cache them to disk
def extract_and_cache(
    model,
    dataset,
    split: str,
    cache_path: str,
    *,
    batch_size: int = 32,
    num_workers: int = 4,
    device=None,
    amp: bool = True,
    overwrite: bool = False,
):
    # Load from cache if it already exists
    if os.path.isfile(cache_path) and not overwrite:
        print(f"[extraction] Loading cached features from {cache_path}")
        saved = torch.load(cache_path, map_location="cpu")
        return saved["feats"], (saved["labels"], saved["species"])

    # Filter dataset rows to the requested split
    split_mask = dataset.data["split"] == split
    indices = dataset.data.index[split_mask].tolist()

    if len(indices) == 0:
        raise ValueError(
            f"No rows found for split='{split}'. "
            f"Available splits: {dataset.data['split'].unique().tolist()}"
        )

    print(
        f"[extraction] Running inference on {len(indices)} images "
        f"(split='{split}') -> {cache_path}"
    )

    # Build a DataLoader over the selected subset
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=(device != "cpu"),
        shuffle=False,
    )

    # Run batch inference using the shared extraction loop
    model.eval()
    feats, extras = extract_features(model, loader, device=device, amp=amp)

    # WildlifeReID10kDataset returns (image, label, species) so extras = (labels, species)
    labels  = list(extras[0]) if extras else []
    species = list(extras[1]) if len(extras) > 1 else []

    # Save to disk
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
    torch.save(
        {"feats": feats, "labels": labels, "species": species, "split": split},
        cache_path,
    )
    print(
        f"[extraction] Saved {feats.shape[0]} embeddings "
        f"(dim={feats.shape[1]}) to {cache_path}"
    )

    return feats, (labels, species)


# Load a previously saved cache file directly
def load_cached(cache_path: str):
    if not os.path.isfile(cache_path):
        raise FileNotFoundError(
            f"Cache file not found: {cache_path}. "
            "Run extract_and_cache() first."
        )
    saved = torch.load(cache_path, map_location="cpu")
    return saved["feats"], (saved["labels"], saved["species"])
